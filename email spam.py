"""
Real-Time Gmail Spam Detection using Gmail API + Multinomial Naive Bayes
--------------------------------------------------------------------------
What this script does:
  1. Authenticates with your Gmail account using OAuth 2.0
  2. Fetches your latest emails
  3. Extracts the subject + body from each one
  4. Trains a Naive Bayes spam classifier (TF-IDF + MultinomialNB)
  5. Classifies each fetched email as Spam / Not Spam, with a confidence score

BEFORE YOU RUN THIS:
  1. Go to console.cloud.google.com -> create a project -> enable "Gmail API"
  2. Go to APIs & Services -> Credentials -> Create OAuth Client ID -> Desktop app
  3. Download the file, rename it to "credentials.json", and place it in this
     same folder as this script.
  4. Install the required packages:
       pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client pandas scikit-learn
  5. Make sure "completeSpamAssassin.csv" is also in this same folder
     (or update CSV_PATH below to point to it).

The first time you run this, a browser window will pop up asking you to log
in to Gmail and approve access. After that, a "token.json" file is saved so
you won't have to log in again.
"""

import os
import base64
import pandas as pd

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# Read-only access to Gmail is all we need
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

CREDENTIALS_FILE = "credentials.json"   # downloaded from Google Cloud Console
TOKEN_FILE = "token.json"               # auto-created after first login
CSV_PATH = "completeSpamAssassin.csv"   # dataset used to train the model
MAX_EMAILS = 10                         # how many recent emails to check


# ---------------------------------------------------------------------------
# STEP 1: Authenticate with Gmail using OAuth 2.0
# ---------------------------------------------------------------------------

def get_gmail_service():
    creds = None

    # Reuse saved login if we have one
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    # Otherwise, log in through the browser
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)

        # Save the login so we don't have to repeat this next time
        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


# ---------------------------------------------------------------------------
# STEP 2: Fetch latest emails and extract subject + body
# ---------------------------------------------------------------------------

def get_email_body(payload):
    """Extracts the plain text body from a Gmail message payload."""
    body = ""

    if "parts" in payload:
        for part in payload["parts"]:
            if part.get("mimeType") == "text/plain" and "data" in part.get("body", {}):
                data = part["body"]["data"]
                body = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
                break
            elif "parts" in part:  # nested multipart (e.g. multipart/alternative)
                body = get_email_body(part)
                if body:
                    break
    else:
        data = payload.get("body", {}).get("data")
        if data:
            body = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")

    return body


def fetch_latest_emails(service, max_results=MAX_EMAILS):
    results = service.users().messages().list(
        userId="me", maxResults=max_results, labelIds=["INBOX"]
    ).execute()

    messages = results.get("messages", [])
    emails = []

    for msg in messages:
        full_msg = service.users().messages().get(
            userId="me", id=msg["id"], format="full"
        ).execute()

        headers = full_msg["payload"].get("headers", [])
        subject = next((h["value"] for h in headers if h["name"] == "Subject"), "(no subject)")
        sender = next((h["value"] for h in headers if h["name"] == "From"), "(unknown sender)")

        body = get_email_body(full_msg["payload"])

        emails.append({
            "subject": subject,
            "from": sender,
            "body": body
        })

    return emails


# ---------------------------------------------------------------------------
# STEP 3: Train the Naive Bayes spam classifier
# ---------------------------------------------------------------------------

def train_model():
    df = pd.read_csv(CSV_PATH)
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in c], errors="ignore")
    df = df.dropna(subset=["Body"])
    df["Body"] = df["Body"].astype(str)

    X = df["Body"]
    y = df["Label"]

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.20, random_state=42, stratify=y
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Model trained. Test accuracy: {accuracy:.4f}\n")

    return model, vectorizer


# ---------------------------------------------------------------------------
# STEP 4: Classify an email and return label + confidence
# ---------------------------------------------------------------------------

def classify_email(text, model, vectorizer):
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    probabilities = model.predict_proba(vec)[0]

    confidence = probabilities[prediction] * 100
    label = "Spam" if prediction == 1 else "Not Spam"

    return label, confidence


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("Training spam classifier...")
    model, vectorizer = train_model()

    print("Connecting to Gmail...")
    service = get_gmail_service()

    print(f"Fetching your latest {MAX_EMAILS} emails...\n")
    emails = fetch_latest_emails(service, MAX_EMAILS)

    print("=" * 70)
    for i, email in enumerate(emails, start=1):
        combined_text = email["subject"] + " " + email["body"]
        label, confidence = classify_email(combined_text, model, vectorizer)

        print(f"Email {i}")
        print(f"  From      : {email['from']}")
        print(f"  Subject   : {email['subject']}")
        print(f"  Prediction: {label}  (confidence: {confidence:.2f}%)")
        print("-" * 70)


if __name__ == "__main__":
    main()