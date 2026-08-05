# 📧 Gmail Spam Detection using Naive Bayes

## 📌 Overview

This project integrates the Gmail API with a Naive Bayes Machine Learning model to automatically detect spam emails. It securely connects to a Gmail account using OAuth 2.0, reads recent emails, extracts the subject and body, and predicts whether each email is Spam or Not Spam along with a confidence score.

## ✨ Features

- Secure Gmail authentication using OAuth 2.0
- Reads the latest emails from Gmail
- Extracts email subject and body
- Cleans and preprocesses email text
- Uses a trained Naive Bayes classifier
- Predicts Spam or Not Spam
- Displays prediction confidence
- Easy to extend for real-time email filtering

## 🛠️ Technologies Used

- Python
- Gmail API
- Google OAuth 2.0
- Scikit-learn
- Pandas
- Joblib
- Regular Expressions (Regex)

## 📂 Project Structure

```
project/
│── email_spam.py
│── credentials.json
│── token.pickle
│── spam_model.joblib
│── vectorizer.joblib
```

## 🚀 How to Run

1. Clone this repository.
2. Install the required libraries:

```bash
pip install google-api-python-client google-auth google-auth-oauthlib scikit-learn pandas joblib
```

3. Download your Gmail API OAuth credentials from Google Cloud Console and save them as `credentials.json`.

4. Place your trained model (`spam_model.joblib`) and vectorizer (`vectorizer.joblib`) in the project folder.

5. Run the project:

```bash
python email_spam.py
```

6. Sign in with your Google account when prompted.

## 📊 Output

For each email, the program displays:

- Email Subject
- Spam / Not Spam Prediction
- Confidence Score

Example:

```
Subject: Congratulations! You won a prize

Prediction: Spam
Confidence: 98.74%

--------------------------------------

Subject: Project Meeting Tomorrow

Prediction: Not Spam
Confidence: 99.12%
```

## 🔮 Future Improvements

- Real-time spam monitoring
- Automatic spam labeling in Gmail
- Email attachment analysis
- Web dashboard for monitoring predictions
- Deep Learning-based spam detection

## 📜 License

This project is for learning and educational purposes.
