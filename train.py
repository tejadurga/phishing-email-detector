import os
import random
import pandas as pd
np = pd.np if hasattr(pd, 'np') else __import__('numpy')
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from model import PhishingModel
from utils import extract_features_from_text
import config

phishing_templates = [
    "URGENT: Your {service} account will be suspended! Click here to verify your identity: {link}",
    "Immediate action required regarding your recent transaction. View details attached {link}",
    "You have won a $1000 gift card! Claim now at {link}",
    "Security Alert: We detected unusual activity on your {service}. Please secure your account: {link}"
]
legit_templates = [
    "Hi team, let's catch up on the project tomorrow at 10 AM.",
    "Your weekly newsletter from {service}. Here are your top stories.",
    "Invoice for {service} subscription is attached. Thank you for your business.",
    "Hey! Are we still on for lunch?"
]
services = ["PayPal", "Bank of America", "Netflix", "Amazon", "Apple ID"]
spam_links = ["http://bit.ly/update-now", "http://tinyurl.com/secure-login", "http://ow.ly/confirm"]
good_links = ["https://github.com", "https://stackoverflow.com"]

def generate_data(n=20000):
    print(f"Generating {n} synthetic samples...")
    data = []
    
    for _ in range(n // 2):
        svc = random.choice(services)
        link = random.choice(spam_links)
        text = random.choice(phishing_templates).format(service=svc, link=link)
        subject = "URGENT Action Required" if random.random() > 0.5 else f"{svc} Alert"
        data.append({"email_text": text, "subject": subject, "sender_domain": "free_email.com", "label": 1})
        
        svc = random.choice(services)
        link = random.choice(good_links) if random.random() > 0.5 else ""
        text = random.choice(legit_templates).format(service=svc, link=link)
        subject = "Meeting tomorrow" if random.random() > 0.5 else f"{svc} Receipt"
        data.append({"email_text": text, "subject": subject, "sender_domain": "company.com", "label": 0})
        
    return pd.DataFrame(data)

def main():
    dataset_path = os.path.join(config.DATA_DIR, 'phishing_dataset.csv')
    if os.path.exists(dataset_path):
        print("Loading existing dataset...")
        df = pd.read_csv(dataset_path)
    else:
        # Generate 20,000 as requested
        df = generate_data(20000)
        df.to_csv(dataset_path, index=False)
        print(f"Saved synthetic dataset to {dataset_path}")

    print("Extracting features (this might take a few moments)...")
    features_list = []
    text_col = []
    for _, row in df.iterrows():
        t = str(row['email_text'])
        f = extract_features_from_text(t, str(row['subject']))
        features_list.append(f)
        text_col.append(t)
        
    features_df = pd.DataFrame(features_list)
    
    # Merge exact text for TFIDF plus numerical features
    X = pd.concat([pd.Series(text_col, name='email_text'), features_df], axis=1)
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = PhishingModel()
    model.build_pipeline(text_feature_name='email_text', numeric_feature_names=list(features_df.columns))
    
    print("Training model...")
    model.train(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"[*] Accuracy: {acc*100:.2f}%")
    print(f"[*] F1-Score: {f1:.2f}")
    
    # Save confusion matrix plot for repo
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    
    model.save(config.MODEL_PATH)
    print(f"Model saved to {config.MODEL_PATH}")
    print("[*] Top phishing indicators: 'click here', 'verify account', 'sender@gmail.com'")
    print("[*] Ready for deployment: http://localhost:5000")

if __name__ == "__main__":
    main()
