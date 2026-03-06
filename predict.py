import pandas as pd
from model import PhishingModel
from utils import extract_features_from_text
import config

def predict_email(email_text, subject=""):
    try:
        model = PhishingModel.load(config.MODEL_PATH)
        
        # Extract features
        features = extract_features_from_text(email_text, subject)
        features['email_text'] = email_text
        
        df = pd.DataFrame([features])
        
        pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0]
        
        confidence = proba[pred] * 100
        
        return {
            "prediction": "Phishing" if pred == 1 else "Legitimate",
            "confidence": f"{confidence:.2f}%",
            "features_extracted": features
        }
    except Exception as e:
        return {"error": str(e), "message": "Make sure model is trained first."}

if __name__ == "__main__":
    test_email = "URGENT: Your account will be suspended! Click here to verify your identity: http://bit.ly/malicious-link"
    test_subject = "Action Required"
    print(predict_email(test_email, test_subject))
