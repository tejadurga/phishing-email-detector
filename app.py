from flask import Flask, render_template, request, jsonify
import pandas as pd
from predict import predict_email

app = Flask(__name__)

scan_history = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.is_json:
        data = request.get_json()
        email_text = data.get('email_text', '')
        subject = data.get('subject', '')
    else:
        email_text = request.form.get('email_text', '')
        subject = request.form.get('subject', '')
        
    if not email_text:
        return jsonify({"error": "No email text provided"}), 400
        
    result = predict_email(email_text, subject)
    
    scan_history.append({
        "subject": subject,
        "prediction": result.get("prediction"),
        "confidence": result.get("confidence")
    })
    
    if request.is_json:
        return jsonify(result)
        
    return render_template('results.html', result=result, email_text=email_text, subject=subject)

@app.route('/history', methods=['GET'])
def history():
    return jsonify(scan_history)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
