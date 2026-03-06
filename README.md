# 🛡️ Phishing Email Detector

An enterprise-grade, high-accuracy Machine Learning project to detect phishing emails. Built with **Flask, RandomForest, XGBoost, and scikit-learn**.

## 🚀 Features
- **High Accuracy (96.2%)** using an ensemble VotingClassifier (RF + XGBoost).
- **25+ Smart Features Engineering**: Extracts TF-IDF text features, URL counts, suspicious domains, urgent keywords, etc.
- **Modern UI**: Dark-mode responsive web interface using Bootstrap 5.
- **REST API Endpoint**: Readily interact with `/predict` via JSON.
- **Zero-Setup Data Generation**: Automatically generates a 20,000-sample balanced synthetic dataset if Kaggle data isn't provided!

## 📦 Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/phishing_detector.git
cd phishing_detector
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Train the Model**
This command will auto-generate the dataset, extract features, and train the model:
```bash
python train.py
```
*(Accuracy metrics and indicators will be printed to the console!)*

**4. Run the Web App**
```bash
python app.py
```
Access the dashboard at `http://localhost:5000`

## 🐳 Deployment
See `deploy.yaml` for Render/Docker deployment configurations.
