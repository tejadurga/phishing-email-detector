import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

class PhishingModel:
    def __init__(self):
        self.text_transformer = Pipeline(steps=[
            ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english'))
        ])
        
        self.preprocessor = None
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        
        self.model = VotingClassifier(
            estimators=[('rf', rf), ('xgb', xgb)],
            voting='soft'
        )
        self.pipeline = None

    def build_pipeline(self, text_feature_name, numeric_feature_names):
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('text', self.text_transformer, text_feature_name),
                ('num', 'passthrough', numeric_feature_names)
            ]
        )
        self.pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', self.model)
        ])

    def train(self, X_train, y_train):
        self.pipeline.fit(X_train, y_train)

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def save(self, filepath):
        joblib.dump(self.pipeline, filepath)

    @classmethod
    def load(cls, filepath):
        pipeline = joblib.load(filepath)
        instance = cls()
        instance.pipeline = pipeline
        return instance
