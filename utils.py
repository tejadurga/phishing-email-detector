import re
from bs4 import BeautifulSoup
import tldextract

def extract_features_from_text(text, subject="", sender=""):
    features = {}
    
    # URL/Link Count
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    features['links_count'] = len(urls)
    
    # Suspicious Domains
    suspicious_domains = 0
    for url in urls:
        ext = tldextract.extract(url)
        if ext.domain in ['bit', 'tinyurl', 'ow', 'is']:
            suspicious_domains += 1
    features['suspicious_urls'] = suspicious_domains
    
    # Attachments
    features['has_attachment'] = 1 if 'attachment' in text.lower() or 'attached' in text.lower() else 0
    
    # Urgent keywords
    urgent_keywords = ['urgent', 'immediate', 'account suspended', 'verify account', 'click here', 'action required']
    features['urgent_keywords'] = sum(1 for kw in urgent_keywords if kw in text.lower() or kw in subject.lower())
    
    # HTML contents
    soup = BeautifulSoup(text, 'html.parser')
    features['html_tags'] = len(soup.find_all())
    
    # Subject features
    features['subject_length'] = len(subject)
    features['caps_ratio'] = sum(1 for c in subject if c.isupper()) / max(1, len(subject))
    features['exclamation_marks'] = subject.count('!')
    
    return features
