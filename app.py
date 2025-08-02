import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer

# PhishingDetector class directly included in app.py to avoid import errors
class PhishingDetector:
    def __init__(self, model_path=None):
        self.tokenizer = RegexpTokenizer(r'[A-Za-z]+')
        self.stemmer = SnowballStemmer("english")
        self.vectorizer = None
        self.model = None
        
        if model_path:
            self.load_model(model_path)
    
    def preprocess_url(self, url):
        # Tokenize the URL
        tokens = self.tokenizer.tokenize(url)
        # Stem the tokens
        stemmed = [self.stemmer.stem(word) for word in tokens]
        # Join the stemmed tokens
        processed = ' '.join(stemmed)
        return processed
    
    def predict(self, url):
        """
        Predict if a URL is a phishing site
        Returns: probability of being a phishing site (0 to 1)
        """
        if not self.model or not self.vectorizer:
            raise ValueError("Model or vectorizer not loaded. Call load_model() first.")
            
        # For direct URL prediction without preprocessing
        vectorized_url = self.vectorizer.transform([url])
        
        # Predict probability (second column is the probability of being a phishing site)
        prediction_proba = self.model.predict_proba(vectorized_url)[0, 1]
        prediction = self.model.predict(vectorized_url)[0]
        
        return {
            "url": url,
            "is_phishing": bool(prediction),
            "probability": float(prediction_proba),
            "status": "Phishing" if prediction == 1 else "Legitimate"
        }
    
    def load_model(self, path="phishing_model.pkl"):
        """
        Load a trained model and vectorizer from a file
        """
        try:
            with open(path, 'rb') as f:
                # Try loading as dictionary containing both model and vectorizer
                try:
                    loaded = pickle.load(f)
                    if isinstance(loaded, dict) and 'model' in loaded and 'vectorizer' in loaded:
                        self.model = loaded['model']
                        self.vectorizer = loaded['vectorizer']
                        print("Loaded model and vectorizer from dictionary")
                    else:
                        # If it's not a dictionary with the expected keys, assume it's just the model
                        self.model = loaded
                        # Create a basic vectorizer that won't work properly
                        self.vectorizer = CountVectorizer()
                        print("Warning: Only model loaded, vectorizer missing. Predictions may fail.")
                except Exception as e:
                    print(f"Error unpacking model: {e}")
                    return False
                
                return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

# Page configuration
st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🔒",
    layout="centered"
)

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load the phishing detector model
@st.cache_resource
def load_model():
    detector = PhishingDetector()
    detector.load_model("phishing_model.pkl")
    return detector

try:
    detector = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    model_loaded = False

# App header
st.title("🔒 Phishing URL Detector")
st.markdown("Detect potential phishing websites with machine learning.")

# URL input section
with st.container():
    url = st.text_input("Enter a URL to analyze:", "example.com")
    analyze_button = st.button("Analyze URL")

# Results section
if analyze_button and model_loaded:
    if not url:
        st.warning("Please enter a URL")
    else:
        with st.spinner("Analyzing URL..."):
            try:
                result = detector.predict(url)
                
                # Display results in two columns
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    st.subheader("Analysis Result")
                    
                    # Calculate safety score (inverse of phishing probability)
                    safety_score = 100 - int(result['probability'] * 100)
                    
                    # Determine color based on safety score
                    if safety_score > 75:
                        color = "green"
                        status = "Likely Safe"
                    elif safety_score > 40:
                        color = "orange"
                        status = "Suspicious"
                    else:
                        color = "red"
                        status = "Likely Phishing"
                    
                    # Show gauge chart for safety score
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=safety_score,
                        title={'text': "Safety Score"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': color},
                            'steps': [
                                {'range': [0, 40], 'color': "lightcoral"},
                                {'range': [40, 75], 'color': "lightyellow"},
                                {'range': [75, 100], 'color': "lightgreen"}
                            ]
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig)
                    
                    # URL info
                    st.info(f"URL: {url}")
                    
                    # Result message
                    if result['is_phishing']:
                        st.error(f"Status: {status} (Confidence: {result['probability']:.2f})")
                        st.warning("⚠️ This URL appears to be suspicious. Be cautious!")
                    else:
                        st.success(f"Status: {status} (Confidence: {1-result['probability']:.2f})")
                        st.info("✅ This URL appears to be legitimate.")
                
                with col2:
                    st.subheader("Safety Tips")
                    st.markdown("""
                    ### Phishing Warning Signs:
                    
                    1. **Misspelled domain names**
                    2. **Unusual TLDs** (.tk, .xyz)
                    3. **Numbers and hyphens** in URLs
                    4. **Missing https://**
                    5. **URL shorteners**
                    
                    Always verify the sender and website before sharing personal information!
                    """)
            
            except Exception as e:
                st.error(f"Error during analysis: {e}")

# Information section
st.markdown("---")
st.subheader("About Phishing")
st.markdown("""
Phishing is a cybercrime where attackers impersonate legitimate organizations via email, text, or fake websites to steal sensitive data.

**Common phishing targets:**
- Login credentials
- Credit card numbers
- Bank account information
- Social security numbers
""")

# Footer
st.markdown("---")
st.caption("© 2023 Phishing URL Detector | Made with Streamlit")

# Example URLs to test
st.markdown("### Try these example URLs:")
cols = st.columns(2)
with cols[0]:
    if st.button("google.com"):
        st.session_state.url = "google.com"
        st.experimental_rerun()
    if st.button("paypal-secure.verifyaccount.tk"):
        st.session_state.url = "paypal-secure.verifyaccount.tk"
        st.experimental_rerun()
with cols[1]:
    if st.button("facebook.com"):
        st.session_state.url = "facebook.com"
        st.experimental_rerun()
    if st.button("secure-banking.com.suspicious.ru"):
        st.session_state.url = "secure-banking.com.suspicious.ru"
        st.experimental_rerun()
