# Phishing URL Detection System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.10+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-green.svg)

A machine learning-powered web application for detecting phishing URLs in real-time to enhance cybersecurity and protect users from online threats.

## 🔍 Problem Statement

Phishing attacks are one of the most common and dangerous cyber threats today. Attackers create fake websites that mimic legitimate ones to steal sensitive information such as login credentials, credit card numbers, and personal data.

Key challenges this project addresses:
- Identifying potentially malicious URLs before users visit them
- Distinguishing between legitimate and phishing websites using URL characteristics
- Providing real-time analysis and risk assessment of web addresses
- Educating users about phishing threats and prevention methods

## 🛠️ Approach & Methodology

### Data Collection and Preprocessing
- Used a dataset of labeled phishing and legitimate URLs
- Applied text preprocessing techniques including:
  - Tokenization using RegexpTokenizer to extract meaningful components from URLs
  - Stemming with SnowballStemmer to normalize words
  - Feature extraction using CountVectorizer to convert URL text into machine-readable features

### Model Development
- Implemented a Logistic Regression classifier for binary classification
- Applied train-test split (80/20) to ensure proper validation
- Evaluated multiple metrics including accuracy, precision, recall, and ROC curve

### Web Application
- Built an interactive Streamlit web app for real-time URL analysis
- Implemented a safety scoring system with visual indicators
- Added educational content to inform users about phishing threats

## 📊 Results

- Model achieved **96% accuracy** in distinguishing between legitimate and phishing URLs
- Balanced performance with high precision and recall for both classes
- ROC-AUC score above 0.95, indicating excellent classification ability
- Real-time analysis capability with fast prediction speed

## 🚀 Getting Started

### Environment Setup

#### Option 1: Using Conda (Recommended)
```bash
# Create a new conda environment
conda create -n phishing-detection python=3.9
conda activate phishing-detection

# Install requirements
pip install -r requirements.txt
```

#### Option 2: Using Python Virtual Environment
```bash
# Create a virtual environment
python -m venv .venv

# Activate the environment
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### Running the Application

1. Ensure the `phishing_model.pkl` file is in the project directory
2. Launch the application:
   ```bash
   streamlit run app.py
   ```
3. Access the web interface in your browser (typically at http://localhost:8501)
4. Enter any URL to analyze its risk level

## 📂 Project Structure

```
phishing-detection-project/
├── app.py                      # Streamlit web application
├── main.py                     # PhishingDetector class and core functionality
├── Phishing-Detection.ipynb    # Notebook with model development process
├── phishing_model.pkl          # Saved model and vectorizer
├── requirements.txt            # Package dependencies
└── README.md                   # Project documentation
```

## 🔗 How It Works

1. **Input**: User enters a URL for analysis
2. **Processing**: 
   - URL is vectorized using the same feature extraction process used in training
   - Model analyzes the vectorized URL for phishing patterns
3. **Output**:
   - Safety score indicating risk level
   - Classification result (Safe/Suspicious/Phishing)
   - Visual indicators and educational information

## 🔮 Future Improvements

- Implement additional machine learning models for comparison
- Add feature importance analysis to explain why a URL is classified as phishing
- Create a browser extension for real-time protection
- Implement periodic model retraining with new data


