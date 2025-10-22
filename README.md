# Introvert vs Extrovert — Personality Classifier Web App

A machine-learning web application that predicts whether a person is an **Introvert** or **Extrovert** based on behavioural and social-interaction features.  
Built with Python, a classification model (e.g., Random Forest/XGBoost), and a Streamlit frontend.

---


##  Installation

### 1. Clone the repository  
```bash
git clone https://github.com/Sameershahh/Introvert-vs-Extrovert.git  
cd Introvert-vs-Extrovert
```  

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Running the web app
```bash
streamlit run app.py
```
## Model Architecture
- Model type: (e.g., Random Forest, XGBoost) classifier
- Input features: behavioural and social-interaction metrics (example: hours alone/day, outgoing event frequency, social-media post count)
- Train/Test split (e.g., 80 % train, 20 % test)
- Evaluation metrics: accuracy, confusion matrix, classification report
- Saved artefacts: model file (e.g., model.pkl or xgb_model.pkl) + feature scaler (e.g., scaler.pkl)
- Explainability: optional SHAP plots or feature-importance visualisations


## Example Query
#### Input Feature
Time_spent_alone = 7 hrs, Social_event_attendance = 2/week, Post_frequency = 1/week
#### Prediction
Introvert (Probability: 0.87)

## Author
**Sameer Shah** — AI & Full-Stack Developer  
[Portfolio](https://sameershah-portfolio.vercel.app/) 
