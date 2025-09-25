from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import joblib

model = joblib.load("personality_model.pkl")

# Input schema (raw fields only)
class PersonalityInput(BaseModel):
    Time_spent_Alone: int
    Stage_fear: int
    Social_event_attendance: int
    Going_outside: int
    Drained_after_socializing: str
    Friends_circle_size: int
    Post_frequency: int

app = FastAPI(title="Personality Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # your Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Welcome to Personality Predictor API"}

@app.post("/predict")
def prediction(data: PersonalityInput):
    # Convert to DataFrame
    df = pd.DataFrame([data.dict()])

    # --- Recreate engineered features (same as Colab training) ---
    df["Post_per_friend"] = df["Post_frequency"] / (df["Friends_circle_size"] + 1)
    df["Engagement"] = df["Social_event_attendance"] + df["Going_outside"]
    df["Social_vs_Alone"] = df["Social_event_attendance"] / (df["Time_spent_Alone"] + 1)
    df["Drained_after_socializing_bin"] = df["Drained_after_socializing"].map({"Yes": 1, "No": 0})
    df["Energy_balance"] = df["Going_outside"] - df["Time_spent_Alone"]

    # --- Prediction ---
    pred = model.predict(df)[0]
    label = "Extrovert" if pred == 1 else "Introvert"

    return {"prediction": label}
