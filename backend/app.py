
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import MODEL_PATH
from src.data_preprocessing import clean_text


# =====================================
# 1. CREATE FASTAPI APPLICATION
# =====================================

app = FastAPI(
    title="Movie Emotion Prediction API",
    description="Predict emotions from movie plot descriptions",
    version="1.0.0"
)


# =====================================
# 2. CONFIGURE CORS
# =====================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================
# 3. LOAD TRAINED MODEL
# =====================================

artifact = None

if MODEL_PATH.exists():
    artifact = joblib.load(MODEL_PATH)
    print("Trained model loaded successfully!")
else:
    print(f"Model not found: {MODEL_PATH}")


# =====================================
# 4. REQUEST DATA STRUCTURE
# =====================================

class PredictionRequest(BaseModel):

    plot: str = Field(min_length=5)

    top_k: int = Field(default=5, ge=1, le=20)

    threshold: float = Field(default=0.35, ge=0, le=1)


# =====================================
# 5. HOME ENDPOINT
# =====================================

@app.get("/")
def home():

    return {
        "message": "Movie Emotion Prediction API is running",
        "model_loaded": artifact is not None
    }


# =====================================
# 6. HEALTH CHECK ENDPOINT
# =====================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model_loaded": artifact is not None
    }


# =====================================
# 7. EMOTION PREDICTION ENDPOINT
# =====================================

@app.post("/predict")
def predict_emotions(request: PredictionRequest):

    if artifact is None:
        raise HTTPException(
            status_code=503,
            detail="Trained model not found. Train the model first."
        )

    try:

        # Get model and labels
        model = artifact["pipeline"]

        labels = artifact["labels"]

        # Clean user input
        cleaned_plot = clean_text(request.plot)

        if not cleaned_plot:
            raise HTTPException(
                status_code=400,
                detail="Please enter a valid movie plot."
            )

        # Get emotion probabilities
        probabilities = model.predict_proba(
            [cleaned_plot]
        )[0]

        # Sort emotions by probability
        ranked_results = sorted(
            zip(labels, probabilities),
            key=lambda item: item[1],
            reverse=True
        )

        # Filter using threshold
        predictions = [
            (label, float(score))
            for label, score in ranked_results
            if score >= request.threshold
        ]

        # Return top predictions if none pass threshold
        if not predictions:

            predictions = [
                (label, float(score))
                for label, score in ranked_results[:request.top_k]
            ]

        # Limit result count
        predictions = predictions[:request.top_k]

        # Format results
        results = []

        for emotion, score in predictions:

            results.append({

                "emotion": emotion.replace("f2_", "").title(),

                "score": round(score, 4),

                "percentage": round(score * 100, 2)

            })

        return {

            "success": True,

            "total_predictions": len(results),

            "predictions": results

        }

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
