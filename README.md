# MovieEmotionModel
# 🎬 Movie Emotion Prediction System

A machine learning web application that analyzes movie plot descriptions and predicts multiple emotions using Natural Language Processing (NLP) and Multi-Label Text Classification.

The application combines a Python-based machine learning model, FastAPI backend, and React frontend to provide an interactive movie emotion prediction experience.

## ✨ Features

- Predict multiple emotions from movie plot descriptions.
- Analyze movie stories using Natural Language Processing.
- Multi-label emotion classification.
- TF-IDF text feature extraction.
- One-vs-Rest Logistic Regression classification.
- Saved machine learning model using Joblib.
- REST API integration using FastAPI.
- Interactive React frontend.
- Display ranked emotion predictions and scores.
- Model evaluation using Precision, Recall, F1-score, and Hamming Loss.

## 🏗️ System Architecture

The application consists of three main components:

1. **React Frontend:** Allows users to enter movie plots and view predictions.
2. **FastAPI Backend:** Receives movie plots and communicates with the trained model.
3. **Machine Learning Model:** Analyzes the text and predicts relevant emotions.

### Prediction Workflow

```text
User enters movie plot
          |
          v
    React Frontend
          |
          v
     FastAPI Backend
          |
          v
    Text Preprocessing
          |
          v
      TF-IDF
          |
          v
 One-vs-Rest Logistic Regression
          |
          v
   Emotion Predictions
          |
          v
    React Results UI
```

## 🧠 Machine Learning Approach

This project uses **Multi-Label Text Classification** because a single movie can contain multiple emotions.

### Algorithms and Techniques

| Technology | Purpose |
|---|---|
| TF-IDF | Converts text into numerical features |
| Logistic Regression | Classifies emotion categories |
| One-vs-Rest | Supports multiple emotion predictions |
| Scikit-learn Pipeline | Combines text vectorization and classification |
| Joblib | Saves and loads the trained model |

## 📁 Project Structure

```text
MovieEmotionModel/
│
├── backend/
│   ├── app.py
│   └── requirements.txt
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   ├── PredictionForm.jsx
│   │   │   └── EmotionResults.jsx
│   │   ├── App.jsx
│   │   ├── App.css
│   │   └── main.jsx
│   └── package.json
│
├── data/
│   ├── movie_feelings_dataset.csv
│   └── synonyms_of_feelings.csv
│
├── models/
│   └── emotion_model.joblib
│
├── reports/
│   ├── cleaned_movie_dataset.csv
│   ├── label_counts.csv
│   ├── classification_report.csv
│   └── evaluation.json
│
├── src/
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
│
├── requirements.txt
└── README.md
```

## 🛠️ Technologies Used

**Machine Learning and Backend**
- Python
- Pandas
- NumPy
- Scikit-learn
- FastAPI
- Uvicorn
- Joblib

**Frontend**
- React.js
- JavaScript
- Vite
- HTML5
- CSS3

**Development Tools**
- Visual Studio Code
- Git
- GitHub
- PowerShell

## 💻 Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd MovieEmotionModel
```

### 2. Create a Python Virtual Environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Python Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install fastapi "uvicorn[standard]"
```

### 4. Install Frontend Dependencies

```powershell
cd frontend
npm install
```

## 🧹 Dataset Preparation

Place both CSV files inside the `data` directory:

- `movie_feelings_dataset.csv`
- `synonyms_of_feelings.csv`

The preprocessing process handles:

- Missing movie plot values.
- Duplicate movie plots.
- Text normalization.
- Emotion label conversion.
- Constant emotion label removal.
- Synonym vocabulary loading.

Run the preprocessing script from the project root:

```powershell
python -m src.data_preprocessing
```

The baseline model uses the `f2_*` columns as training labels.

## 🤖 Train the Machine Learning Model

Run:

```powershell
python -m src.train
```

The training process:

1. Loads the datasets.
2. Cleans movie plot descriptions.
3. Extracts emotion labels.
4. Splits the dataset into training and testing sets.
5. Converts text into TF-IDF features.
6. Trains Logistic Regression classifiers.
7. Evaluates the trained model.
8. Saves the model using Joblib.

The trained model is saved at:

```text
models/emotion_model.joblib
```

## 🚀 Run the Backend

From the project root:

```powershell
python -m uvicorn backend.app:app --reload
```

Backend URL:

http://127.0.0.1:8000

API documentation:

http://127.0.0.1:8000/docs

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API welcome message |
| GET | `/health` | Check backend and model status |
| POST | `/predict` | Predict emotions from movie plots |

### Example API Request

```json
{
  "plot": "A young man falls in love but faces many challenges in his life.",
  "top_k": 5,
  "threshold": 0.35
}
```

## ⚛️ Run the React Frontend

Open a second terminal:

```powershell
cd frontend
npm run dev
```

Frontend URL:

http://localhost:5173

Keep both the frontend and backend running while using the application.

## 📊 Model Evaluation

Initial baseline training results:

| Evaluation Metric | Result |
|---|---:|
| Micro-F1 | 0.2366 |
| Macro-F1 | 0.0674 |
| Weighted-F1 | 0.1946 |
| Micro-Precision | 0.4217 |
| Micro-Recall | 0.1644 |
| Hamming Loss | 0.0707 |

These are initial experimental results. Further model improvements are required.

Run the evaluation script:

```powershell
python -m src.evaluate
```

Generated reports are stored in the `reports` directory.

## 🚧 Future Improvements

- Improve model performance and emotion classification accuracy.
- Handle imbalanced emotion categories.
- Optimize prediction thresholds.
- Integrate synonym-aware text processing.
- Experiment with BERT and DistilBERT.
- Add movie prediction history.
- Add emotion visualization charts.
- Improve frontend responsiveness.
- Deploy the application online.

## 🔒 Important Notes

- The model must be trained before starting the prediction API.
- React communicates with Python through FastAPI.
- The saved Joblib model is reused for predictions.
- Prediction scores are model scores, not guaranteed emotional intensity measurements.
- Do not load model files from untrusted sources.
- Ensure dataset redistribution permissions before publishing the datasets.

## 🙌 Acknowledgements

This project uses open-source technologies including Python, Scikit-learn, Pandas, FastAPI, React, and Vite.

---

**Movie Emotion Prediction System**

*Understanding the emotions behind every movie story using Machine Learning.*