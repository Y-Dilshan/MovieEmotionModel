
import React, { useState } from "react";

import PredictionForm from "./components/PredictionForm";

import EmotionResults from "./components/EmotionResults";

import "./App.css";


function App() {

    const [predictions, setPredictions] = useState([]);

    const [loading, setLoading] = useState(false);

    const [error, setError] = useState("");

    const handlePrediction = async (plot) => {

        setLoading(true);

        setError("");

        setPredictions([]);

        try {

            const response = await fetch(
                "http://127.0.0.1:8000/predict",
                {
                    method: "POST",

                    headers: {
                        "Content-Type": "application/json"
                    },

                    body: JSON.stringify({

                        plot: plot,

                        top_k: 5,

                        threshold: 0.35

                    })

                }
            );

            const data = await response.json();

            if (!response.ok) {

                throw new Error(
                    data.detail || "Prediction failed."
                );

            }

            setPredictions(data.predictions);

        } catch (err) {

            setError(err.message);

        } finally {

            setLoading(false);

        }

    };


    return (

        <div className="app">

            <header className="header">

                <div className="logo">
                    <span>ME</span>
                    Movie Emotion AI
                </div>

                <span className="status">
                    ML Powered
                </span>

            </header>


            <main className="main-container">

                <section className="hero">

                    <span className="hero-tag">
                        ARTIFICIAL INTELLIGENCE
                    </span>

                    <h1>
                        Discover the Emotions
                        <br />
                        Behind Every Movie
                    </h1>

                    <p>
                        Enter a movie plot and let our machine
                        learning model identify its emotions.
                    </p>

                </section>


                <section className="prediction-container">

                    <PredictionForm

                        onPredict={handlePrediction}

                        loading={loading}

                    />

                    {loading && (

                        <p className="loading">
                            Analyzing movie emotions...
                        </p>

                    )}

                    {error && (

                        <p className="error">
                            {error}
                        </p>

                    )}

                    <EmotionResults

                        predictions={predictions}

                    />

                </section>

            </main>


            <footer>

                Movie Emotion Prediction System

                <span>Powered by Machine Learning</span>

            </footer>

        </div>

    );

}

export default App;
