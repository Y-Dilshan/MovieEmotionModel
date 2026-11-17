
import React, { useState } from "react";

function PredictionForm({ onPredict, loading }) {

    const [plot, setPlot] = useState("");

    const handleSubmit = (e) => {

        e.preventDefault();

        if (!plot.trim()) {
            alert("Please enter a movie plot.");
            return;
        }

        onPredict(plot);

    };

    return (

        <form onSubmit={handleSubmit} className="prediction-form">

            <label>Enter Movie Plot</label>

            <textarea

                value={plot}

                onChange={(e) => setPlot(e.target.value)}

                placeholder="Enter your movie story or plot here..."

                rows={7}

                required

            />

            <div className="form-footer">

                <span>{plot.length} characters</span>

                <button type="submit" disabled={loading}>

                    {loading ? "Analyzing..." : "Predict Emotions"}

                </button>

            </div>

        </form>

    );

}

export default PredictionForm;
