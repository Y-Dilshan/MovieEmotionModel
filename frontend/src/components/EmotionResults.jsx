
import React from "react";

function EmotionResults({ predictions }) {

    if (!predictions || predictions.length === 0) {
        return null;
    }

    return (

        <div className="results-section">

            <h2>Predicted Emotions</h2>

            <p className="results-description">
                Emotions identified from your movie plot.
            </p>

            <div className="emotion-list">

                {predictions.map((item, index) => (

                    <div className="emotion-card" key={item.emotion}>

                        <div className="emotion-header">

                            <h3>{item.emotion}</h3>

                            <span>
                                {item.percentage.toFixed(2)}%
                            </span>

                        </div>

                        <div className="progress-background">

                            <div

                                className="progress-fill"

                                style={{
                                    width: `${item.percentage}%`
                                }}

                            />

                        </div>

                    </div>

                ))}

            </div>

        </div>

    );

}

export default EmotionResults;
