import sys
from pathlib import Path

# Ensure root directory is on sys.path so the script can be run directly
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.predict import predict_emotions

test_movies = [

    """
    A young boy discovers magical powers
    and begins an exciting adventure.
    """,

    """
    A woman loses her family and struggles
    with sadness and loneliness.
    """,

    """
    A detective investigates a mysterious
    murder while facing dangerous enemies.
    """

]

for i, plot in enumerate(test_movies, 1):

    print(f"\nMovie Test {i}:")
    print(f"Plot: {plot.strip()}")
    results = predict_emotions(plot)
    print("Predicted emotions:")
    for emotion, score in results:
        print(f"  {emotion.replace('f2_', '').title():20s} {score * 100:5.1f}%")
