import json
from pathlib import Path

def main():
    path = Path(__file__).resolve().parents[1] / "reports" / "evaluation.json"
    if not path.exists():
        print("Evaluation report not found. Run: python -m src.train")
        return

    with open(path, encoding="utf-8") as file:
        metrics = json.load(file)

    print("MODEL EVALUATION")
    print("----------------")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:20s}: {value:.4f}")
        elif key != "labels":
            print(f"{key:20s}: {value}")

    print("\nPer-emotion details are in reports/classification_report.csv")


if __name__ == "__main__":
    main()
