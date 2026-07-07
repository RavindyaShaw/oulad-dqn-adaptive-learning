import os
import pandas as pd

def load_raw_data(data_dir="data"):
    """Load all OULAD CSV files."""

    print("Loading OULAD datasets...")

    files = [
        "assessments", "courses", "vle",
        "studentInfo", "studentRegistration",
        "studentAssessment", "studentVle"
    ]

    data = {}

    for file in files:
        path = os.path.join(data_dir, f"{file}.csv")

        if not os.path.exists(path):
            raise FileNotFoundError(f"{file}.csv not found.")

        data[file] = pd.read_csv(path)

    print("All datasets loaded successfully!")
    return data


if __name__ == "__main__":
    tables = load_raw_data()
    print(tables["studentInfo"].columns.tolist())
