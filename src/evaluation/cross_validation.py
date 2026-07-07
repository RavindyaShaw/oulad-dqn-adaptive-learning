import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from scipy.stats import wilcoxon, shapiro
from src.evaluation.metrics import calculate_evaluation_metrics


class StratifiedCrossValidator:
    def __init__(self, n_splits=5):
        self.cv = StratifiedGroupKFold(n_splits=n_splits)

    def execute_cross_validation(self, X, y, groups):
        print("Running 5-Fold Cross Validation...")

        dqn_scores, baseline_scores = [], []

        for fold, (_, val_idx) in enumerate(self.cv.split(X, y, groups), 1):
            print(f"Fold {fold}")

            y_val = y.iloc[val_idx]

            dqn_pred = np.random.randint(0, 2, len(y_val))
            dqn_prob = np.random.rand(len(y_val))
            dqn_scores.append(
                calculate_evaluation_metrics(y_val, dqn_pred, dqn_prob)["f1_macro"]
            )

            base_pred = np.random.randint(0, 2, len(y_val))
            base_prob = np.random.rand(len(y_val))
            baseline_scores.append(
                calculate_evaluation_metrics(y_val, base_pred, base_prob)["f1_macro"]
            )

        self.statistical_test(dqn_scores, baseline_scores)

    def statistical_test(self, dqn, baseline):
        _, normality = shapiro(np.array(dqn) - np.array(baseline))
        _, p = wilcoxon(dqn, baseline, alternative="greater")

        print(f"\nShapiro p-value : {normality:.4f}")
        print(f"Wilcoxon p-value: {p:.4f}")
        print("DQN performs better." if p < 0.05 else "No significant difference.")


if __name__ == "__main__":
    X = pd.DataFrame(np.random.rand(100, 4))
    y = pd.Series(np.random.randint(0, 2, 100))
    groups = pd.Series(np.repeat(range(20), 5))

    validator = StratifiedCrossValidator()
    validator.execute_cross_validation(X, y, groups)
