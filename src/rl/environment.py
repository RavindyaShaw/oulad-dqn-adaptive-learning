import numpy as np
import pandas as pd

class OULADLearningEnv:
    """Simple RL environment for OULAD."""

    def __init__(self, weekly_data, outcomes_data):
        self.weekly_data = weekly_data
        self.outcomes_data = outcomes_data
        print("OULAD Environment Initialized.")

    def get_student_state(self, student_id, week):
        record = self.weekly_data[
            (self.weekly_data["id_student"] == student_id) &
            (self.weekly_data["week"] == week)
        ]

        if record.empty:
            return np.zeros(5)

        return np.array([record["sum_click"].iloc[0], week, 0, 0, 0])

    def compute_reward(self, student_id):
        reward = -0.1
        outcome = self.outcomes_data[
            self.outcomes_data["id_student"] == student_id
        ]

        if not outcome.empty and outcome["completed"].iloc[0] == 1:
            reward += 5.0

        return reward


if __name__ == "__main__":
    mock_weekly = pd.DataFrame({
        "id_student": [1001],
        "week": [1],
        "sum_click": [45]
    })

    mock_outcomes = pd.DataFrame({
        "id_student": [1001],
        "completed": [1]
    })

    env = OULADLearningEnv(mock_weekly, mock_outcomes)

    state = env.get_student_state(1001, 1)
    reward = env.compute_reward(1001)

    print("State:", state)
    print("Reward:", reward)
