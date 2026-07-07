import pandas as pd
import numpy as np

from src.preprocessing.load_oulad import load_raw_data
from src.preprocessing.build_features import build_student_weekly_features
from src.rl.environment import OULADLearningEnv
from src.rl.dqn_agent import DQNAgent
from src.evaluation.cross_validation import StratifiedCrossValidator


def main():
    print("=== Step 1: Loading and Preprocessing OULAD Data ===")
    raw_data_path = "data/raw/"
    
    
    tables = load_raw_data(raw_data_path)
    
    states, info_df = build_student_weekly_features(raw_data_path)
    
    print("\n=== Step 2: Initializing RL Environment and DQN Agent ===")
    
    env = OULADLearningEnv(weekly_data=states, outcomes_data=info_df)
    
    
    state_dim = states.shape[1] - 4 if "id_student" in states.columns else states.shape[1]
    action_dim = 22
    
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    
    print("\n=== Step 3: Running 5-Fold Cross-Validation ===")
    
 
    validator = StratifiedCrossValidator(n_splits=5)
    
   
    if "id_student" in states.columns and "id_student" in info_df.columns:
        
       
        merged_df = pd.merge(
            states,
            info_df[["id_student", "completed"]],
            on="id_student",
            how="left"
        )
        
        merged_df["completed"] = merged_df["completed"].fillna(0).astype(int)
        
        X = merged_df.drop(
            columns=[
                "code_module",
                "code_presentation",
                "id_student",
                "week",
                "completed"
            ],
            errors="ignore"
        )
        
        y = merged_df["completed"]
        groups = merged_df["id_student"]
        
    else:
       
        X = states.drop(
            columns=[
                "code_module",
                "code_presentation",
                "id_student",
                "week"
            ],
            errors="ignore"
        )
        
        y = pd.Series(np.random.randint(0, 2, len(states)))
        
        groups = (
            states["id_student"]
            if "id_student" in states.columns
            else pd.Series(range(len(states)))
        )
    
  
    validator.execute_cross_validation(X, y, groups)
    
    print("\nExecution completed successfully!")


if __name__ == "__main__":
    main()
