import os
import pandas as pd

def build_student_weekly_features(data_dir="data"):
    print("⚙️ Processing...")

    info = pd.read_csv(os.path.join(data_dir, "studentInfo.csv"))
    vle = pd.read_csv(os.path.join(data_dir, "studentVle.csv"))

    # target
    info["completed"] = info["final_result"].isin(["Pass", "Distinction"]).astype(int)

    # weekly feature
    vle = vle.dropna(subset=["date", "sum_click"])
    vle["week"] = (vle["date"] // 7).astype(int)

    vle = vle[vle["week"] >= 0]

    weekly = vle.groupby(
        ["code_module", "code_presentation", "id_student", "week"],
        as_index=False
    )["sum_click"].sum()

    weekly.rename(columns={"sum_click": "weekly_clicks"}, inplace=True)

    print("✅ Done!")

    return weekly, info


if __name__ == "__main__":
    try:
        weekly_df, outcomes = build_student_weekly_features()

        print("Shape:", weekly_df.shape)
        print("Completion Rate:", outcomes["completed"].mean())

    except Exception as e:
        print("Error:", e)
