import os
import pandas as pd

def build_prereq_graph(data_dir="data"):
    """Create prerequisite graph from assessment dates."""

    print("Building prerequisite graph...")

    assessments = pd.read_csv(os.path.join(data_dir, "assessments.csv"))

    # Sort assessments by module, presentation, and date
    assessments = assessments.sort_values(
        by=["code_module", "code_presentation", "date"]
    )

    # Group assessments into ordered sequences
    prereq_graph = (
        assessments.groupby(
            ["code_module", "code_presentation"]
        )["id_assessment"]
        .apply(list)
        .to_dict()
    )

    print("Graph created successfully!")
    return prereq_graph


if __name__ == "__main__":
    graph = build_prereq_graph()

    if graph:
        first_key = next(iter(graph))
        print(f"{first_key}: {graph[first_key][:4]}...")
