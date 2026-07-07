class RuleSequentialAgent:
    """Rule-based agent that recommends the next incomplete activity."""

    def __init__(self, prereq_graph):
        self.prereq_graph = prereq_graph
        print("Rule-Sequential Agent Initialized.")

    def select_action(self, course_key, completed):
        course = self.prereq_graph.get(course_key, [])

        for activity in course:
            if activity not in completed:
                return activity

        return None


if __name__ == "__main__":
    mock_graph = {
        ("AAA", "2013J"): [1752, 1753, 1754, 1755]
    }

    agent = RuleSequentialAgent(mock_graph)

    next_activity = agent.select_action(
        ("AAA", "2013J"),
        completed=[1752]
    )

    print(f"Next recommended activity: {next_activity}")
