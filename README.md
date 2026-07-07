# Adaptive Learning Path Generation Using Reinforcement Learning.

## A Comparative Study with Rule-Based Systems (DQN vs. Prerequisite-Graph Baseline on OULAD)

**IT41043 — Intelligent Systems**  
**Faculty of IT, Horizon Campus | Academic Year 2026**

**Authors:**  
Ravindya Hasangi Bandara Shaw (ITBIN-2313-0108)  
S. Chamudi Thamasha (ITBIN-2313-0114) 

---

# 1. Project Summary

This project investigates an adaptive learning path generation approach using a Deep Q-Network (DQN) reinforcement learning agent trained on the **Open University Learning Analytics Dataset (OULAD)**.

The study evaluates whether a reinforcement learning-based sequencing strategy can achieve a statistically significant improvement in predicted course completion rates compared with a prerequisite-graph-based rule-sequential baseline.

The DQN agent learns adaptive learning pathway decisions from historical student interaction patterns, while the baseline follows predefined module dependency relationships represented through a prerequisite graph.

The evaluation framework uses **student-level stratified 5-fold cross-validation** to ensure reliable performance comparison and prevent data leakage.

Full research design, including dataset description, problem formulation, model architecture, baseline definition, reward modelling, and evaluation methodology, is documented in:

```
docs/methodology.md
```

---

# 2. Research Objective

The objective of this research is to determine whether reinforcement learning can generate more effective personalized learning pathways compared with traditional rule-based sequencing approaches.

The study focuses on answering the following research question:

> Can a Deep Q-Network-based adaptive learning path generator improve predicted student course completion outcomes compared with a prerequisite-graph rule-based learning sequence?

The project compares:

- **DQN Adaptive Learning Agent**
  - Learns sequencing policies through reinforcement learning.
  - Uses historical student learning interactions as environmental feedback.
  - Optimizes pathway decisions based on a defined reward function.

- **Prerequisite-Graph Rule-Based Baseline**
  - Uses manually constructed module dependency relationships.
  - Generates learning sequences according to predefined prerequisite constraints.

---

# 3. Current Status (Milestone 2 Complete)

## Completed

- [x] Research gap, question, and project scope finalised (Milestone 1)
- [x] Dataset analysis and problem formulation completed
- [x] Model architecture and reinforcement learning approach designed
- [x] Rule-based baseline methodology defined
- [x] Evaluation methodology and statistical testing plan prepared
- [x] Repository structure created
- [x] Initial preprocessing pipeline structure implemented

## Pending Implementation

- [ ] OULAD data download and complete preprocessing execution
- [ ] Student weekly sequence generation
- [ ] DKT knowledge-state estimator training
- [ ] DQN agent implementation and training
- [ ] Prerequisite-graph baseline implementation
- [ ] Cross-validation experiments
- [ ] Statistical performance comparison

No trained models or experimental results are included at this stage. This milestone establishes the research methodology, software architecture, and implementation framework for subsequent experimental evaluation.

---

# 4. Repository Structure

```
.
├── data/
│   ├── raw/                         # OULAD CSV files (gitignored)
│   └── processed/                   # Engineered feature tables (gitignored)
│
├── graph/
│   └── prerequisite_graph.json      # Author-constructed module prerequisite graph
│
├── src/
│   │
│   ├── preprocessing/
│   │   ├── download_oulad.py        # Downloads and extracts OULAD dataset
│   │   ├── load_and_merge.py        # Merges OULAD tables into analysis dataset
│   │   ├── build_sequences.py       # Generates student weekly learning sequences
│   │   └── build_prereq_graph.py    # Creates and validates prerequisite graph
│   │
│   ├── models/
│   │   └── dkt.py                   # Deep Knowledge Tracing (LSTM) model
│   │
│   ├── agents/
│   │   └── dqn_agent.py             # PyTorch DQN adaptive sequencing agent
│   │
│   └── evaluation/
│       └── evaluate.py              # Cross-validation and statistical testing
│
├── notebooks/
│   └──                               # Exploratory data analysis notebooks
│
├── tests/
│   └──                               # Unit tests for preprocessing modules
│
├── docs/
│   └── methodology.md                # Full research methodology document
│
├── requirements.txt
├── main.py
├── LICENSE
└── README.md
```

---

# 5. Technical Overview

## Dataset

The project uses the **Open University Learning Analytics Dataset (OULAD)**, which contains anonymized student demographic information, assessment results, and virtual learning environment (VLE) interaction records.

The dataset provides:

- Student demographic attributes
- Course registration information
- Assessment performance
- Learning material interaction logs
- Temporal learning behaviour patterns

---

## Reinforcement Learning Model

The adaptive learning pathway generator is implemented using a Deep Q-Network (DQN).

The agent consists of:

- State representation:
  - Student engagement features
  - Assessment performance
  - Weekly learning progress
  - Historical interaction behaviour

- Action space:
  - Possible next learning module selections

- Reward function:
  - Learning progress improvement
  - Completion probability
  - Pathway efficiency

The Q-network architecture:

```
Input State
     |
Fully Connected Layer (128 neurons)
     |
ReLU Activation
     |
Fully Connected Layer (64 neurons)
     |
Output Q-values
```

---

# 6. Setup and Execution

## Clone Repository

```bash
git clone <this-repo-url>
cd <this-repo>
```

---

## Create Virtual Environment

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Download OULAD Dataset

Requires internet access.

```bash
python src/preprocessing/download_oulad.py --output data/raw
```

---

## Preprocess Dataset

```bash
python src/preprocessing/load_and_merge.py \
--input data/raw \
--output data/processed
```

---

## Run Complete Pipeline

```bash
python main.py
```

---

# 7. Evaluation Methodology

The evaluation compares the DQN agent against the prerequisite-graph baseline using:

## Cross-Validation

A student-level stratified 5-fold cross-validation strategy is applied to:

- Prevent student data leakage
- Maintain balanced outcome distributions
- Provide reliable generalization estimates

## Statistical Testing

The comparison uses:

- Normality testing for evaluation score distributions
- Wilcoxon signed-rank test for paired model comparison

The statistical analysis determines whether the DQN agent provides a significant improvement over the baseline approach.

---

# 8. Data Note

The **Open University Learning Analytics Dataset (OULAD)** is released by The Open University under the **CC-BY 4.0 licence**.

Raw and processed datasets are excluded from version control through `.gitignore` to keep the repository lightweight.

The dataset can be reproduced locally by executing:

```bash
python src/preprocessing/download_oulad.py
```

---

# 9. License

The source code in this repository is released under the **MIT License**.

The OULAD dataset is distributed separately under its own **CC-BY 4.0 licence** by The Open University.

See:

```
LICENSE
```

for repository licensing details.
