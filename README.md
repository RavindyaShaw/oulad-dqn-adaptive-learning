Adaptive Learning Path Generation Using Reinforcement Learning

*A Comparative Study with Rule-Based Systems (DQN vs. Prerequisite-Graph Baseline on OULAD)*

IT41043 — Intelligent Systems | Faculty of IT, Horizon Campus | Academic Year 2026

Authors: Ravindya Hasangi Bandara Shaw (ITBIN-2313-0108), S. Chamudi Thamasha

## 1. Project Summary

This project trains a Deep Q-Network (DQN) adaptive learning path agent on the
[Open University Learning Analytics Dataset (OULAD)](https://analyse.kmi.open.ac.uk/open_dataset)
and evaluates whether it achieves statistically significantly higher predicted course
completion rates than a prerequisite-graph-based rule-sequential baseline, using
stratified 5-fold cross-validation.

Full research design (dataset description, model architecture, baseline definition,
and evaluation plan) is documented in docs/methodology.md.

## 2. Current Status (Milestone 2)

- [x] Research gap, question, and scope finalised (Milestone 1)
- [x] Dataset description, model architecture, baseline, and evaluation plan finalised (Milestone 2)
- [x] Repository structure and preprocessing scripts scaffolded
- [ ] OULAD data download and full preprocessing run
- [ ] DKT knowledge-state estimator training
- [ ] DQN agent training (offline RL)
- [ ] Rule-based baseline implementation
- [ ] Cross-validated evaluation and statistical testing (Milestone 4)

No trained models or final results exist yet — this submission establishes the
project's technical foundation ahead of full implementation.

## 3. Repository Structure


.
├── data/
│   ├── raw/                  # OULAD CSVs (gitignored — see Section 4 to download)
│   └── processed/            # Cleaned / merged feature tables (gitignored)
├── graph/
│   └── prerequisite_graph.json   # Author-constructed module prerequisite graph
├── src/
│   ├── preprocessing/
│   │   ├── download_oulad.py       # Fetches and unpacks the OULAD CSVs
│   │   ├── load_and_merge.py       # Merges the 7 OULAD tables into one analysis frame
│   │   ├── build_sequences.py      # Builds per-student weekly interaction sequences
│   │   └── build_prereq_graph.py   # Constructs / validates the module prerequisite graph
│   ├── models/
│   │   └── dkt.py                  # DKT (LSTM) knowledge-state estimator
│   ├── agents/
│   │   └── dqn_agent.py            # DQN sequencing agent (stable-baselines3 wrapper)
│   └── evaluation/
│       └── evaluate.py             # Stratified 5-fold CV + Wilcoxon signed-rank test
├── notebooks/                # Exploratory analysis (not yet populated)
├── tests/                    # Unit tests for preprocessing functions
├── docs/
│   └── methodology.md        # Full Milestone 2 methodology writeup
├── requirements.txt
└── README.md


## 4. Setup

bash
git clone <this-repo-url>
cd <this-repo>
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Download OULAD (requires internet access; ~700MB uncompressed)
python src/preprocessing/download_oulad.py --output data/raw

# Merge tables and engineer features
python src/preprocessing/load_and_merge.py --input data/raw --output data/processed


## 5. Data Note

OULAD is released by The Open University under a CC-BY 4.0 licence. Raw and
processed data files are excluded from version control via .gitignore to keep
the repository lightweight; running download_oulad.py reproduces them locally.

## 6. License

Code in this repository is released under the MIT License (see LICENSE).
OULAD itself is licensed separately by The Open University under CC-BY 4.0.
