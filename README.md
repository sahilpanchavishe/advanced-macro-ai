# Advanced AI: Macro-Risk & Portfolio Dashboard

A 4th-year engineering capstone utilizing Probabilistic Graphical Models (PGM) to process macroeconomic data and simulate optimal portfolio allocation.

## Syllabus Implementation Overview
* **Unit 1: Probability Theory and Exact Inference.** Defines a 5-node Directed Acyclic Graph (DAG) visualizing dependencies between Global Events, Inflation, Rates, Volatility, and Strategies.
* **Unit 3: Learning Graphical Models.** Utilizes `MaximumLikelihoodEstimator` to perform Parameter Estimation, dynamically learning Conditional Probability Tables (CPTs) from a 5,000-row historical CSV dataset.
* **Unit 2: Exact Inference Techniques.** Implements `VariableElimination` to dynamically process 4 simultaneous user inputs (evidence) to calculate the precise probability distribution of portfolio strategies.
* **Unit 3: MAP Inference.** Utilizes Maximum A Posteriori optimization to decode the most probable hidden states (root cause analysis) based on terminal node observations.

## How to Run
1. Ensure your conda environment is active: `conda activate global_market`
2. Install dependencies: `pip install -r requirements.txt`
3. Generate data: `python generate_data.py`
4. Launch dashboard: `streamlit run app.py`