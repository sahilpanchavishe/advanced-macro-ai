# Institutional Macro-Risk & Portfolio AI Dashboard

A 4th-year engineering capstone utilizing Probabilistic Graphical Models (PGM) to process macroeconomic data and simulate optimal portfolio allocation.

## Syllabus Implementation Overview

* **Unit 1: Probability Theory and Exact Inference.** Defines a 5-node Directed Acyclic Graph (DAG) visualizing dependencies between Global Events, Inflation, Rates, Volatility, and Strategies.
* **Unit 3: Learning Graphical Models.** Utilizes `MaximumLikelihoodEstimator` to perform Parameter Estimation, dynamically learning Conditional Probability Tables (CPTs) from a 5,000-row historical CSV dataset.
* **Unit 2: Exact Inference Techniques.** Implements `VariableElimination` to dynamically process 4 simultaneous user inputs (evidence) to calculate the precise probability distribution of portfolio strategies.
* **Unit 3: MAP Inference.** Utilizes Maximum A Posteriori optimization to decode the most probable hidden states (root cause analysis) based on terminal node observations.

---

## How to Run This Project on Any Laptop

Follow these exact steps in your terminal or command prompt to download and run the AI dashboard locally.

### 1. Clone the Repository

Avoid downloading as a ZIP file to prevent hidden system file errors. Use Git:

```bash
git clone https://github.com/sahilpanchavishe/advanced-macro-ai.git
cd advanced-macro-ai
```

### 2. Install All Required Dependencies

It is recommended to use a virtual environment such as Conda or `venv`.

Install the required libraries:

```bash
pip install -r requirements.txt
```

### 3. Generate the AI Training Data

Run the data generator to build the 5,000-row macroeconomic CSV dataset. This is required for the AI to perform parameter estimation:

```bash
python generate_data.py
```

### 4. Launch the AI Dashboard

Start the interactive Streamlit web application:

```bash
streamlit run app.py
```

The application will open automatically in your browser. If it does not, open the URL shown in the terminal, usually:

```text
http://localhost:8501
```

---

## Project Architecture

The system models the relationship between macroeconomic conditions and investment strategies using a Bayesian Network.

```text
                 Global Events
                       |
                       v
                   Inflation
                       |
                       v
                    Rates
                       |
                       v
                  Volatility
                       |
                       v
                  Strategy
```

The model allows macroeconomic conditions to propagate through the network and influence the probability of different portfolio strategies.

---

## Core AI Components

### 1. Probabilistic Graphical Model

A Directed Acyclic Graph (DAG) represents the causal dependencies between the five variables:

- Global Events
- Inflation
- Rates
- Volatility
- Strategy

This provides a structured representation of how macroeconomic factors influence financial-market conditions and portfolio decisions.

### 2. Parameter Estimation

The system generates a dataset containing 5,000 simulated macroeconomic observations.

`MaximumLikelihoodEstimator` learns the Conditional Probability Tables (CPTs) from this dataset.

This allows the Bayesian Network to learn probability relationships directly from the generated data.

### 3. Variable Elimination

The dashboard allows the user to provide evidence for multiple macroeconomic variables.

`VariableElimination` performs exact probabilistic inference using the provided evidence.

The model calculates the resulting probability distribution of the possible portfolio strategies.

### 4. MAP Inference

Maximum A Posteriori (MAP) inference is used to determine the most probable hidden states based on observed terminal conditions.

This provides a form of root-cause analysis by working backward from observed market conditions.

---

## Example Workflow

The general workflow of the application is:

```text
Macroeconomic Data
        |
        v
Data Generation
        |
        v
5,000 Historical Observations
        |
        v
Parameter Estimation
        |
        v
Bayesian Network
        |
        +----------------------+
        |                      |
        v                      v
Variable Elimination       MAP Inference
        |                      |
        v                      v
Strategy Probabilities    Root Cause Analysis
```

---

## Technologies Used

- **Python**
- **Streamlit**
- **pgmpy**
- **Pandas**
- **NumPy**
- **Probabilistic Graphical Models**
- **Bayesian Networks**
- **Maximum Likelihood Estimation**
- **Variable Elimination**
- **MAP Inference**

---

## Project Structure

```text
advanced-macro-ai/
│
├── app.py
├── generate_data.py
├── requirements.txt
├── README.md
│
└── data/
    └── macro_data.csv
```

> Note: The `macro_data.csv` file is generated automatically by running `generate_data.py`.

---

## Requirements

The project requires Python 3.10 or later.

All Python dependencies are listed in:

```text
requirements.txt
```

Install them using:

```bash
pip install -r requirements.txt
```

---

## Running the Project

After completing the installation steps:

```bash
python generate_data.py
```

Then:

```bash
streamlit run app.py
```

The dashboard will be available at:

```text
http://localhost:8501
```

---

## Purpose of the Project

The objective of this project is to demonstrate how Probabilistic Graphical Models can be applied to a real-world financial decision-support problem.

Instead of relying on a conventional black-box machine learning model, the system explicitly represents probabilistic relationships between macroeconomic variables.

This makes it possible to:

- Model dependencies between economic variables.
- Learn probability distributions from data.
- Perform exact probabilistic inference.
- Estimate portfolio strategy probabilities.
- Perform MAP-based root-cause analysis.
- Provide an interactive decision-support dashboard.

---

## Disclaimer

This project is an academic simulation developed for educational and demonstration purposes.

The portfolio strategies and probabilities generated by the system should not be considered financial advice or real-world investment recommendations.
