import pandas as pd
import numpy as np

def create_advanced_dataset():
    print("Generating 5,000 records for the Macro-Risk Dataset...")
    np.random.seed(42)
    n = 5000

    events = np.random.choice(['Crisis', 'Normal', 'Economic Boom'], size=n, p=[0.15, 0.65, 0.20])
    inflation = np.random.choice(['Low', 'Target', 'High'], size=n, p=[0.25, 0.50, 0.25])

    rates = []
    for e, i in zip(events, inflation):
        if e == 'Crisis' or i == 'Low':
            rates.append(np.random.choice(['Cut (Lowered)', 'Hold (Unchanged)', 'Hike (Raised)'], p=[0.7, 0.2, 0.1]))
        elif e == 'Economic Boom' and i == 'High':
            rates.append(np.random.choice(['Cut (Lowered)', 'Hold (Unchanged)', 'Hike (Raised)'], p=[0.05, 0.15, 0.80]))
        else:
            rates.append(np.random.choice(['Cut (Lowered)', 'Hold (Unchanged)', 'Hike (Raised)'], p=[0.2, 0.6, 0.2]))

    volatility = []
    for e, r in zip(events, rates):
        if e == 'Crisis' or r == 'Hike (Raised)':
            volatility.append(np.random.choice(['Low Volatility', 'High Volatility'], p=[0.2, 0.8]))
        else:
            volatility.append(np.random.choice(['Low Volatility', 'High Volatility'], p=[0.75, 0.25]))

    strategy = []
    for v, i in zip(volatility, inflation):
        if v == 'High Volatility' and i == 'High':
            strategy.append(np.random.choice(['Defensive', 'Balanced', 'Aggressive'], p=[0.8, 0.15, 0.05]))
        elif v == 'Low Volatility' and i == 'Target':
            strategy.append(np.random.choice(['Defensive', 'Balanced', 'Aggressive'], p=[0.1, 0.3, 0.6]))
        else:
            strategy.append(np.random.choice(['Defensive', 'Balanced', 'Aggressive'], p=[0.3, 0.5, 0.2]))

    df = pd.DataFrame({
        'Global_Event': events,
        'Inflation_Level': inflation,
        'Interest_Rates': rates,
        'Market_Volatility': volatility,
        'Portfolio_Strategy': strategy
    })
    df.to_csv('macro_portfolio_data.csv', index=False)
    print("Success! Dataset 'macro_portfolio_data.csv' created.")

if __name__ == "__main__":
    create_advanced_dataset()