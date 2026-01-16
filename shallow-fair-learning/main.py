import time
from sklearn.model_selection import ParameterGrid

import pandas as pd
from utils.data_preprocessing import get_Xy
from utils.tools import estimate_runtime
from runner import run_fbc_experiment
from config import get_default_config

def main():
    datasets = ['german'] # add according desired experiments
    
    # Define the experiments here
    scenarios = [
        
        ('xgb', 'xgb', 'fbc-s'),
    ]
    
    results = []

    for ds_name in datasets:
        print(f"\n>>> Processing: {ds_name.upper()}")
        X, y = get_Xy(pd.read_csv(f'data/{ds_name}.csv', sep=" "), ds_name)

        for m_old, m_new, method in scenarios:
            config = get_default_config(m_old, m_new, method)
            total_fits = estimate_runtime(config)
            
            print(f"--- Scenario: {m_old} to {m_new} ({method}) ---")
            print(f"--- Estimated total model fits: {total_fits:,} ---")
            
            start_time = time.time()
            res = run_fbc_experiment(ds_name, X, y, config)
            end_time = time.time()
            
            print(f"--- Finished in {((end_time - start_time)/60):.2f} minutes ---")
            
            results.append({**res, 'dataset': ds_name, 'scenario': f"{m_old}_{m_new}", 'method': method})

    pd.DataFrame(results).to_csv('experiment_results.csv', index=False)
    print("\nExperiments finished. Data saved to experiment_results.csv")

if __name__ == "__main__":
    main()

