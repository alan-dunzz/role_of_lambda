import re
from pathlib import Path
import numpy as np
import sys
import pandas as pd

# Env name based on command line argument
env_name = sys.argv[1] 

# Folder with the runs
runs_folder = 'runs/' + env_name + '/'
runs_folder = Path(runs_folder)
if not runs_folder.exists():
    raise Exception(f'Folder {runs_folder} does not exist!')

# Get folder for each lambda
subfolders = [item for item in runs_folder.iterdir() if item.is_dir()]
lambda_folders = sorted([item.name for item in runs_folder.iterdir() if item.is_dir()])
lambdas = [re.findall(r'\d*\.?\d+', lambda_folder)[-1] for lambda_folder in lambda_folders]

# Load the csv for each lambda and each seed as a Pandas array
return_per_timestep_for_each_lambda = pd.DataFrame(columns=lambdas)
for lambda_folder in lambda_folders:
    print(f'Reading folder: {runs_folder/lambda_folder}')
    
    # Get lambda value
    labas = re.findall(r'\d*\.?\d+', lambda_folder)[-1]

    # Getting all the csv files in folder
    same_lambda_different_seeds_csvs = [e for e in (runs_folder/lambda_folder).iterdir() if str(e).endswith('.csv')]
    number_of_seeds = len(same_lambda_different_seeds_csvs)

    # Interpolate and average different seeds
    averaged_interpolated_returns = np.zeros(500_000+1)
    for csv_path in same_lambda_different_seeds_csvs:
        # Reading CSV
        dataframe = pd.read_csv(csv_path)
        
        # Interpolation
        x_col = 'global_step'    
        y_col = 'episodic_return'

        interpolated_values = np.interp(np.linspace(0, 500_000-1, 500_000+1), dataframe[x_col], dataframe[y_col])

        # Averaging
        averaged_interpolated_returns += interpolated_values/number_of_seeds
        if np.isnan(interpolated_values.mean()):
            print(csv_path)
    # Storing the averaged returns for this lambda 
    return_per_timestep_for_each_lambda[labas] = averaged_interpolated_returns

# Saving the final dataframe
analysed_data_folder = 'runs/' + 'analysed_data'
analysed_data_folder.mkdir(exist_ok=True)
return_per_timestep_for_each_lambda.to_csv(analysed_data_folder / f'average_return_per_timestep_for_each_lambda_{env_name}.csv', index=False)