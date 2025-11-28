import re
from pathlib import Path
import numpy as np
import sys
import pandas as pd

'''
This script processes the results from multiple runs of PPO with different GAE lambda values and seeds, computes the average return per timestep for each lambda,
and saves the results to a CSV file.
Input:
    - Command line argument 1: environment name (str)
'''

# Recover env name from command line argument
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
average_return_per_timestep_for_each_lambda = pd.DataFrame(columns=lambdas)
convergence_info = pd.DataFrame(columns=['lambda', 'convergence_value_mean', 'convergence_value_ci95'])
early_learning_info = pd.DataFrame(columns=['lambda', 'early_learning_value_mean', 'early_learning_value_ci95'])
for lambda_folder in lambda_folders:
    print(f'Reading folder: {runs_folder/lambda_folder}')
    
    # Get lambda value
    labas = re.findall(r'\d*\.?\d+', lambda_folder)[-1]

    # Getting all the csv files in folder
    same_lambda_different_seeds_csvs = [e for e in (runs_folder/lambda_folder).iterdir() if str(e).endswith('.csv')]
    number_of_seeds = len(same_lambda_different_seeds_csvs)

    # Interpolate and average different seeds
    averaged_interpolated_returns = np.zeros(500_000+1)
    averaged_seeds = []
    convergence_values = []
    early_learning_values = []
    for csv_path in same_lambda_different_seeds_csvs:
        # Reading CSV
        dataframe = pd.read_csv(csv_path)
        
        # Interpolation
        x_col = 'global_step'    
        y_col = 'episodic_return'

        interpolated_values = np.interp(np.linspace(0, 500_000-1, 500_000+1), dataframe[x_col], dataframe[y_col])

        # Averaging timestep returns over seeds
        averaged_interpolated_returns += interpolated_values/number_of_seeds
        if np.isnan(interpolated_values.mean()):
            print(csv_path)

        # Averaging average returns of seeds
        averaged_seeds.append(interpolated_values.mean())

        # Average of last 10_000 steps
        if len(sys.argv) > 2:
            percentage_to_average = sys.argv[2]
            last_n_steps = int(500_000 * float(percentage_to_average))
            convergence_value = interpolated_values[-last_n_steps:].mean()
        else:
            convergence_value = interpolated_values[-10_000:].mean()
        convergence_values.append(convergence_value)

        if len(sys.argv) > 3:
            percentage_to_average = sys.argv[3]
            first_n_steps = int(500_000 * float(percentage_to_average))
            early_learning_value = interpolated_values[:first_n_steps].mean()
        else:
            early_learning_value = interpolated_values[:125_000].mean()
        early_learning_values.append(early_learning_value)
    
    early_percentile_5,early_percentile_95 = np.percentile(early_learning_values,[5,95])
    conv_percentile_5,conv_percentile_95 = np.percentile(convergence_values,[5,95])

    # Calculating 95% confidence interval over seeds for the average return per timestep
    confidence_interval_95_percent = 1.96 * (np.array(averaged_seeds).std() / np.sqrt(number_of_seeds))
    print(f'Average over seeds for lambda={labas}: {np.array(averaged_seeds).mean():.2f} +- {confidence_interval_95_percent:.2f}')
    
    # Storing the averaged returns for this lambda 
    average_return_per_timestep_for_each_lambda[labas] = [*averaged_interpolated_returns, confidence_interval_95_percent]
    print(f'Average for lambda={labas}: {averaged_interpolated_returns.mean()}')

    # Calculating convergence value over seeds and confidence interval
    convergence_value_mean = np.array(convergence_values).mean()
    convergence_value_ci95 = 1.96 * (np.array(convergence_values).std() / np.sqrt(number_of_seeds))
    convergence_info = pd.concat([convergence_info, pd.DataFrame([[float(labas), convergence_value_mean, convergence_value_ci95,conv_percentile_5,conv_percentile_95]], columns=['lambda', 'convergence_value_mean', 'convergence_value_ci95','convergence_p5','convergence_p95'])], ignore_index=True)

    early_learning_value_mean = np.array(early_learning_values).mean()
    early_learning_value_ci95 =  1.96 * (np.array(early_learning_values).std() / np.sqrt(number_of_seeds))  
    early_learning_info = pd.concat([early_learning_info, pd.DataFrame([[float(labas), early_learning_value_mean, early_learning_value_ci95,early_percentile_5,early_percentile_95]], columns=['lambda', 'early_learning_value_mean', 'early_learning_value_ci95','early_learning_p5','early_learning_p95'])], ignore_index=True)

# Saving the final dataframe of average return per timestep for each lambda
analyzed_data_folder = 'runs/' + 'analyzed_data'
analyzed_data_folder = Path(analyzed_data_folder)
analyzed_data_folder.mkdir(exist_ok=True)
average_return_per_timestep_for_each_lambda.to_csv(analyzed_data_folder / f'average_return_per_timestep_for_each_lambda_{env_name}.csv', index=False)

# Saving convergence info
convergence_info.to_csv(analyzed_data_folder / f'convergence_info_{env_name}.csv', index=False)
early_learning_info.to_csv(analyzed_data_folder / f'early_learning_info_{env_name}.csv', index = False)
