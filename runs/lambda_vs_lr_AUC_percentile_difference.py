import re
from pathlib import Path
import numpy as np
import sys
import pandas as pd
from itertools import product

'''
This script processes the results from multiple runs of PPO with different GAE lambda values and seeds, computes the average return per timestep for each lambda,
and saves the results to a CSV file.
Input:
    - Command line argument 1: environment name (str)
'''

# Recover env name from command line argument
env_name = sys.argv[1]
n_timesteps = 500_000

# Folder with the runs
#script_dir = str(Path(__file__).parent)
#runs_folder = script_dir +'/runs/' + env_name + '/lr_sweep/'
runs_folder = 'runs/' + env_name + '/lr_sweep/'
runs_folder = Path(runs_folder)
if not runs_folder.exists():
    raise Exception(f'Folder {runs_folder} does not exist!')

# Get folder for each lambda
subfolders = [item for item in runs_folder.iterdir() if item.is_dir()]
lambda_folders = sorted([item.name for item in runs_folder.iterdir() if item.is_dir()])
lr_folders = sorted([item.name for item in Path(f'{runs_folder}/'+lambda_folders[0]).iterdir() if item.is_dir()])
lr_scheme_folders = sorted([item.name for item in Path(f'{runs_folder}/'+f'{lambda_folders[0]}/'+lr_folders[0]).iterdir() if item.is_dir()])
lr_folders_shortned = [re.sub('learning_rate', 'lr', s) for s in lr_folders]

# Get each possible combination of alphas(lr) and lambdas (81 combinations)
combinations = [a + b for a, b in product(lambda_folders, lr_folders_shortned)]


# avg_ret_per_t_for_each_lambda_lr_anneal_lr_False = pd.DataFrame(columns=combinations)
# avg_ret_per_t_for_each_lambda_lr_anneal_lr_True = pd.DataFrame(columns=combinations)

# AUC_anneal_lr_False = pd.DataFrame(columns=lambda_folders,index=lr_folders_shortned)
# AUC_anneal_lr_True = pd.DataFrame(columns=lambda_folders,index=lr_folders_shortned)

# Declare dataframes for percentiles difference heatmap
AUC_percentiles_difference_lr_False = pd.DataFrame(columns=lambda_folders,index=lr_folders_shortned)
AUC_percentiles_difference_lr_True = pd.DataFrame(columns=lambda_folders,index=lr_folders_shortned)

for lambda_folder in lambda_folders:
    for lr_folder in lr_folders:
        for lr_scheme in lr_scheme_folders:
            # Getting all the csv files in folder
            same_lambda_lr_scheme_different_seeds_csvs = [e for e in (runs_folder/lambda_folder/lr_folder/lr_scheme).iterdir() if str(e).endswith('.csv')]
            number_of_seeds = len(same_lambda_lr_scheme_different_seeds_csvs)

            # Interpolate and average different seeds
            averaged_interpolated_returns = np.zeros(n_timesteps+1)
            averaged_seeds = []
            #convergence_values = []
            #early_learning_values = []
            list_AUC_all_seeds= []

            for csv_path in same_lambda_lr_scheme_different_seeds_csvs:
                # Reading CSV
                dataframe = pd.read_csv(csv_path)
                
                # Interpolation
                x_col = 'global_step'    
                y_col = 'episodic_return'

                # Curve per seed
                interpolated_values = np.interp(np.linspace(0, n_timesteps-1, n_timesteps+1), dataframe[x_col], dataframe[y_col])

                # AUC per seed
                AUC_interpolated_per_seed=interpolated_values.mean()
                # Save the AUC in list
                list_AUC_all_seeds.append(AUC_interpolated_per_seed)


                # Averaging timestep returns over seeds
                averaged_interpolated_returns += interpolated_values/number_of_seeds
                if np.isnan(interpolated_values.mean()):
                    print(csv_path)

            # Extract 95 and 5 percentiles of a single alpha-lambda-scheme combination

            percentile_95, percentile_5 = np.percentile(list_AUC_all_seeds,[95,5])


            # Storing the averaged returns for this lambda and lr
            if "True" in lr_scheme:
                #avg_ret_per_t_for_each_lambda_lr_anneal_lr_True[f'{lambda_folder}{re.sub('learning_rate', 'lr', lr_folder)}'] = [*averaged_interpolated_returns]
                # To obtain data for Averaged AUC 
                #AUC_anneal_lr_True.loc[f'{re.sub('learning_rate', 'lr', lr_folder)}',f'{lambda_folder}'] = averaged_interpolated_returns.mean()
                #print(f"AUC {lambda_folder}, {lr_folder}, {lr_scheme} = {averaged_interpolated_returns.mean()}")

                # To obtain data for percentiles difference
                AUC_percentiles_difference_lr_True.loc[f'{re.sub('learning_rate', 'lr', lr_folder)}',f'{lambda_folder}'] = np.abs(percentile_95-percentile_5)
                    
            else:
                #avg_ret_per_t_for_each_lambda_lr_anneal_lr_False[f'{lambda_folder}{re.sub('learning_rate', 'lr', lr_folder)}'] = [*averaged_interpolated_returns]
                #AUC_anneal_lr_False.loc[f'{re.sub('learning_rate', 'lr', lr_folder)}',f'{lambda_folder}'] = averaged_interpolated_returns.mean()
                #print(f"AUC {lambda_folder}, {lr_folder}, {lr_scheme} = {averaged_interpolated_returns.mean()}")

                # To obtain data for percentiles difference
                AUC_percentiles_difference_lr_False.loc[f'{re.sub('learning_rate', 'lr', lr_folder)}',f'{lambda_folder}'] = np.abs(percentile_95-percentile_5)
            
            print(f"95-5 percentiles in {lambda_folder}, {lr_folder}, {lr_scheme} = {np.abs(percentile_95-percentile_5)}")




# Saving the final dataframe of average return per timestep for each lambda and lr
analyzed_data_folder = 'runs/' + 'analyzed_data'
analyzed_data_folder = Path(analyzed_data_folder)
analyzed_data_folder.mkdir(exist_ok=True)


#avg_ret_per_t_for_each_lambda_lr_anneal_lr_True.to_csv(analyzed_data_folder / f'avg_ret_per_t_for_each_lambda_lr_anneal_True_{env_name}.csv', index=False)
#avg_ret_per_t_for_each_lambda_lr_anneal_lr_False.to_csv(analyzed_data_folder / f'avg_ret_per_t_for_each_lambda_lr_anneal_False_{env_name}.csv', index=False)
# Dataframe files=lr, columns=lambdas, intersection=Averaged AUC
# AUC_anneal_lr_False.to_csv(analyzed_data_folder / f'AUC_anneal_lr_False_{env_name}.csv',index = True)
# AUC_anneal_lr_True.to_csv(analyzed_data_folder / f'AUC_anneal_lr_True_{env_name}.csv',index = True)
# Dataframe files=lr, columns=lambdas, intersection= AUC95percentile-AUC5percentile
AUC_percentiles_difference_lr_True.to_csv(analyzed_data_folder / f'AUC_percentiles_difference_lr_True_{env_name}.csv',index = True)
AUC_percentiles_difference_lr_False.to_csv(analyzed_data_folder / f'AUC_percentiles_difference_lr_False_{env_name}.csv',index = True)
