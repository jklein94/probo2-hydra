
import pandas as pd
from typing import Any
from hydra.experimental.callback import Callback
from omegaconf import DictConfig
import os
from src import utils
from functools import reduce
from hydra.core.hydra_config import HydraConfig


class All(Callback):
    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        result_file = config['combined_results_file']
        df = pd.read_csv(result_file)
        # Filter out rows where 'exit_with_error' is True
        filtered_data = df[df['exit_with_error'] == False]

        # Group by solvername and calculate statistics
        stats = filtered_data.groupby(['task','benchmark_name','solver_name']).agg(
            total_runs=('run', 'count'),
            mean_perfcounter_time=('perfcounter_time', 'mean'),
            mean_user_sys_time=('user_sys_time', 'mean'),
            num_timed_out=('timed_out', lambda x: x.sum()),
            num_not_timed_out=('timed_out', lambda x: (~x).sum()),
            proportion_timed_out=('timed_out', 'mean')
        ).reset_index()

        print(stats)

class MeanRuntime(Callback):
    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        print(utils.create_callback_chain(config.hydra.callbacks, 'mean_runtime'))
        result_file = config['combined_results_file']
        #print(config.hydra.callbacks)
        #Check if files exists before reading
        if not os.path.exists(result_file):
            print(f"File {result_file} does not exist")
            return None
        df = pd.read_csv(result_file)

        # Group by solvername and calculate statistics
        stats = df.groupby(['task','benchmark_name','solver_name']).agg(
            mean_perfcounter_time=('perfcounter_time', 'mean'),
            mean_user_sys_time=('user_sys_time', 'mean'),
        ).reset_index()

        print(stats)

        # Save the statistics to a csv file
        mean_runtime_result_file = os.path.join(config['evaluation_output_dir'], 'mean_runtime.csv')
        stats.to_csv(mean_runtime_result_file)

        # Write filepath to evaluation file index
        utils.write_evaluation_file_to_index(mean_runtime_result_file, config['evaluation_result_index_file'])


class Coverage(Callback):
    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        print(utils.create_callback_chain(config.hydra.callbacks, 'coverage'))
        result_file = config['combined_results_file']

        if not os.path.exists(result_file):
            print(f"File {result_file} does not exist")
            return None

        df = pd.read_csv(result_file)



        # Group by solvername and calculate statistics
        stats = df.groupby(['task','benchmark_name','solver_name']).agg(
            coverage=('solver_output_valid',  lambda x: (x).mean())
        ).reset_index()

        # Save the statistics to a csv file
        coverage_result_file = os.path.join(config['evaluation_output_dir'], 'coverage.csv')
        stats.to_csv(coverage_result_file)

        print(stats)

        # Write filepath to evaluation file index
        utils.write_evaluation_file_to_index(coverage_result_file, config['evaluation_result_index_file'])


class AggreateEvaluationResults(Callback):
    '''This callback is used to aggregate the evaluation results into a single file'''
    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        print(utils.create_callback_chain(config.hydra.callbacks, 'aggregate_evaluation_results'))
        index_file = config['evaluation_result_index_file']
        output_file = config['evaluation_combined_results_file']
        print(f"Aggregating evaluation results from {index_file} into {output_file}")
        try:
        # List to store individual dataframes
            dataframes = []

         # Read the index file line by line
            with open(index_file, "r") as file:
                file_paths = file.readlines()

         # Process each CSV file
            for file_path in file_paths:
                file_path = file_path.strip()  # Remove any leading/trailing whitespace
                if file_path:  # Ensure the line is not empty
                    try:
                        # Read the CSV file
                        df = pd.read_csv(file_path,index_col=0)
                        dataframes.append(df)
                        print(f"Loaded {file_path}")
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

            # Combine all dataframes
            if dataframes:
                # Assuming 'dataframes' is a list of DataFrames and 'key_column' is the column to merge on
                combined_df = reduce(lambda left, right: pd.merge(left, right, on=['task','benchmark_name','solver_name'], how='inner'), dataframes)
                # Save to the output CSV file
                combined_df.to_csv(output_file, index=False)
                print(f"Results aggregated into {output_file}")
            else:
                print("No dataframes to aggregate.")

        except Exception as e:
            print(f"An error occurred: {e}")

class PenalizedAverageRuntime2(Callback):
    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        """
        This callback calculates the Penalized Average Runtime (PAR_α) after
        all runs are complete. It assumes there's an aggregated CSV file
        containing at least:
            - 'perfcounter_time' (float): the runtime of each instance
            - 'solver_output_valid' (bool): indicates if a solver succeeded
            - 'solver_name' (str): name of the solver
            - 'task' (str): optional grouping column
            - 'benchmark_name' (str): optional grouping column

        The callback will:
            1. Read the aggregated CSV.
            2. Compute T′_i = T_i if solver_output_valid, else α * T_max.
            3. Group by (task, benchmark_name, solver_name) and calculate the
               mean of T′_i for PAR_α.
            4. Save to par_runtime.csv and register that file in an index file.
        """

        # 1. Print the callback chain with this callback highlighted
        current_callback_name = "PAR2"
        print(utils.create_callback_chain(config.hydra.callbacks, current_callback_name))

        # 2. Read the aggregated CSV file that combines all results
        result_file = config.get("combined_results_file")
        if not result_file or not os.path.exists(result_file):
            print(f"File '{result_file}' does not exist. Skipping PAR calculation.")
            return

        df = pd.read_csv(result_file)
        par_stats = calculate_par_score(df,config.get("timeout", 600),2)

        print(f"Calculated PAR2 score with timeout {config.timeout} statistics:\n")
        print(par_stats)

        # 6. Save the PAR results to a CSV file
        output_dir = config.get("evaluation_output_dir", ".")
        par_runtime_file = os.path.join(output_dir, "par_2_score.csv")
        par_stats.to_csv(par_runtime_file, index=False)

        # 7. Optionally, register this file in the evaluation file index
        index_file = config.get("evaluation_result_index_file")
        if index_file:
            utils.write_evaluation_file_to_index(par_runtime_file, index_file)

class PenalizedAverageRuntime10(Callback):
    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        """
        This callback calculates the Penalized Average Runtime (PAR_α) after
        all runs are complete. It assumes there's an aggregated CSV file
        containing at least:
            - 'perfcounter_time' (float): the runtime of each instance
            - 'solver_output_valid' (bool): indicates if a solver succeeded
            - 'solver_name' (str): name of the solver
            - 'task' (str): optional grouping column
            - 'benchmark_name' (str): optional grouping column

        The callback will:
            1. Read the aggregated CSV.
            2. Compute T′_i = T_i if solver_output_valid, else α * T_max.
            3. Group by (task, benchmark_name, solver_name) and calculate the
               mean of T′_i for PAR_α.
            4. Save to par_runtime.csv and register that file in an index file.
        """

        # 1. Print the callback chain with this callback highlighted
        current_callback_name = "PAR10"
        print(utils.create_callback_chain(config.hydra.callbacks, current_callback_name))

        # 2. Read the aggregated CSV file that combines all results
        result_file = config.get("combined_results_file")
        if not result_file or not os.path.exists(result_file):
            print(f"File '{result_file}' does not exist. Skipping PAR calculation.")
            return

        df = pd.read_csv(result_file)
        par_stats = calculate_par_score(df,config.get("timeout", 600),10)

        print(f"Calculated PAR10 score with timeout {config.timeout} statistics:\n")
        print(par_stats)

        # 6. Save the PAR results to a CSV file
        output_dir = config.get("evaluation_output_dir", ".")
        par_runtime_file = os.path.join(output_dir, "par_10_score.csv")
        par_stats.to_csv(par_runtime_file, index=False)

        # 7. Optionally, register this file in the evaluation file index
        index_file = config.get("evaluation_result_index_file")
        if index_file:
            utils.write_evaluation_file_to_index(par_runtime_file, index_file)

def calculate_par_score(df: pd.DataFrame,t_max: int,alpha: int) -> pd.DataFrame:
            # 3. Retrieve T_max (timeout) and α (penalty factor) from the config
    #    Provide defaults if not specified
    # 4. Compute penalized runtime for each instance
    #    If solver_output_valid is True -> T'_i = T_i
    #    else -> T'_i = α * T_max
    df[f"PAR_{alpha}"] = df.apply(
        lambda row: row["perfcounter_time"] if row["solver_output_valid"] else alpha * t_max,
        axis=1
    )
    # 5. Group by relevant columns and calculate the average penalized time (PAR_α)
    #    Feel free to adjust grouping columns as needed
    output_column = f"PAR_{alpha}"
    par_stats = (
    df.groupby(["task", "benchmark_name", "solver_name"], dropna=False)
      .agg(**{output_column: (f"PAR_{alpha}", "mean")})
      .reset_index()
)
    return par_stats