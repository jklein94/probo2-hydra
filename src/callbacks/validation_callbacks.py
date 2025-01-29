import pandas as pd
from typing import Any
from hydra.experimental.callback import Callback
from omegaconf import DictConfig, OmegaConf
import os
from src import utils
import re

from hydra.core.hydra_config import HydraConfig


def validate_solver_results(df: pd.DataFrame, config: DictConfig):

    benchmark_name = df["benchmark_name"].iloc[0]
    task = df["task"].iloc[0]
    solver_name = df["solver_name"].iloc[0]
    benchmark_config = OmegaConf.load(os.path.join(config.root_dir, "benchmark_config", f"{benchmark_name}_config.yaml"))

    # Validate the solver results against given solutions
    # Check if in the benchmarks config a solutions_path is given
    if "solution_path" in benchmark_config.keys():
        # Check if solution_path is not empty and the path exists
        if benchmark_config.solution_path == "" or not os.path.exists(benchmark_config.solution_path):
                    raise FileNotFoundError("The solution_path is not valid")
        else:
            print(f"Validating solver results for benchmark: {benchmark_name}, task: {task}, solver: {solver_name}")
            instance_names_in_df = df["instance"].unique()
            missing_file_path = os.path.join(config.result_validation_output_dir, f"{solver_name}_{benchmark_name}_{task}_missing_instances.txt")
            reference_solutions_df = load_reference_solutions(benchmark_config.solution_path, task,expected_instances=instance_names_in_df, missing_file=missing_file_path)
            # compare the reference solution with the solver solution
            solver_solutions_df = load_solver_solutions(df)




            # Add a new column called raw_instance_name to the dataframe by striping the path and extension from the instance name
            solver_solutions_df["raw_instance_name"] = solver_solutions_df["instance"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])

            #iterate over the rows of the dataframe and compare the solver solution with the reference solution
            for index, row in solver_solutions_df.iterrows():
                if row["solver_solution"] == "INVALID":
                    df.loc[index, "solver_output_valid"] = False
                    # write the path of the instance with the invalid solution to a file
                    with open(os.path.join(config.result_validation_output_dir, f"{solver_name}_{benchmark_name}_{task}_invalid_solution.txt"), "a") as file:
                        file.write(row["instance"] + "\n")
                    print(f"Invalid solution for instance: {row['instance']}")
                else:
                    # get the reference solution for the instance
                    reference_solution = reference_solutions_df[reference_solutions_df["instance"] == row["raw_instance_name"]]["solution"].iloc[0]
                    df.loc[index, "solver_output_valid"] = row["solver_solution"] == reference_solution
                    print(f"Instance: {row['instance']}, Solver Solution: {row['solver_solution']}, Reference Solution: {reference_solution}, Valid: {df.loc[index, 'solver_output_valid']}")

            return df


def load_solver_solutions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Load the solutions from the solver results
    """
    # Create a copy of the dataframe to avoid inplace modification
    df_copy = df.copy()
    # iterate over the rows of the dataframe and load the solver solutions from the result_path
    for index, row in df_copy.iterrows():
        result_path = row["result_path"]
        # load the solution from the result_path
        df_copy.loc[index, "solver_solution"] = read_solver_solution_file(result_path)
    return df_copy

def read_solver_solution_file(result_path: str) -> str:
    """
    Read the solution from the solver result file
    """
    with open(result_path, "r") as file:
        # Read the first line of the file
        solution = file.readline().strip()
        # Check if the solution is YES or NO
        if solution not in ["YES", "NO"]:
            return "INVALID"
    return solution



def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataframe from rows with exit_with_error and timed_out
    """
    df["exit_with_error"] = df["exit_with_error"].astype(bool)
    df["timed_out"] = df["timed_out"].astype(bool)
    df_clean = df[(df["exit_with_error"] == False) & (df["timed_out"] == False)]
    return df_clean

def filter_for_task(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """
    Filter the dataframe for a specific task
    """
    return df[df["task"].str.contains(task)]

class ValidateSolutionsAccaptanceTasks(Callback):
    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
            # Initialize HydraConfig if not already set

        # Create directory for the validation results
        os.makedirs(config['result_validation_output_dir'], exist_ok=True)

        result_file = config["combined_results_file"]
        if not os.path.exists(result_file):
            print(f"File {result_file} does not exist")
            return None
        combined_results_df = pd.read_csv(result_file)

        # Clean the dataframe from rows with exit_with_error and timed_out
        combined_results_df_clean = clean_df(combined_results_df)
        # # Filter df for DC and DS tasks
        accaptance_df = combined_results_df_clean[
            combined_results_df_clean["task"].str.contains("DC|DS")
        ]

        accaptance_df = filter_for_task(combined_results_df_clean, "DC|DS")
        validated_solutions_df = accaptance_df.groupby(['benchmark_name','task','solver_name']).apply(lambda _df : validate_solver_results(_df,config))

        # >>> validated_solutions_df has the new column "solver_output_valid" <<<
        # It also has all the same identifying columns, but only for DC/DS rows.

        # 4. Merge the validated results back
        #    (Adjust the identifying columns to match your real case.)
        identifying_cols = ['benchmark_name', 'task', 'solver_name', 'instance']  # Example

        # If "validated_solutions_df" still has a groupby index, reset it
        # so that the identifying columns are actual columns, not just index levels.
        validated_solutions_df = validated_solutions_df.reset_index(drop=True)

        # Keep only the identifying columns + the new "solver_output_valid" column.
        validated_solutions_subset = validated_solutions_df[
            identifying_cols + ['solver_output_valid']
        ]
        # # Merge with "left" join on the big combined_results_df
        # # so that we only update rows that exist in validated_solutions_df
        merged_df = pd.merge(
            combined_results_df,
            validated_solutions_subset,
            on=identifying_cols,
            how="left",
            suffixes=("", "_validated")  # handle column name collisions
        )

        merged_df.to_csv(config['result_validation_combined_results_file'], index=False)



        # 5. Combine old and new values of solver_output_valid
        # If a row was in validated_solutions_df, the merged column
        # "solver_output_valid_validated" will have True/False.
        # If not, it will have NaN. We overwrite only the matching rows:
        merged_df['solver_output_valid'] = merged_df['solver_output_valid_validated'].fillna(
            merged_df['solver_output_valid']  # The original (could be None, or default)
        )

        # We do not need the helper column anymore
        merged_df.drop(columns=['solver_output_valid_validated'], inplace=True)



        # 6. Write out the final updated dataframe
        merged_df.to_csv(config['combined_results_file'], index=False)




def check_solution_content_acceptance(df: pd.DataFrame, task: str) -> pd.Series:

    if "DC" in task or "DS" in task:
        valid_solutions = df["solution"].isin(["YES", "NO"])
        return valid_solutions

    return pd.Series([False] * len(df), index=df.index)


def load_reference_solutions(
    directory: str,
    task: str,
    expected_instances=None,
    missing_file: str = "missing_instances.txt",
) -> pd.DataFrame:
    """
    Reads all files in 'directory' that follow the naming convention:
        <anything>_<task>.sol
    For example, if task="DS-CO", it will match:
        BA_23423_234234_DS-CO.sol
    The part before "_<task>.sol" is treated as the 'instance' name.
    Parameters
    ----------
    directory : str
        Path to the directory containing the solution files.
    task : str
        The task name used in the file naming convention (<anything>_<task>.sol).
    expected_instances : list or set, optional
        A list or set of instance names you expect. If provided, any of these
        that are not found in the directory will be written out as "missing."
        If None, we'll simply parse whatever is present in the directory.
    missing_file : str, optional
        Filename (or full path) to write the missing instance names.
    Returns
    -------
    pd.DataFrame
        A DataFrame with columns: [instance, solution].
        'instance' is derived from the filename, and 'solution' is the file contents.
    """
    # Regex to match filenames like <anything>_<task>.sol, where <anything> can be any string
    #   '(.+)' captures everything until '_<task>.sol'
    print(f"Looking for files in {directory} with pattern '(.+)_{task}.sol'")
    pattern = re.compile(rf"^(.+)_{re.escape(task)}\.sol$")
    if expected_instances is None:
        expected_instances = set()
    found_data = []
    found_instances = set()
    # Look through the directory
    for fname in os.listdir(directory):
        match = pattern.match(fname)
        if match:
            # group(1) = everything before `_<task>.sol`
            instance_name = match.group(1)
            file_path = os.path.join(directory, fname)
            with open(file_path, "r") as f:
                content = f.read().strip()
            found_data.append({"instance": instance_name, "solution": content})
            found_instances.add(instance_name)
    # Convert results to a DataFrame
    print(f"Found {len(found_data)} solutions.")
    df = pd.DataFrame(found_data, columns=["instance", "solution"])
    # Check solutions are strictly YES or NO
    valid_solutions = check_solution_content_acceptance(df, task)
    print(f"Valid solutions: {valid_solutions.sum()}/{len(valid_solutions)}")
    if not valid_solutions.empty and not valid_solutions.all():
        invalid = df[~valid_solutions]
        print("WARNING: Some solutions are not 'YES' or 'NO':")
        print(invalid)
        # Or raise an error: raise ValueError("Some solutions are not 'YES' or 'NO'.")
    # Identify missing instances if we have an expected set
    print(f"Expected instances: {len(expected_instances)}")
    if expected_instances is not None and len(expected_instances) > 0:
        expected_set = set(os.path.splitext(os.path.basename(instance))[0] for instance in expected_instances)
        missing = sorted(expected_set - found_instances)
        if missing is not None and len(missing) > 0:
            print(
                f"{len(missing)} expected instance(s) are missing. Writing them to {missing_file}"
            )
            #missing_path = os.path.join(directory, missing_file)
            with open(missing_file, "w") as fp:
                for m in missing:
                    fp.write(m + "\n")
        else:
            print("No missing instance files.")
    return df
