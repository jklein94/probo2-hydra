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
    benchmark_config = OmegaConf.load(os.path.join(config.configs_output_dir, "benchmark_config", f"{benchmark_name}_config.yaml"))

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
                    print(f"Invalid solution for instance: {row['instance']}")
                    with open(os.path.join(config.result_validation_output_dir, f"{solver_name}_{benchmark_name}_{task}_invalid_solution.txt"), "a") as file:
                        file.write(row["instance"] + "\n")
                else:
                    # get the reference solution for the instance
                    reference_solution = reference_solutions_df[reference_solutions_df["instance"] == row["raw_instance_name"]]["solution"].iloc[0]
                    df.loc[index, "solver_output_valid"] = row["solver_solution"] == reference_solution
                    #print(f"Instance: {row['instance']}, Solver Solution: {row['solver_solution']}, Reference Solution: {reference_solution}, Valid: {df.loc[index, 'solver_output_valid']}")

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
        utils.create_callback_chain(config.hydra.callbacks, 'validate_acceptance_tasks')
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


        if config['update_solver_output_valid_flag']:
            # 6. Write out the final updated dataframe
            merged_df.to_csv(config['combined_results_file'], index=False)




def check_solution_content_acceptance(df: pd.DataFrame, task: str) -> pd.Series:

    if "DC" in task or "DS" in task:
        valid_solutions = df["solution"].isin(["YES", "NO"])
        return valid_solutions

    return pd.Series([False] * len(df), index=df.index)

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()

# def load_reference_solutions(
#     directory: str,
#     task: str,
#     expected_instances=None,
#     missing_file: str = "missing_instances.txt",
# ) -> pd.DataFrame:
#     """
#     Reads all files in 'directory' that follow the naming convention:
#         <anything>_<task>.sol
#     For example, if task="DS-CO", it will match:
#         BA_23423_234234_DS-CO.sol
#     The part before "_<task>.sol" is treated as the 'instance' name.
#     """

#     # Display header panel
#     console.print(Panel.fit(f"[bold blue]🔍 Loading Reference Solutions[/bold blue]", border_style="blue"))

#     pattern = re.compile(rf"^(.+)_{re.escape(task)}\.sol$")
#     if expected_instances is None:
#         expected_instances = set()

#     found_data = []
#     found_instances = set()

#     # Show directory and pattern being searched
#     console.print(f"[bold cyan]📂 Searching in:[/bold cyan] {directory}")
#     console.print(f"[bold cyan]🔎 File pattern:[/bold cyan] '(.+)_{task}.sol'")

#     # Look through the directory
#     for fname in os.listdir(directory):
#         match = pattern.match(fname)
#         if match:
#             instance_name = match.group(1)
#             file_path = os.path.join(directory, fname)

#             with open(file_path, "r") as f:
#                 content = f.read().strip()

#             found_data.append({"instance": instance_name, "solution": content})
#             found_instances.add(instance_name)

#     # Print results
#     console.print(f"[bold cyan]✅ Found solutions:[/bold cyan] {len(found_data)}")

#     # Convert results to a DataFrame
#     df = pd.DataFrame(found_data, columns=["instance", "solution"])

#     # Check solutions are strictly YES or NO
#     valid_solutions = check_solution_content_acceptance(df, task)
#     valid_count = valid_solutions.sum()

#     console.print(f"[bold green]✔ Valid solutions:[/bold green] {valid_count} / {len(valid_solutions)}")

#     if not valid_solutions.empty and not valid_solutions.all():
#         console.print("[bold red]⚠ WARNING:[/bold red] Some solutions are not 'YES' or 'NO':")

#         # Create a table to show invalid entries
#         invalid_table = Table(show_header=True, header_style="bold red")
#         invalid_table.add_column("Instance", style="cyan", justify="left")
#         invalid_table.add_column("Solution", style="red", justify="left")

#         for _, row in df[~valid_solutions].iterrows():
#             invalid_table.add_row(row["instance"], row["solution"])

#         console.print(invalid_table)

#     # Expected instance count
#     console.print(f"[bold cyan]📌 Expected instances:[/bold cyan] {len(expected_instances)}")

#     # Identify missing instances
#     if expected_instances is not None and len(expected_instances) > 0:
#         expected_set = set(os.path.splitext(os.path.basename(instance))[0] for instance in expected_instances)
#         missing = sorted(expected_set - found_instances)

#         if missing:
#             console.print(f"[bold yellow]⚠ Missing instances:[/bold yellow] {len(missing)}")

#             # Write missing instances to file
#             with open(missing_file, "w") as fp:
#                 for m in missing:
#                     fp.write(m + "\n")

#             console.print(f"[bold cyan]📝 Missing instances written to:[/bold cyan] {missing_file}")
#         else:
#             console.print("[bold green]✔ No missing instance files.[/bold green]")

#     # Return the DataFrame
#     return df

def load_reference_solutions(
    directory: str,
    task: str,
    expected_instances=None,
    missing_file: str = "missing_instances.txt",
) -> pd.DataFrame:
    """
    Reads all files in 'directory' that follow the naming convention:
        <anything>_<task>.sol
    The part before "_<task>.sol" is treated as the 'instance' name.
    """

    # Header with Monokai-inspired styling
    #console.print(Panel.fit("[bold magenta]Loading Reference Solutions[/bold magenta]", border_style="bright_magenta"))

    pattern = re.compile(rf"^(.+)_{re.escape(task)}\.sol$")
    if expected_instances is None:
        expected_instances = set()

    found_data = []
    found_instances = set()

    # Show directory and pattern being searched
    console.print(f"[bold bright_cyan]Directory:[/bold bright_cyan] [bright_white]{directory}[/bright_white]")
    console.print(f"[bold bright_cyan]File pattern:[/bold bright_cyan] [bright_white]'(.+)_{task}.sol'[/bright_white]")

    # Look through the directory
    for fname in os.listdir(directory):
        match = pattern.match(fname)
        if match:
            instance_name = match.group(1)
            file_path = os.path.join(directory, fname)

            with open(file_path, "r") as f:
                content = f.read().strip()

            found_data.append({"instance": instance_name, "solution": content})
            found_instances.add(instance_name)

    # Print results
    console.print(f"[bold bright_blue]Found solutions:[/bold bright_blue] [bright_white]{len(found_data)}[/bright_white]")

    # Convert results to a DataFrame
    df = pd.DataFrame(found_data, columns=["instance", "solution"])

    # Check solutions are strictly YES or NO
    valid_solutions = check_solution_content_acceptance(df, task)
    valid_count = valid_solutions.sum()

    console.print(f"[bold bright_green]Valid solutions:[/bold bright_green] [bright_white]{valid_count} / {len(valid_solutions)}[/bright_white]")

    if not valid_solutions.empty and not valid_solutions.all():
        console.print("[bold bright_red]WARNING: Some solutions are not 'YES' or 'NO'[/bold bright_red]")

        # Create a table to show invalid entries
        invalid_table = Table(show_header=True, header_style="bold bright_red")
        invalid_table.add_column("Instance", style="bright_magenta", justify="left")
        invalid_table.add_column("Solution", style="bright_red", justify="left")

        for _, row in df[~valid_solutions].iterrows():
            invalid_table.add_row(row["instance"], row["solution"])

        console.print(invalid_table)

    # Expected instance count
    console.print(f"[bold bright_yellow]Expected instances:[/bold bright_yellow] [bright_white]{len(expected_instances)}[/bright_white]")

    # Identify missing instances
    if expected_instances is not None and len(expected_instances) > 0:
        expected_set = set(os.path.splitext(os.path.basename(instance))[0] for instance in expected_instances)
        missing = sorted(expected_set - found_instances)

        if missing:
            console.print(f"[bold bright_red]Missing instances:[/bold bright_red] [bright_white]{len(missing)}[/bright_white]")

            # Write missing instances to file
            with open(missing_file, "w") as fp:
                for m in missing:
                    fp.write(m + "\n")

            console.print(f"[bold bright_cyan]Missing instances written to:[/bold bright_cyan] [bright_white]{missing_file}[/bright_white]")
        else:
            console.print("[bold bright_green]No missing instance files.[/bold bright_green]")

    # Return the DataFrame
    return df