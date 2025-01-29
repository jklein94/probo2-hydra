# Managing Solvers

Solvers are managed via YAML configuration files. Each solver requires its own dedicated file, ensuring modularity and maintainability.


## Example Solver YAML File

Below is an example of a solver configuration file in YAML format:

```yaml
name: MySolver # Name of the solver must be unique!
version: ICCM23
path: /full/path/to/MySolver # Path to the solver executable or script (bash or python)
format: tgf # Supported formats: tgf, apx, i23
interface: legacy # Supported interfaces: legacy (ICCMA15-21 interface), i23  (ICCMA23 interface)
argument: [paramA, paramB] # Optional: List of arguments for the solver
```

### Description of YAML Fields

- **`name`** *(string, required)*: A unique identifier for the solver.
- **`version`** *(string, optional)*: Specifies the solver version (e.g., ICCM23).
- **`path`** *(string, required)*: The absolute path to the solver executable or script.
- **`format`** *(string, required)*: Defines the supported input format(s). Supported values:
  - `tgf` (Trivial Graph Format)
  - `apx` (Argumentation Framework Format)
  - `i23` (ICCMA23 Format)
- **`interface`** *(string, required)*: Determines the interface the solver adheres to. Options:
  - `legacy` (ICCMA15-21 interface)
  - `i23` (ICCMA23 interface)
- **`argument`** *(list, optional)*: An optional list of arguments that can be passed to the solver during execution.

## Adding a New Solver

To integrate a new solver into the system:
1. Create a new YAML file inside `configs/solver/`.
2. Define the solver's configuration using the structure described above.
3. (Optional) Organize solvers into subdirectories, e.g., `configs/solver/MySolverCollection/`, for better management.

Below you can see a example directory structure for the solver config files:
```
configs/
├── solver/
│   ├── MySolver.yaml
│   ├── AnotherSolver.yaml
│   ├── MySolverCollection/
│   │   ├── CustomSolver.yaml
```

**Note**: If you want to use a solver file in your experiment which is located in a subfolder, you must also specify the subfolder. For the example above, the relative path `MySovlerCollection/CustomSolver.yaml` would have to be specified in order to use `CustomSolver.yaml`. See the documentation on [Experiments](_experiments.md)

## Guidelines for Solver Config Files

- Each solver must have a separate YAML configuration file.
- Solver names **must be unique** as they serve as identifiers within the system.
- If multiple solvers share the same name, their results will be **combined**.
- YAML files should be placed inside the `configs/solver` directory.
- You may organize solvers into subdirectories, e.g., `configs/solver/MySolverCollection`.
-

