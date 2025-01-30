# Plotting


## Cactus Plots

Cactus plots are a common way to visualize solver performance across a set of instances. They’re frequently used in fields like SAT solving, constraint programming, optimization, and related areas where you run a solver on multiple benchmarks.




```yaml
hydra:
  callbacks:
    cactus_plot:
      _target_: src.callbacks.plotting_callbacks.CactusPlot
      combine_runs_multiple_runs: False
      runtime_key: user_sys_time # perfcounter_time or user_sys_time
```

- Two important options: (1) **combine_runs_multiple_runs** and (2) **runtime_key**
- If **combine_runs_multiple_runs** is true, the results of multiple runs are average
  - Instances that are not valid for one of the runs are excluded