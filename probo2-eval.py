import hydra
from omegaconf import DictConfig, OmegaConf
from src import solver_interfaces
from src import utils, config_validater
from hydra.core.utils import JobReturn
from typing import Any, List
from hydra.experimental.callback import Callback

from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from pathlib import Path

import os

import tqdm




@hydra.main(
    version_base=None, config_path="configs", config_name="config"
)
def run_evaluation_pipeline(cfg: DictConfig) -> None:
    # Generate custom directories
    pass



if __name__ == "__main__":
    run_evaluation_pipeline()
