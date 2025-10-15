import yaml
from dataclasses import dataclass
from typing import List
import dacite

# -----------------
# Config Structures
# -----------------

@dataclass
class GenerateConfig:
    method: str
    num_of_fibres: int
    poisson_radius: float
    resolution: int
    fibre_r_initial: float
    domain_size_initial: List[float]
    angle_std_dev: float

@dataclass
class LoadConfig:
    name: str
    step: int

@dataclass
class InitializationConfig:
    method: str
    generate: GenerateConfig
    load: LoadConfig

@dataclass
class SpringSystemConfig:
    k_length: float
    k_curvature: float
    k_boundary: float
    k_collision: float

@dataclass
class EvolutionConfig:
    max_iterations: int
    fibre_r_target: float
    fibre_r_steps: float
    fibre_r_std: float
    domain_size_target: List[float]
    domain_size_steps: float
    collision_threshold: float
    apply_pbc: bool

@dataclass
class OptimizationConfig:
    optimizer: str
    learning_rate: float
    grad_clipping: bool
    max_grad_norm: float

@dataclass
class StatsConfig:
    to_plot: bool
    logging_freq: int

@dataclass
class Config:
    initialization: InitializationConfig
    spring_system: SpringSystemConfig
    evolution: EvolutionConfig
    optimization: OptimizationConfig
    stats: StatsConfig

# -----------------
# Loader (automatic with dacite)
# -----------------

def load_config(filepath: str) -> Config:
    with open(filepath, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return dacite.from_dict(data_class=Config, data=raw)