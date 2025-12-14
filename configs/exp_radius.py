# configs/exp_radius.py
import numpy as np
from dataclasses import dataclass, asdict


"""""
DO NOT FORGET TO EXPLAIN THE DATA GENERATION IN THE EXPERIMENTS SECTION OFTHE PAPER
"""

@dataclass(frozen=True)
class ExpConfig:
    # randomness
    base_seed: int = 123

    # data
    n_total: int = 10_000
    d: int = 10
    n_train_all: tuple = (2*d,)    
    replications: int = 10

    # experiment knobs
    n_rs: int = 10
    p: str = "inf"
    noise_mag: float = 0.1
    sparsity: int = max([int(d / 2), 1])
    label_noise: float = 0.2
    beta_constrained: bool = False

    # grid for the radius 
    c_rs_start_exp: float = -5
    c_rs_stop_exp: float = 1

    # theta1 grid specification for cross validation
    theta1_log_start_exp: float = -5
    theta1_log_stop_exp: float = 0
    theta1_log_num: int = 10
    theta1_nmbrs_start: int = 1
    theta1_nmbrs_stop: int = 10
    theta1_nmbrs_from_index: int = 2
    theta1_tail: tuple = (10.0, 1e2, 1e3, 1e4, 1e5)

    n_train_multipliers: tuple = (3, 4, 5) # (1, 2, 3, 4, 5)

    def build_n_train_all(self):
        return tuple(int(self.d * m) for m in self.n_train_multipliers)


    def build_grids(self):
        c_rs = np.logspace(self.c_rs_start_exp, self.c_rs_stop_exp, self.n_rs)

        nmbrs = np.arange(self.theta1_nmbrs_start, self.theta1_nmbrs_stop, 1)
        theta1s = np.hstack([
            1.0 + np.logspace(self.theta1_log_start_exp, self.theta1_log_stop_exp, self.theta1_log_num),
            nmbrs[self.theta1_nmbrs_from_index:],
            np.array(self.theta1_tail, dtype=float),
        ])
        return c_rs, theta1s

    def tag(self) -> str:
        # a filesystem safe descriptive name
        p_tag = str(self.p)
        bc_tag = "bc1" if self.beta_constrained else "bc0"
        return (
            f"beta_constrained_{self.beta_constrained}"
            f"_radius_d{self.d}"
            f"_N{self.n_total}"
            f"_train{','.join(map(str, self.n_train_all))}"
            f"_reps{self.replications}"
            f"_nrs{self.n_rs}"
            f"_p{p_tag}"
            f"_noise{self.noise_mag}"
            f"_spars{self.sparsity}"
            f"_ln{self.label_noise}"
            f"_{bc_tag}"
            f"_seed{self.base_seed}"
        )

def to_jsonable_dict(cfg: ExpConfig) -> dict:
    # useful to dump alongside results
    d = asdict(cfg)
    d["n_train_all"] = list(cfg.n_train_all)
    d["theta1_tail"] = list(cfg.theta1_tail)
    return d
