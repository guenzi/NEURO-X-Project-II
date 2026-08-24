import numpy as np
import random
from dataclasses import dataclass, field
from typing import Tuple, List

def sample_uniform_range(min_val, max_val):
    return min_val + (max_val - min_val) * random.random()


# ======================================================================
# Session info
# ======================================================================
@dataclass
class SessionBaseInfo:
    duration: float = 60          # minutes
    resolution: float = 0.1       # seconds
    number_bin: int = field(init=False)
    time_vector: np.ndarray = field(init=False)

    def __post_init__(self):
        # 0 .. (duration*60 - resolution) avec pas = resolution
        self.time_vector = np.arange(0.0, self.duration * 60, self.resolution)
        self.number_bin = len(self.time_vector)


# ======================================================================
# Mouse
# ======================================================================
@dataclass
class Mouse:
    noise: Tuple[float, float, float] = (1.2, 3.0, 0.05)  # (a, b, gain)
    lick_thrs: float = 1.0

    motivation: Tuple[float, float] = (1.0, 0.003)        # (init, delta par reward)

    exp_update_reward: Tuple[float, float] = (2000.0, 0.2)  # (tau, gain)
    exp_update_no_reward: Tuple[float, float] = (4.0, 0.4)  # (tau, gain)

    exp_update_stim_tau: float = 1.0
    stim_gain1: float = 0.10   # whisker
    stim_gain2: float = 0.00   # auditif

    # 
    learning_nonrew_lick: Tuple[int, float] = (2, 1.0)
    learning_stim: float = 0.015
    tau_eligibility: float = 2.0

    noise_adapt_enable: bool = True
    noise_target_rpe: float = 0.20
    noise_window_bins: int = 5
    noise_update_period_bins: int = 5
    noise_alpha: float = 0.3
    noise_gain_bounds: Tuple[float, float] = (0.01, 0.20)

    # --- Alias de compatibilité pour les fonctions: exp_update_stim1/2 = (tau, gain)
    @property
    def exp_update_stim1(self) -> Tuple[float, float]:
        return (float(self.exp_update_stim_tau), float(self.stim_gain1))

    @exp_update_stim1.setter
    def exp_update_stim1(self, val: Tuple[float, float]) -> None:
        tau, gain = val
        self.exp_update_stim_tau = float(tau)     # un seul tau partagé
        self.stim_gain1 = float(gain)

    @property
    def exp_update_stim2(self) -> Tuple[float, float]:
        return (float(self.exp_update_stim_tau), float(self.stim_gain2))

    @exp_update_stim2.setter
    def exp_update_stim2(self, val: Tuple[float, float]) -> None:
        tau, gain = val
        self.exp_update_stim_tau = float(tau)     # un seul tau partagé
        self.stim_gain2 = float(gain)


# ======================================================================
# Free-licking params & state
# ======================================================================
@dataclass
class FreeLickingSessionParams:
    no_lick_wind: Tuple[float, float] = (3.0, 3.0)
    response_wind: Tuple[float, float] = (1.0, 1.0)  # placeholder, non utilisé en FL
    reward_prob: float = 1.0
    reward_size: float = 1.0
    forced_reward: Tuple[int, float] = (1, 500.0)

@dataclass
class FreeLickingSessionState:
    last_lick_time: float = 0.0
    last_reward_time: float = 0.0
    no_lick_wind: float = 0.0
    reward: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))
    stim1: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))

    def initialize(self, session_length: int, no_lick_range: Tuple[float, float]):
        self.last_lick_time = 0.0
        self.last_reward_time = 0.0
        self.no_lick_wind = sample_uniform_range(*no_lick_range)
        self.reward = np.zeros((session_length, 1))
        self.stim1 = np.zeros((session_length, 1))


# ======================================================================
# Two-stim detection params & state
# ======================================================================
@dataclass
class WDTSesssionParams:
    no_lick_wind: Tuple[float, float] = (3.0, 3.0)
    iti: Tuple[float, float] = (6.0, 12.0)
    response_wind: float = 1.0
    reward_prob: float = 1.0
    reward_size: float = 1.0

    trial_types: List[float] = field(default_factory=lambda: [0.0, 1.0])

    two_stim_mix: bool = False
    mix_probs: Tuple[float, float, float] = (1/3, 1/3, 1/3)  # (p_catch, p_stim1, p_stim2)
    stim1_amp: float = 1.0
    stim2_amp: float = 1.0

    trial_kinds: List[int] = field(default_factory=lambda: [0, 1, 2])  # 0/1/2
    reward_stim: int = 1  # 1 = whisker (sessions 1..5), 2 = audio (sessions 6..10)

    block_numb: int = 5

    # --- Alias de compatibilité: rewarded_stim <-> reward_stim
    @property
    def rewarded_stim(self) -> int:
        return int(self.reward_stim)

    @rewarded_stim.setter
    def rewarded_stim(self, v: int) -> None:
        self.reward_stim = int(v)


@dataclass
class WDTSesssionState:
    last_lick_time: float = 0.0
    last_reward_time: float = 0.0
    last_stim_time: float = 0.0
    last_trial_time: float = 0.0
    trial_times: list = field(default_factory=list)

    no_lick_wind: float = 0.0
    iti: float = 0.0

    reward_window: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))
    reward: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))

    stim1: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))
    stim2: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))

    trial_kind: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=int))

    last_trial_kind: int = 0

    def initialize(self, session_length: int, no_lick_range: Tuple[float, float], iti_range: Tuple[float, float]):
        self.no_lick_wind = sample_uniform_range(*no_lick_range)
        self.iti = sample_uniform_range(*iti_range)
        self.reward_window = np.zeros((session_length, 1))
        self.reward = np.zeros((session_length, 1))
        self.stim1 = np.zeros((session_length, 1))
        self.stim2 = np.zeros((session_length, 1))
        self.trial_kind = np.zeros((session_length, 1), dtype=int)
        self.trial_times = []
        self.last_lick_time = 0.0
        self.last_reward_time = 0.0
        self.last_stim_time = 0.0
        self.last_trial_time = 0.0
        self.last_trial_kind = 0


# ======================================================================
# Mouse session state (single expectation + eligibilités)
# ======================================================================

@dataclass
class MouseSessionState:
    expectation: np.ndarray = field(default_factory=lambda: np.array([]))
    motivation: np.ndarray = field(default_factory=lambda: np.array([]))
    rpe: np.ndarray = field(default_factory=lambda: np.array([]))
    p_lick: np.ndarray = field(default_factory=lambda: np.array([]))
    lick: np.ndarray = field(default_factory=lambda: np.array([]))

    eligibility: np.ndarray = field(default_factory=lambda: np.array([]))

    eligibility1: np.ndarray = field(default_factory=lambda: np.array([]))
    eligibility2: np.ndarray = field(default_factory=lambda: np.array([]))

    non_rew_lick_cnt: int = 0

    def initialize(self, session_length: int, init_motivation: float, init_expectation: float = 0.0):
        self.expectation = np.full((session_length, 1), float(init_expectation))
        self.motivation = np.zeros((session_length, 1)); self.motivation[0, 0] = init_motivation
        self.rpe = np.zeros((session_length, 1))
        self.p_lick = np.zeros((session_length, 1))
        self.lick = np.zeros((session_length, 1))
        self.eligibility = np.zeros((session_length, 1))
        self.eligibility1 = np.zeros((session_length, 1))
        self.eligibility2 = np.zeros((session_length, 1))
        self.non_rew_lick_cnt = 0
