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
    # Gain recalibré (0.05 -> 0.03) : dans D, Nd n'entre plus qu'à moitié poids (sigmoïde à U=0),
    # contrairement à l'ancienne formule p_lick=(E+Nd)*M où Nd comptait à plein poids. Sans ce
    # recalibrage, le bruit seul suffisait à dépasser le seuil dès t=0 (souris naïve qui lèche
    # immédiatement) — cf. exigence du PDF "Nd calibré tel qu'une souris naïve ne lèche pas".
    # Vérifié empiriquement (grid search, 15+ runs) : gain=0.03 + seuil=0.35 => 0 lick spontané
    # avant le forced reward de FL1 sur la quasi-totalité des seeds, et un apprentissage WDT qui
    # démarre de façon fiable (le dual-stim a besoin d'un peu plus de bruit que le mono car
    # l'exploration se partage entre 2 stimuli au lieu d'1 ; à 0.025 quelques runs sur 10
    # restaient bloqués près de 0% de hit rate).
    noise: Tuple[float, float, float] = (1.2, 3.0, 0.03)   # (a, b, gain) — bruit Nd (exploration)
    lick_thrs: float = 0.35                                 # Threshold: la souris lick si D > Threshold

    # --- Decision function D = [E/(1+e^{A.U}) + Nd/(1+e^{-A.U})] . [V.M - C]
    decision_slope: float = 5.0                 # "A" : pente de transition exploitation/exploration (fn de U)
    # V et C PROVISOIREMENT NEUTRALISÉS (V=1, C=0) en attendant l'implémentation dynamique du
    # collègue : D se réduit à [E.w_exploit + Nd.w_explore] . M. À remettre à V=1 / C=0.8
    # (valeurs d'init du PDF) dès que les vraies fonctions Value/Cost seront prêtes.
    value: float = 1.0                          # V : valeur de l'outcome (neutralisé, cf. implémentation collègue)
    cost: float = 0.0                           # C : coût de l'action (neutralisé, cf. implémentation collègue)

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

    # --- Uncertainty function U (cahier des charges: U_{t+1} = U_t + A*(2*|RPE_t|-1)^3, à chaque lick)
    uncertainty_gain: float = 0.1               # "A" dans la formule de U (facteur d'échelle, dans [0,1])
    uncertainty_max: float = 1.0                # U_max : borne haute de U (U reste dans [0, U_max])

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
    p_lick: np.ndarray = field(default_factory=lambda: np.array([]))  # gardé pour compat plots, contient D
    decision: np.ndarray = field(default_factory=lambda: np.array([]))  # D: valeur de la fonction de décision par bin
    lick: np.ndarray = field(default_factory=lambda: np.array([]))

    eligibility: np.ndarray = field(default_factory=lambda: np.array([]))

    eligibility1: np.ndarray = field(default_factory=lambda: np.array([]))
    eligibility2: np.ndarray = field(default_factory=lambda: np.array([]))

    uncertainty: np.ndarray = field(default_factory=lambda: np.array([]))  # U: confidence in the internal model, [0, U_max]

    non_rew_lick_cnt: int = 0

    def initialize(self, session_length: int, init_motivation: float, init_expectation: float = 0.0,
                   init_uncertainty: float = 0.0):
        self.expectation = np.full((session_length, 1), float(init_expectation))
        self.motivation = np.zeros((session_length, 1)); self.motivation[0, 0] = init_motivation
        # U est un "state" (comme Expectation, cf. slide 6) : pas de forgetting entre sessions,
        # on repart du dernier niveau de confiance atteint (sauf pour la toute 1ere session, U=0).
        self.uncertainty = np.full((session_length, 1), float(init_uncertainty))
        self.decision = np.zeros((session_length, 1))
        self.rpe = np.zeros((session_length, 1))
        self.p_lick = np.zeros((session_length, 1))
        self.lick = np.zeros((session_length, 1))
        self.eligibility = np.zeros((session_length, 1))
        self.eligibility1 = np.zeros((session_length, 1))
        self.eligibility2 = np.zeros((session_length, 1))
        self.non_rew_lick_cnt = 0
