import numpy as np
import random
from dataclasses import dataclass, field



def sample_uniform_range(min_val, max_val):                     # Redefined here to avoid circular dependencies
    return min_val + (max_val - min_val) * random.random()



@dataclass
class SessionBaseInfo:
    duration: float = 60  # in minutes                 # session length (minutes)
    resolution: float = 0.1  # in seconds              # time step (bin size) in seconds
    number_bin: int = field(init=False)                # number of time bins (computed in __post_init__)
    time_vector: np.ndarray = field(init=False)        # time vector (computed in __post_init__)

    def __post_init__(self):
        self.time_vector = np.arange(0.0, self.duration * 60, self.resolution)
        self.number_bin = len(self.time_vector)




@dataclass
class Mouse:
    # Gain recalibré (0.07 -> 0.022) : dans D, Nd n'entre plus qu'à moitié poids (sigmoïde à U=0),
    # contrairement à l'ancienne formule p_lick=(E+Nd)*M où Nd comptait à plein poids. Sans ce
    # recalibrage, le bruit seul suffisait à dépasser le seuil dès t=0 (souris naïve qui lèche
    # immédiatement) — cf. exigence du PDF "Nd calibré tel qu'une souris naïve ne lèche pas".
    # Vérifié empiriquement (grid search) : gain=0.022 + seuil=0.35 => 0 lick spontané avant le
    # forced reward de FL1 (sur plusieurs seeds) tout en laissant l'apprentissage WDT démarrer.
    noise: tuple = (1.2, 3, 0.022)              # Gamma noise: (shape a, scale b, gain) — bruit Nd (exploration)
    lick_thrs: float = 0.35                     # Threshold: la souris lick si D > Threshold

    # --- Decision function D = [E/(1+e^{A.U}) + Nd/(1+e^{-A.U})] . [V.M - C]
    decision_slope: float = 5.0                 # "A" : pente de transition exploitation/exploration (fn de U)
    # V et C PROVISOIREMENT NEUTRALISÉS (V=1, C=0) en attendant l'implémentation dynamique du
    # collègue : D se réduit à [E.w_exploit + Nd.w_explore] . M, càd le comportement sans coût
    # ni valeur, motivation seule. À remettre à V=1 / C=0.8 (valeurs d'init du PDF) dès que les
    # vraies fonctions Value/Cost seront prêtes.
    value: float = 1.0                          # V : valeur de l'outcome (neutralisé, cf. implémentation collègue)
    cost: float = 0.0                           # C : coût de l'action (neutralisé, cf. implémentation collègue)
    motivation: tuple = (1.0, 0.003)            # (initial value, loss per reward)
    exp_update_reward: tuple = (2000, 0.2)      # (tau, gain) for expectation update after reward
    exp_update_no_reward: tuple = (4, 0.4)      # (tau, gain) for update after lick without reward
    exp_update_stim: tuple = (1, 0.1)           # (tau, gain) for update after a stimulus
    learning_nonrew_lick: tuple = (2, 1)        # (threshold of non-reward licks, tau increment)
    learning_stim: float = 0.015                # learning rate for stimulus gain (scaled by RPE * eligibility)
    tau_eligibility: float = 2.0                # eligibility trace time constant (seconds)

    noise_adapt_enable: bool = True             # active/désactive la feature
    noise_target_rpe: float = 0.20              # |RPE| "attendu" (0..1)
    noise_window_bins: int = 5                  # moyenne sur les 5 derniers RPE
    noise_update_period_bins: int = 5           # met à jour tous les 5 bins
    noise_alpha: float = 0.3                    # pas d'adaptation (0.1..0.4 raisonnable)
    noise_gain_bounds: tuple = (0.01, 0.20)     # [gain_min, gain_max] pour rester stable

    # --- Uncertainty function U (cahier des charges: U_{t+1} = U_t + A*(2*|RPE_t|-1)^3, à chaque lick)
    uncertainty_gain: float = 0.1               # "A" dans la formule de U (facteur d'échelle, dans [0,1])
    uncertainty_max: float = 1.0                # U_max : borne haute de U (U reste dans [0, U_max])



@dataclass
class FreeLickingSessionParams:
    no_lick_wind: tuple = (3, 3)                # (min, max) seconds: required no-lick window before reward eligibility
    response_wind: tuple = (1, 1)               # (unused here for FL) placeholder response window
    reward_prob: float = 1.0                    # probability to deliver reward when conditions are met
    reward_size: float = 1.0                    # reward magnitude
    forced_reward: tuple = (1, 500)             # (on/off, time in seconds) force a reward at a specific time



@dataclass
class WDTSesssionParams:
    no_lick_wind: tuple = (3, 3)                                    # (min, max) seconds: no-lick window before starting a trial
    iti: tuple = (6, 12)                                            # inter-trial interval range (seconds), sampled uniformly
    response_wind: float = 1.0                                      # response window duration (seconds) after stimulus onset
    reward_prob: float = 1.0                                        # probability of reward if lick occurs within response window
    reward_size: float = 1.0                                        # reward magnitude
    trial_types: list = field(default_factory=lambda: [0, 1.0])     # stimulus amplitudes (0 = catch trial)
    block_numb: int = 5                                             # number of blocks (not used in current code)



@dataclass
class FreeLickingSessionState:
    last_lick_time: float = 0.0                                             # timestamp (s) of last lick
    last_reward_time: float = 0.0                                           # timestamp (s) of last reward
    no_lick_wind: float = 0.0                                               # current no-lick window (s), drawn from range
    reward: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))    # reward time-course
    stim1: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))     # stimulus trace (always 0 in FL)


    def initialize(self, session_length: int, no_lick_range: tuple):
        self.last_lick_time = 0.0                                       # reset last lick time
        self.last_reward_time = 0.0                                     # reset last reward time
        self.no_lick_wind = sample_uniform_range(*no_lick_range)        # draw initial no-lick window
        self.reward = np.zeros((session_length, 1))                     # allocate reward vector (session length)
        self.stim1 = np.zeros((session_length, 1))                      # allocate stimulus vector (kept for consistency even if 0)


@dataclass
class WDTSesssionState:
    last_lick_time: float = 0.0                         # timestamp (s) of last lick
    last_reward_time: float = 0.0                       # timestamp (s) of last reward
    last_stim_time: float = 0.0                         # timestamp (s) of last stimulus
    last_trial_time: float = 0.0                        # timestamp (s) of last trial start
    trial_times: list = field(default_factory=list)     # list of trial onset times (s)
    no_lick_wind: float = 0.0                           # current no-lick window (s)
    iti: float = 0.0                                    # current ITI (s), sampled from range
    reward_window: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))  # response window mask over time
    reward: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))         # reward time-course
    stim1: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))          # stimulus amplitude trace

    def initialize(self, session_length: int, no_lick_range: tuple, iti_range: tuple):
        self.no_lick_wind = sample_uniform_range(*no_lick_range)    # sample initial no-lick window
        self.iti = sample_uniform_range(*iti_range)                 # sample initial ITI
        self.reward_window = np.zeros((session_length, 1))          # allocate response-window mask (0/1)
        self.reward = np.zeros((session_length, 1))                 # allocate reward trace
        self.stim1 = np.zeros((session_length, 1))                  # allocate stimulus trace
        self.trial_times = []                                       # reset trial onset log



@dataclass
class MouseSessionState:
    expectation: np.ndarray = field(default_factory=lambda: np.array([]))  # reward expectation (0–1) per time bin
    motivation: np.ndarray = field(default_factory=lambda: np.array([]))   # current motivation (scales p_lick)
    rpe: np.ndarray = field(default_factory=lambda: np.array([]))          # reward prediction error (reward - expectation)
    p_lick: np.ndarray = field(default_factory=lambda: np.array([]))       # historique: gardé pour compat plots, contient D
    decision: np.ndarray = field(default_factory=lambda: np.array([]))     # D: valeur de la fonction de décision par bin
    lick: np.ndarray = field(default_factory=lambda: np.array([]))         # 0/1 lick emitted at this bin
    eligibility: np.ndarray = field(default_factory=lambda: np.array([]))  # eligibility trace (exponential decay)
    uncertainty: np.ndarray = field(default_factory=lambda: np.array([]))  # U: confidence in the internal model, [0, U_max]
    non_rew_lick_cnt: int = 0

    def initialize(self, session_length: int, init_motivation: float, init_expectation: float = 0.0,
                   init_uncertainty: float = 0.0):
        self.expectation = np.full((session_length, 1), float(init_expectation))    # set initial expectation constant
        self.motivation = np.zeros((session_length, 1))                             # allocate motivation curve
        self.motivation[0, 0] = init_motivation                                     # set motivation at t=0
        self.decision = np.zeros((session_length, 1))                               # allocate D curve
        self.rpe = np.zeros((session_length, 1))                                    # allocate RPE curve
        self.p_lick = np.zeros((session_length, 1))                                 # allocate p(lick) curve
        self.lick = np.zeros((session_length, 1))                                   # allocate lick (0/1) curve
        # U est un "state" (comme Expectation, cf. slide 6) : pas de forgetting entre sessions,
        # on repart du dernier niveau de confiance atteint (sauf pour la toute 1ere session, U=0).
        self.uncertainty = np.full((session_length, 1), float(init_uncertainty))
        self.eligibility = np.zeros((session_length, 1))                            # allocate eligibility trace
        self.non_rew_lick_cnt = 0                                                   # reset non-reward lick counter