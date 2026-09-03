import numpy as np
import random
from dataclasses import dataclass, field
from typing import Any, List, Optional



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




# Reconstruit directement sur le squelette de Code_mono_stim_v2 (meme Mouse, meme fonction de
# decision D, meme desapprentissage general) — la seule vraie difference structurelle avec le
# mono est qu'ici DEUX stimuli (whisker=stim1, auditif=stim2) alimentent la MEME Expectation
# partagee, chacun avec son propre gain appris (exp_update_stim1/2), au lieu d'un seul stimulus.
# Contrairement au dual "gauche/droite" du collegue (deux Expectations independantes + un choix
# de cote), il n'y a ici qu'UNE seule decision "je leche ou pas" — exactement comme en mono —
# ce qui permet de reproduire le paradigme WDT->AUD (switch de contingence) de l'ancien
# Code_dual_stim, et donc de tester une vraie extinction specifique a un stimulus.
@dataclass
class Mouse:
    noise: tuple = (1.2, 0.4)                   # Gamma noise Nd: (shape a, scale b) — pas de gain separe,
                                                 # son impact sur la decision vient uniquement du poids
                                                 # pilote par Uncertainty (w_explore), pas d'un multiplicateur.
    lick_thrs: float = 1.0                      # threshold for Free Licking: if p_lick >= threshold → emit lick
    lick_thrs_wdt: float = 1.0                  # threshold for WDT sessions (separate scale, e.g. when V/C shrink the gate)
    motivation: tuple = (1.0, 0.003)            # (initial value, loss per reward)
    exp_update_reward: tuple = (2000, 0.3)      # (tau, gain) for expectation update after reward
    exp_update_no_reward: tuple = (4, 0.4)      # (tau, gain) for update after lick without reward
    learning_nonrew_lick: tuple = (2, 1)        # (threshold of non-reward licks, tau increment)
    tau_eligibility: float = 2.0                # eligibility trace time constant (seconds), partagee stim1/stim2

    reward_value: float = 1.0                   # V: value of the reward, gates the decision (V*M - C)
    cost: float = 0.0                           # C: cost of licking, gates the decision (V*M - C)

    # Decision function D = [E/(1+e^{A_D.(U-U0)}) + Nd/(1+e^{-A_D.(U-U0)})] . [V.M - C]
    # Identique au mono : une seule Expectation partagee, une seule decision de lecher.
    decision_slope: float = 5.0                 # "A_D" pour Free Licking: pente de transition exploit/explore
    decision_slope_wdt: float = 3.0             # "A_D" pour WDT/AUD: pente plus douce, apprentissage progressif
    uncertainty_offset: float = 0.5             # "U0": point neutre de la sigmoide — a U=U0, poids 50/50.
    uncertainty_gain: float = 0.1               # "A_U": facteur d'echelle de la mise a jour de U, dans [0,1]
    uncertainty_max: float = 1.0                # U_max: borne haute de U (U reste dans [0, U_max])

    # --- Désapprentissage général (Go/No-Go, identique à Code_mono_stim_v2) -------------------
    # Off par défaut : ne change rien au comportement existant tant que delearning_enable=False.
    # Capte l'impulsivité générale (licks non liés à un stimulus précis), exactement comme en mono.
    delearning_enable: bool = False             # active/désactive toute l'architecture Go/No-Go
    exp_update_nogo: tuple = (8.0, 0.15)        # (tau, gain) pour la voie No-Go générale
    nogo_relief: float = 0.4                    # fraction de E_nogo effacée à chaque récompense confirmée

    # --- Deux stimuli, une seule Expectation partagée ------------------------------------------
    # Chaque stimulus a son propre (tau, gain) d'anticipation — comme l'ancien Code_dual_stim
    # (stim_gain1/stim_gain2) — mais le tau est partagé (un seul mouse.tau_eligibility et un seul
    # tau de decroissance de l'injection, ici fixe a 1s comme en mono) : seul le gain differe et
    # apprend independamment pour chaque stimulus.
    exp_update_stim1: tuple = (1, 0.1)          # (tau, gain) whisker — gain apprend via eligibility1
    exp_update_stim2: tuple = (1, 0.1)          # (tau, gain) auditif — gain apprend via eligibility2
    # 0.1 (et non 0.004 comme en mono) : calibré par grid search (seeds 1-7) pour qu'un switch
    # de contingence sur seulement 5 sessions produise une extinction nette de stim1 (le gain
    # net retombe à ~0 dès AUD6-7) et une réacquisition visible de stim2 (remonte fortement,
    # franchit parfois 0) — à 0.004 les deux gains bougent à peine sur 5 sessions.
    learning_stim: float = 0.1                  # taux d'apprentissage partagé pour les deux gains

    # --- Désapprentissage SPÉCIFIQUE au stimulus (le vrai objectif de ce fichier) --------------
    # Miroir exact de la règle d'apprentissage du gain ci-dessus, mais déclenché par un RPE
    # négatif (lick sur ce stimulus non récompensé) au lieu d'un RPE positif. Le gain net
    # utilisé pour booster l'expectation devient (stim_gain - stim_nogo_gain) : quand un
    # stimulus cesse d'être récompensé (switch de contingence), stim_nogo_gain grimpe et
    # éteint spécifiquement l'effet de CE stimulus, sans toucher l'autre (eligibility séparées).
    # Off automatiquement quand delearning_enable=False (jamais mis à jour dans ce cas).
    stim1_nogo_gain: float = 0.0                # whisker — état mutable, comme exp_update_stim1[1]
    stim2_nogo_gain: float = 0.0                # auditif — état mutable, comme exp_update_stim2[1]
    # Plafond commun aux 4 gains ci-dessus (go1/go2/nogo1/nogo2). Sans lui, go et nogo peuvent
    # s'emballer ensemble sans fin (le gain NET reste petit, mais les deux valeurs brutes
    # grimpent indéfiniment — observé jusqu'à ~9 sur certaines graines avec learning_stim=0.1).
    # Volontairement assez haut (8, pas juste 1.5) : un plafond trop bas fait saturer le gain Go
    # dès la 1ère session (rewardé), après quoi il ne peut plus "compenser" l'accumulation lente
    # du No-Go pendant les sessions suivantes — le gain NET se met alors à décliner tout de
    # suite, y compris PENDANT la phase où le stimulus est encore récompensé, ce qui casse le
    # récit "stable tant que récompensé, puis extinction au switch". Avec un plafond haut, le
    # Go continue de monter sur plusieurs sessions et compense largement le No-Go tant que le
    # stimulus est récompensé — le déclin ne devient visible qu'après le switch de contingence.
    stim_gain_max: float = 8.0
    # Taux d'apprentissage SEPARE pour le No-Go, plus lent que learning_stim (Go) — sinon les
    # deux grimperaient à la même vitesse et le gain NET s'effondrerait dès les premières
    # sessions, avant même le switch de contingence (le stimulus déclenche toujours quelques
    # licks ratés même quand il est correctement récompensé : fenêtre de réponse manquée,
    # cooldown post-récompense). Calibré par grid search (cap=8, learning_stim=0.1) pour que le
    # gain NET reste haut et stable tant que le stimulus est récompensé, puis chute nettement
    # dans les 2-3 sessions suivant le switch — sans jamais retomber tout à fait à zéro.
    learning_stim_nogo: float = 0.02

    noise_adapt_enable: bool = True             # active/désactive la feature (héritage, non utilisé ici)
    noise_target_rpe: float = 0.20
    noise_window_bins: int = 5
    noise_update_period_bins: int = 5
    noise_alpha: float = 0.3
    noise_gain_bounds: tuple = (0.01, 0.20)



@dataclass
class FreeLickingSessionParams:
    no_lick_wind: tuple = (3, 3)                # (min, max) seconds: required no-lick window before reward eligibility
    response_wind: tuple = (1, 1)               # (unused here for FL) placeholder response window
    reward_prob: float = 1.0                    # probability to deliver reward when conditions are met
    reward_size: float = 1.0                    # reward magnitude
    forced_reward: tuple = (1, 500)             # (on/off, time in seconds) force a reward at a specific time



@dataclass
class FreeLickingSessionState:
    last_lick_time: float = 0.0                                             # timestamp (s) of last lick
    last_reward_time: float = 0.0                                           # timestamp (s) of last reward
    no_lick_wind: float = 0.0                                               # current no-lick window (s), drawn from range
    reward: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))    # reward time-course
    stim1: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))     # kept for plotting compat (always 0 in FL)
    stim2: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))     # kept for plotting compat (always 0 in FL)


    def initialize(self, session_length: int, no_lick_range: tuple):
        self.last_lick_time = 0.0                                       # reset last lick time
        self.last_reward_time = 0.0                                     # reset last reward time
        self.no_lick_wind = sample_uniform_range(*no_lick_range)        # draw initial no-lick window
        self.reward = np.zeros((session_length, 1))                     # allocate reward vector (session length)
        self.stim1 = np.zeros((session_length, 1))
        self.stim2 = np.zeros((session_length, 1))


@dataclass
class WDTSesssionParams:
    no_lick_wind: tuple = (3, 3)                                    # (min, max) seconds: no-lick window before starting a trial
    iti: tuple = (6, 12)                                            # inter-trial interval range (seconds), sampled uniformly
    response_wind: float = 1.0                                      # response window duration (seconds) after stimulus onset
    reward_prob: float = 1.0                                        # probability of reward if lick occurs within response window
    reward_size: float = 1.0                                        # reward magnitude
    trial_kinds: list = field(default_factory=lambda: [0, 1, 2])    # 0=catch, 1=whisker, 2=auditif — tires ~1/3 chacun
    stim1_amp: float = 1.0                                          # amplitude du whisker quand présent
    stim2_amp: float = 1.0                                          # amplitude de l'auditif quand présent
    reward_stim: int = 1                                            # QUEL stimulus est actuellement récompensé (1 ou 2)
                                                                     # — c'est ce paramètre que l'orchestration change
                                                                     # d'une session à l'autre pour créer le switch de
                                                                     # contingence WDT (stim1)->AUD (stim2).
    block_numb: int = 5                                             # number of blocks (used for block-rate plots)



@dataclass
class WDTSesssionState:
    last_lick_time: float = 0.0                         # timestamp (s) of last lick
    last_reward_time: float = 0.0                       # timestamp (s) of last reward
    last_stim_time: float = 0.0                         # timestamp (s) of last stimulus
    last_trial_time: float = 0.0                        # timestamp (s) of last trial start
    last_trial_kind: int = 0                            # 0/1/2 du dernier trial demarre (pour juger le reward)
    trial_times: list = field(default_factory=list)     # list of trial onset times (s)
    no_lick_wind: float = 0.0                           # current no-lick window (s)
    iti: float = 0.0                                    # current ITI (s), sampled from range
    reward_window: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))  # response window mask over time
    reward: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))         # reward time-course
    stim1: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))          # whisker amplitude trace
    stim2: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))          # auditory amplitude trace

    def initialize(self, session_length: int, no_lick_range: tuple, iti_range: tuple):
        self.no_lick_wind = sample_uniform_range(*no_lick_range)    # sample initial no-lick window
        self.iti = sample_uniform_range(*iti_range)                 # sample initial ITI
        self.reward_window = np.zeros((session_length, 1))          # allocate response-window mask (0/1)
        self.reward = np.zeros((session_length, 1))                 # allocate reward trace
        self.stim1 = np.zeros((session_length, 1))
        self.stim2 = np.zeros((session_length, 1))
        self.trial_times = []                                       # reset trial onset log
        self.last_lick_time = 0.0
        self.last_reward_time = 0.0
        self.last_stim_time = 0.0
        self.last_trial_time = 0.0
        self.last_trial_kind = 0



@dataclass
class MouseSessionState:
    expectation: np.ndarray = field(default_factory=lambda: np.array([]))  # reward expectation (0–1) per time bin,
                                                                            # PARTAGEE entre les deux stimuli (comme en mono)
    motivation: np.ndarray = field(default_factory=lambda: np.array([]))   # current motivation (scales p_lick)
    rpe: np.ndarray = field(default_factory=lambda: np.array([]))          # reward prediction error (reward - expectation)
    p_lick: np.ndarray = field(default_factory=lambda: np.array([]))       # lick drive / probability per bin
    lick: np.ndarray = field(default_factory=lambda: np.array([]))         # 0/1 lick emitted at this bin
    eligibility1: np.ndarray = field(default_factory=lambda: np.array([])) # eligibility trace du whisker
    eligibility2: np.ndarray = field(default_factory=lambda: np.array([])) # eligibility trace de l'auditif
    uncertainty: np.ndarray = field(default_factory=lambda: np.array([]))  # U: confidence in the internal model, [0, U_max]
    expectation_nogo: np.ndarray = field(default_factory=lambda: np.array([]))  # E_nogo général (Go/No-Go)
    non_rew_lick_cnt: int = 0

    def initialize(self, session_length: int, init_motivation: float, init_expectation: float = 0.0,
                   init_uncertainty: float = 0.0, init_expectation_nogo: float = 0.0):
        self.expectation = np.full((session_length, 1), float(init_expectation))    # set initial expectation constant
        self.motivation = np.zeros((session_length, 1))                             # allocate motivation curve
        self.motivation[0, 0] = init_motivation                                     # set motivation at t=0
        self.rpe = np.zeros((session_length, 1))                                    # allocate RPE curve
        self.p_lick = np.zeros((session_length, 1))                                 # allocate p(lick) curve
        self.lick = np.zeros((session_length, 1))                                   # allocate lick (0/1) curve
        self.eligibility1 = np.zeros((session_length, 1))                           # allocate eligibility traces
        self.eligibility2 = np.zeros((session_length, 1))
        # U est un "state" (comme Expectation) : pas de forgetting entre sessions, on repart
        # du dernier niveau de confiance atteint (sauf pour la toute 1ere session, U=0).
        self.uncertainty = np.full((session_length, 1), float(init_uncertainty))
        # E_nogo persiste aussi entre sessions (meme logique que E et U).
        self.expectation_nogo = np.full((session_length, 1), float(init_expectation_nogo))
        self.non_rew_lick_cnt = 0                                                   # reset non-reward lick counter



# Regroupe les constantes qui etaient des globales de module dans main.py, pour que les
# fonctions d'orchestration (initialization/run_FL1/run_FL2/run_all_wdt/plot_all_results/
# save_run_parameters, dans functions.py) les recoivent en un seul argument plutot qu'une
# longue liste de parametres — meme pattern que Code_mono_stim_v2/models.py::SimConfig.
@dataclass
class SimConfig:
    # switches plots
    PLOT_TRACES: bool = True
    PLOT_BLOCK_HRFA: bool = True
    MAX_TRIALS_BLOCKS: int = 400
    PLOT_SESSIONS_COMPARISON: bool = True
    PLOT_SINGLE_SESSION_ALL: bool = True
    PLOT_STIM_GAINS: bool = True            # evolution des gains stim1/stim2 (+ nogo si actif) — LE plot d'extinction
    PLOT_RPE_ALL: bool = True
    PLOT_ABS_RPE_ALL: bool = True
    SAVE_PARAMETERS_TXT: bool = True

    # reglages generaux — switch de contingence WDT (stim1) -> AUD (stim2)
    NUM_WDT: int = 10                       # nombre total de sessions de detection
    SWITCH_SESSION: int = 6                 # sessions 1..SWITCH_SESSION-1: stim1 recompense ; SWITCH_SESSION..NUM_WDT: stim2
    TRIAL_KINDS: list = field(default_factory=lambda: [0, 1, 2])  # 0=catch, 1=whisker, 2=auditif
    STIM1_AMP: float = 1.0
    STIM2_AMP: float = 1.0

    # decision gate V*M - C (statique ici ; pas de VC_VARY dans cette version, cf. mono_stim_v2)
    REWARD_VALUE: float = 1.0
    COST: float = 0.25

    # seuils de lechage, separes Free Licking / WDT-AUD
    LICK_THRS_FL: float = 0.3
    LICK_THRS_WDT: float = 0.40

    # desapprentissage general (architecture Go/No-Go) — voir Mouse.delearning_enable.
    DELEARNING: bool = False
    NOGO_TAU: float = 8.0
    NOGO_GAIN: float = 0.15
    NOGO_RELIEF: float = 0.4

    # apprentissage des gains de stimulus (Go et No-Go specifique au stimulus) — voir les notes
    # sur Mouse.learning_stim / Mouse.learning_stim_nogo pour la justification de ces valeurs.
    LEARNING_STIM: float = 0.1
    LEARNING_STIM_NOGO: float = 0.02


@dataclass
class FLSessionResult:
    """Resultat d'une session Free Licking : etat de session + etat de la souris."""
    session: Any = None          # FreeLickingSessionState
    mouse_session: Any = None    # MouseSessionState


@dataclass
class WDTRunResult:
    """Resultat d'une session WDT/AUD (utilisee aussi bien pour WDT1..N que pour WDT_TEST)."""
    label: str = ""
    session: Any = None          # WDTSesssionState
    mouse_session: Any = None    # MouseSessionState
    perf: Any = None             # np.ndarray (function_performance_wdt)
    stim1_gain: float = 0.0      # gain net (go) de stim1 a la fin de cette session — pour le plot d'evolution
    stim2_gain: float = 0.0
    stim1_nogo_gain: float = 0.0
    stim2_nogo_gain: float = 0.0


@dataclass
class WDTBundle:
    """Accumulation des N sessions WDT/AUD, dans l'ordre, plus le dernier etat pour chainer vers WDT_TEST."""
    results: List[WDTRunResult] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)

    @property
    def last_mouse_session(self):
        return self.results[-1].mouse_session

    @property
    def last_reward(self):
        return self.results[-1].session.reward
