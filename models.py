import numpy as np
import random
from dataclasses import dataclass, field
from typing import Any, List, Optional



# Tire un nombre uniforme dans [min_val, max_val). Redefinie ici (au lieu d'un import depuis
# functions.py) pour eviter une dependance circulaire : functions.py importe deja models.py.
def sample_uniform_range(min_val, max_val):
    return min_val + (max_val - min_val) * random.random()



@dataclass
class SessionBaseInfo:
    """Discretisation temporelle commune a toutes les sessions (FL, WDT, WDT_TEST, whisker/auditif).
    Toute la simulation tourne bin par bin sur time_vector ; resolution=0.1s donne 36000 bins
    pour une session de 60 minutes."""
    duration: float = 60                                # duree d'une session, en minutes
    resolution: float = 0.1                             # taille d'un bin de temps, en secondes
    number_bin: int = field(init=False)                 # nombre de bins, calcule dans __post_init__
    time_vector: np.ndarray = field(init=False)         # vecteur des temps (s), calcule dans __post_init__

    def __post_init__(self):
        self.time_vector = np.arange(0.0, self.duration * 60, self.resolution)
        self.number_bin = len(self.time_vector)




@dataclass
class Mouse:
    """Parametres d'une souris (constants ou appris) et reglages de la decision."""
    noise: tuple = (1.2, 0.4)                   # bruit Gamma Nd : (forme, echelle)
    lick_thrs: float = 1.0                      # seuil de decision en Free Licking
    lick_thrs_wdt: float = 1.0                  # seuil de decision en WDT
    motivation: tuple = (1.0, 0.003)            # (valeur initiale, perte par reward) de M
    exp_update_reward: tuple = (2000, 0.3)      # (tau, gain) de E_go a la recompense
    exp_update_no_reward: tuple = (4, 0.4)      # (tau, gain) de E_go au lick non recompense
    exp_update_stim: tuple = (1, 0.1)           # (tau, gain) de l'anticipation stimulus (mono)
    learning_nonrew_lick: tuple = (2, 1)        # (seuil de licks non recompenses, increment de tau sur exp_update_no_reward)
    learning_stim: float = 0.004                # taux d'apprentissage du gain de stimulus
    tau_eligibility: float = 2.0                # constante de temps de la trace d'eligibilite

    reward_value: float = 1.0                   # V du gate V*M - C
    cost: float = 0.0                           # C du gate V*M - C

    # D = [E/(1+e^{A.(U-U0)}) + Nd/(1+e^{-A.(U-U0)})] . [V.M - C]
    decision_slope: float = 5.0                 # A pour Free Licking
    decision_slope_wdt: float = 3.0             # A pour WDT (transition plus douce)
    uncertainty_offset: float = 0.5             # U0, point neutre de la sigmoide
    uncertainty_gain: float = 0.1               # facteur d'echelle de la mise a jour de U
    uncertainty_max: float = 1.0                # plafond de U

    # Voie No-Go : E_nogo s'oppose a E_go dans la decision (E_go - E_nogo), inhibition plus
    # lente et plus persistante. Recurrence a chaque bin (comme motivation/uncertainty), pas
    # injection de noyau futur.
    exp_update_nogo: tuple = (8.0, 0.15)        # (tau, gain) de E_nogo, mono
    exp_update_nogo_right: tuple = (8.0, 0.15)  # idem, cote droit (dual)
    exp_update_nogo_left: tuple = (8.0, 0.15)   # idem, cote gauche (dual)
    nogo_relief: float = 0.4                    # fraction de E_nogo effacee a chaque reward
    nogo_streak_incr: tuple = (2, 0.0)          # (seuil de bouts non recompenses consecutifs, increment de tau_nogo)
    nogo_tau_max: float = 8.0                   # plafond de tau_nogo
    learning_nogo_gain: float = 0.0             # taux d'apprentissage du gain No-Go (0 = desactive)
    nogo_gain_max: float = 0.6                  # plafond du gain No-Go appris

    # Choix du cote (dual) : bruit Gumbel independant sur E_droite/E_gauche puis argmax
    # (Gumbel-max trick), equivalent a une sigmoide sur leur difference.
    side_readout_slope: float = 5.0

    # Dual : deux Expectations independantes, gain de stimulus separe par cote.
    stim_gain_right: float = 0.1
    stim_gain_left: float = 0.1
    stim_gain_max: float = 0.6                  # plafond des deux gains ci-dessus

    # Baisse symetrique de stim_gain_right/left quand un lick correct n'est pas recompense a
    # cause du tirage de reward_prob (jamais d'un mismatch ou du cooldown seul) — inerte hors
    # desapprentissage puisque reward_prob=1.0 par defaut.
    stim_gain_noreward_active: bool = False

    # Dual : generalisation sensorielle precoce, un stimulus pousse aussi un peu le canal
    # oppose. Diminue tout seul avec l'entrainement puisque stim_gain_right/left grandissent
    # alors que ce terme reste fixe.
    cross_stim_gain: float = 0.05

    # Whisker/auditif : une seule Expectation partagee (comme le mono), alimentee par deux
    # canaux (whisker=1, auditif=2) avec chacun son propre gain. Le gain d'un stimulus
    # redescend directement (meme regle que sa montee) quand un lick dessus n'est pas
    # recompense parce que ce n'est plus le stimulus actuellement recompense — equivalent
    # whisker/auditif de stim_gain_noreward_active, mais toujours actif ici. Champs _wa,
    # jamais lus par le mono ou le dual gauche/droite.
    exp_update_stim1_wa: tuple = (1, 0.1)       # (tau, gain) whisker
    exp_update_stim2_wa: tuple = (1, 0.1)       # (tau, gain) auditif
    stim_gain_max_wa: float = 8.0               # plafond des deux gains ci-dessus
    learning_stim_wa: float = 0.1               # taux d'apprentissage (memes deux sens)

    # Non utilises actuellement (voir README) — gardes pour ne pas casser une reference externe.
    noise_adapt_enable: bool = True
    noise_target_rpe: float = 0.20
    noise_window_bins: int = 5
    noise_update_period_bins: int = 5
    noise_alpha: float = 0.3
    noise_gain_bounds: tuple = (0.01, 0.20)



@dataclass
class FreeLickingSessionParams:
    """Parametres d'une session Free Licking (FL1/FL2)."""
    no_lick_wind: tuple = (3, 3)                # (min, max) s : immobilite requise avant qu'un lick redevienne eligible
    response_wind: tuple = (1, 1)               # non utilise en FL, garde pour la meme structure que WDTSesssionParams
    reward_prob: float = 1.0                    # probabilite de recompense
    reward_size: float = 1.0                    # magnitude de la recompense
    forced_reward: tuple = (1, 500)             # (actif 0/1, temps en s) : recompense forcee pour amorcer lick->reward



@dataclass
class WDTSesssionParams:
    """Parametres d'une session WDT (mono-stimulus)."""
    no_lick_wind: tuple = (3, 3)                                    # (min, max) s : immobilite requise avant un nouvel essai
    iti: tuple = (6, 12)                                            # intervalle inter-essai (s)
    response_wind: float = 1.0                                      # duree (s) de la fenetre de reponse apres le stimulus
    reward_prob: float = 1.0                                        # probabilite de recompense
    reward_size: float = 1.0                                        # magnitude de la recompense
    trial_types: list = field(default_factory=lambda: [0, 1.0])     # amplitudes de stimulus possibles (0 = catch)
    block_numb: int = 5                                             # nombre de blocs pour les plots par bloc

    # Dual whisker/auditif uniquement (function_wdt_session_wa), jamais lus par le mono.
    trial_kinds: list = field(default_factory=lambda: [0, 1, 2])    # 0=catch, 1=whisker, 2=auditif
    stim1_amp: float = 1.0                                          # amplitude du whisker quand present
    stim2_amp: float = 1.0                                          # amplitude de l'auditif quand present
    reward_stim: int = 1                                            # quel stimulus est actuellement recompense (1 ou 2)



@dataclass
class FreeLickingSessionState:
    """Etat mutable d'une session FL en cours, rempli bin par bin."""
    last_lick_time: float = 0.0                                             # temps (s) du dernier lick
    last_reward_time: float = 0.0                                           # temps (s) de la derniere recompense
    no_lick_wind: float = 0.0                                               # duree d'immobilite requise en vigueur (s)
    reward: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))    # recompense a chaque bin
    stim1: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))     # trace de stimulus (toujours nulle en FL)


    def initialize(self, session_length: int, no_lick_range: tuple):
        self.last_lick_time = 0.0
        self.last_reward_time = 0.0
        self.no_lick_wind = sample_uniform_range(*no_lick_range)
        self.reward = np.zeros((session_length, 1))
        self.stim1 = np.zeros((session_length, 1))


@dataclass
class WDTSesssionState:
    """Etat mutable d'une session WDT en cours."""
    last_lick_time: float = 0.0                         # temps (s) du dernier lick
    last_reward_time: float = 0.0                       # temps (s) de la derniere recompense
    last_stim_time: float = 0.0                         # temps (s) du dernier stimulus presente
    last_trial_time: float = 0.0                        # temps (s) du debut du dernier essai
    trial_times: list = field(default_factory=list)     # historique des temps de debut d'essai
    no_lick_wind: float = 0.0                           # duree d'immobilite requise en vigueur (s)
    iti: float = 0.0                                    # ITI en vigueur (s)
    reward_window: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))  # masque 0/1 : fenetre de reponse
    reward: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))         # recompense a chaque bin
    stim1: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))          # amplitude du stimulus (mono) / whisker (whisker/auditif)
    stim2: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))          # amplitude auditif (whisker/auditif uniquement)
    last_trial_kind: int = 0                                                     # type du dernier essai demarre (0/1/2)

    def initialize(self, session_length: int, no_lick_range: tuple, iti_range: tuple):
        self.no_lick_wind = sample_uniform_range(*no_lick_range)
        self.iti = sample_uniform_range(*iti_range)
        self.reward_window = np.zeros((session_length, 1))
        self.reward = np.zeros((session_length, 1))
        self.stim1 = np.zeros((session_length, 1))
        self.stim2 = np.zeros((session_length, 1))
        self.last_trial_kind = 0
        self.trial_times = []



@dataclass
class MouseSessionState:
    """Etat mutable de la souris (mono) au fil d'une session, herite de la session precedente."""
    expectation: np.ndarray = field(default_factory=lambda: np.array([]))  # E_go, dans [0, 1]
    motivation: np.ndarray = field(default_factory=lambda: np.array([]))   # M
    rpe: np.ndarray = field(default_factory=lambda: np.array([]))          # reward - E_go(t-1)
    p_lick: np.ndarray = field(default_factory=lambda: np.array([]))       # valeur de la decision d_t
    lick: np.ndarray = field(default_factory=lambda: np.array([]))         # 0/1 : lick emis a ce bin
    eligibility: np.ndarray = field(default_factory=lambda: np.array([]))  # trace d'eligibilite au stimulus
    uncertainty: np.ndarray = field(default_factory=lambda: np.array([]))  # U, dans [0, uncertainty_max]
    expectation_nogo: np.ndarray = field(default_factory=lambda: np.array([]))  # E_nogo
    # Dual whisker/auditif uniquement, jamais lues par le mono.
    eligibility1: np.ndarray = field(default_factory=lambda: np.array([]))  # eligibilite whisker
    eligibility2: np.ndarray = field(default_factory=lambda: np.array([]))  # eligibilite auditif
    non_rew_lick_cnt: int = 0  # compte chaque bin d'un lick non recompense (voir Mouse.learning_nonrew_lick)
    nogo_streak_cnt: int = 0   # compte les bouts non recompenses consecutifs DANS un essai (sensibilisation No-Go)

    def initialize(self, session_length: int, init_motivation: float, init_expectation: float = 0.0,
                   init_uncertainty: float = 0.0, init_expectation_nogo: float = 0.0):
        self.expectation = np.full((session_length, 1), float(init_expectation))
        self.motivation = np.zeros((session_length, 1))
        self.motivation[0, 0] = init_motivation
        self.rpe = np.zeros((session_length, 1))
        self.p_lick = np.zeros((session_length, 1))
        self.lick = np.zeros((session_length, 1))
        self.eligibility = np.zeros((session_length, 1))
        self.eligibility1 = np.zeros((session_length, 1))
        self.eligibility2 = np.zeros((session_length, 1))
        self.uncertainty = np.full((session_length, 1), float(init_uncertainty))
        self.expectation_nogo = np.full((session_length, 1), float(init_expectation_nogo))
        self.non_rew_lick_cnt = 0
        self.nogo_streak_cnt = 0



# Regroupe les constantes qui etaient des globales de module dans main.py, pour que les
# fonctions d'orchestration (initialization/run_FL1/run_FL2/run_all_wdt/plot_all_results/
# save_run_parameters, dans functions.py) les recoivent en un seul argument plutot qu'une
# longue liste de parametres. Valeurs par defaut = valeurs qui etaient dans main.py.
@dataclass
class SimConfig:
    """Tous les reglages d'un run — plots, mode, desapprentissage, population."""

    # Switches plots (dossier de sortie entre parentheses) : pas d'effet sur la simulation.
    PLOT_TRACES: bool = True                    # traces detaillees par session (traces/)
    PLOT_BLOCK_HRFA: bool = True                 # HR/FA par bloc d'essais (block_rates/)
    MAX_TRIALS_BLOCKS: int = 400                 # taille des blocs pour PLOT_BLOCK_HRFA
    PLOT_SESSIONS_COMPARISON: bool = True        # courbe d'apprentissage WDT1..N (learning_curve/)
    PLOT_SINGLE_SESSION_ALL: bool = True         # resume par session (session_summary/)
    PLOT_STOCHASTIC_IN_POPULATION: bool = True   # mini-cohorte annexe, mono (voir N_MICE plus bas)
    PLOT_RPE_ALL: bool = True                    # RPE par lick (rpe/)
    PLOT_ABS_RPE_ALL: bool = True                # |RPE| par lick (rpe/)
    PLOT_PSYCHO_TEST: bool = True                # courbe psychometrique sur WDT_TEST (psychometric/)
    SAVE_PARAMETERS_TXT: bool = True             # ecrit parameters.txt dans le dossier du run

    NUM_WDT: int = 10                            # nombre de sessions WDT jouees
    DEFAULT_TYPES: list = field(default_factory=lambda: [0.0, 1.0])   # types d'essai (0=catch, 1.0=stimulus)
    WDT_TEST_TYPES: list = field(default_factory=lambda: [0, 0, 0, 0, 0, 0.2, 0.5, 0.7, 1.0, 1.5])  # amplitudes de WDT_TEST

    # Modulation du bruit par sigmoide sur le RPE moyen — desactivee, c'est Uncertainty qui pilote l'exploration
    USE_SIGMOID_NOISE: bool = False
    SIG_GAIN_MIN: float = 0.08
    SIG_GAIN_MAX: float = 0.28
    SIG_X0: float = 0.4                          # point d'inflexion de la sigmoide
    SIG_SLOPE: float = 14.0
    SIG_AVG_LICKS: int = 15                      # fenetre de moyenne du RPE
    SIG_UPDATE_EVERY: int = 1
    PLOT_SIGMOID_MAPPING: bool = False

    # Mini-cohorte annexe (mono), voir PLOT_STOCHASTIC_IN_POPULATION / population_hr_fa_plot
    N_MICE: int = 10
    NOISE_GAIN: float = 0.06
    LEARNING_STIM: float = 0.02
    SESSION_NAME: str = "WDT10"                  # session comparee entre ces souris

    # Gate de decision V*M - C
    REWARD_VALUE: float = 1.0
    COST: float = 0.25
    VC_VARY: bool = False                        # V/C tires par essai au lieu de fixes (sample_trial_value_cost)
    V_RANGE: tuple = (0.5, 1.0)
    C_RANGE: tuple = (0.0, 0.5)
    VC_BUFFER_SIZE: int = 5                      # fenetre de la moyenne glissante V/C

    LICK_THRS_FL: float = 0.3
    LICK_THRS_WDT: float = 0.40

    # Mode dual gauche/droite (voir les classes Dual* plus bas)
    dual_stim: bool = False
    FL_FORCED_REWARD_RIGHT_T: float = 500.0      # temps (s) de la recompense forcee a droite en FL1
    FL_FORCED_REWARD_LEFT_T: float = 700.0       # temps (s) de la recompense forcee a gauche en FL1
    V_RIGHT: float = 1.0                         # V/C par cote, equivalent de REWARD_VALUE/COST
    V_LEFT: float = 1.0
    C_RIGHT: float = 0.0
    C_LEFT: float = 0.0
    DUAL_TRIAL_KINDS: list = field(default_factory=lambda: [0, 1, 2])  # 0=catch, 1=stim droite, 2=stim gauche
    DUAL_STIM_AMP: float = 1.0

    # Troisieme mode, independant de dual_stim : whisker/auditif, une seule Expectation
    # partagee et un switch de contingence WDT->AUD. Prioritaire sur dual_stim (voir main.py).
    WHISKER_AUD_STIM: bool = False
    WA_NUM_SESSIONS: int = 10
    WA_SWITCH_SESSION: int = 6           # avant : stim1 (whisker) recompense ; a partir de la : stim2 (auditif)
    WA_TRIAL_KINDS: list = field(default_factory=lambda: [0, 1, 2])  # 0=catch, 1=whisker, 2=auditif
    WA_STIM1_AMP: float = 1.0
    WA_STIM2_AMP: float = 1.0

    # Voie No-Go (toujours active dans la decision, voir Mouse.exp_update_nogo)
    NOGO_TAU: float = 8.0
    NOGO_GAIN: float = 0.15
    NOGO_RELIEF: float = 0.4
    NOGO_STREAK_THRESHOLD: int = 4      # bouts non recompenses consecutifs DANS un essai avant sensibilisation
    NOGO_TAU_GROWTH: float = 1.0        # increment de tau_nogo a chaque seuil atteint (0 = desactive)
    NOGO_TAU_MAX: float = 60.0
    LEARNING_NOGO_GAIN: float = 0.05    # taux d'apprentissage du gain No-Go
    NOGO_GAIN_MAX: float = 0.6

    STIM_GAIN_NOREWARD_ACTIVE: bool = True   # voir Mouse.stim_gain_noreward_active

    # Desapprentissage (mono/dual, integre au pipeline normal, pas un scenario a part)
    DELEARNING_FROM_SESSION: Optional[int] = None    # coupe la recompense a partir de WDT{N} (None = pas d'effet)
    DELEARNING_SIDE: Optional[int] = None            # dual : cote cible (+1 droite, -1 gauche, None = les deux)
    DELEARNING_UNTIL_SESSION: Optional[int] = None   # reapprentissage : recompense restauree a partir de WDT{N}

    # Population (mono/dual) : remplace le run normal, voir run_population
    POPULATION_RANGE: bool = False
    POPULATION_N_MICE: int = 10
    POPULATION_PARAM_SPREAD: float = 0.10            # spread +/- de learning_stim et noise[1] entre souris
    POPULATION_SPREAD_SWEEP: bool = False            # relance pour chaque pourcentage de POPULATION_SWEEP_VALUES
    POPULATION_SWEEP_VALUES: tuple = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)


@dataclass
class FLSessionResult:
    """Resultat d'une session Free Licking : etat de session + etat de la souris."""
    session: Any = None          # FreeLickingSessionState
    mouse_session: Any = None    # MouseSessionState


@dataclass
class WDTRunResult:
    """Resultat d'une session WDT (utilisee aussi bien pour WDT1..N que pour WDT_TEST)."""
    label: str = ""
    session: Any = None          # WDTSesssionState
    mouse_session: Any = None    # MouseSessionState
    perf: Any = None             # np.ndarray (function_performance_wdt)
    vc_log: dict = field(default_factory=lambda: {
        "trial_times": [], "v": [], "c": [], "expected_v": [], "expected_c": []
    })
    # --- Dual whisker/auditif uniquement : gains a la fin de cette session, pour le plot
    # d'evolution (plot_stim_gains). Restent a 0.0 pour le mono et le dual gauche/droite.
    stim1_gain: float = 0.0
    stim2_gain: float = 0.0


@dataclass
class WDTBundle:
    """Accumulation des N sessions WDT1..N, dans l'ordre, plus le dernier etat pour chainer vers WDT_TEST."""
    results: List[WDTRunResult] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)

    @property
    def last_mouse_session(self):
        return self.results[-1].mouse_session

    @property
    def last_reward(self):
        return self.results[-1].session.reward


# Deux stimuli, chacun menant a une recompense a un endroit different (droite/gauche).
# Classes separees des versions a un seul stimulus ci-dessus, rien n'y est modifie.
# Convention de signe utilisee partout dans le mode dual : +1 = droite, -1 = gauche, 0 = ni l'un ni l'autre.

@dataclass
class DualFreeLickingSessionParams:
    """Equivalent dual de FreeLickingSessionParams, une recompense forcee par cote."""
    no_lick_wind: tuple = (3, 3)
    response_wind: tuple = (1, 1)
    reward_prob: float = 1.0
    reward_size: float = 1.0
    forced_reward_right: tuple = (1, 500.0)   # (on/off, temps en s) recompense forcee a droite
    forced_reward_left: tuple = (1, 700.0)    # (on/off, temps en s) recompense forcee a gauche


@dataclass
class DualWDTSessionParams:
    """Equivalent dual de WDTSesssionParams."""
    no_lick_wind: tuple = (3, 3)
    iti: tuple = (6, 12)
    response_wind: float = 1.0
    reward_prob: float = 1.0
    reward_prob_right: Optional[float] = None    # surcharge par cote (None = utilise reward_prob), desapprentissage cible
    reward_prob_left: Optional[float] = None
    reward_size: float = 1.0
    trial_kinds: list = field(default_factory=lambda: [0, 1, 2])   # 0=catch, 1=stim droite, 2=stim gauche
    stim_amp: float = 1.0
    block_numb: int = 5


@dataclass
class DualFreeLickingSessionState:
    """Equivalent dual de FreeLickingSessionState."""
    last_lick_time: float = 0.0
    last_reward_time: float = 0.0
    no_lick_wind: float = 0.0
    reward: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))        # magnitude de la recompense
    reward_side: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))   # cote recompense (+1/-1/0)
    stim1: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))         # pas de stimulus en FL, garde pour compat

    def initialize(self, session_length: int, no_lick_range: tuple):
        self.last_lick_time = 0.0
        self.last_reward_time = 0.0
        self.no_lick_wind = sample_uniform_range(*no_lick_range)
        self.reward = np.zeros((session_length, 1))
        self.reward_side = np.zeros((session_length, 1))
        self.stim1 = np.zeros((session_length, 1))


@dataclass
class DualWDTSessionState:
    """Equivalent dual de WDTSesssionState (un seul stim_signed, signe = cote)."""
    last_lick_time: float = 0.0
    last_reward_time: float = 0.0
    last_stim_time: float = 0.0
    last_trial_time: float = 0.0
    trial_times: list = field(default_factory=list)
    trial_correct_side: list = field(default_factory=list)   # cote correct par trial (+1/-1/0), meme ordre que trial_times
    last_trial_kind: int = 0
    last_trial_correct_side: int = 0
    no_lick_wind: float = 0.0
    iti: float = 0.0
    reward_window: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))
    reward: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))
    reward_side: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))
    stim_signed: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))    # +amp droite, -amp gauche, 0 sinon
    stim1: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))          # alias |stim_signed|, garde pour compat plots

    def initialize(self, session_length: int, no_lick_range: tuple, iti_range: tuple):
        self.no_lick_wind = sample_uniform_range(*no_lick_range)
        self.iti = sample_uniform_range(*iti_range)
        self.reward_window = np.zeros((session_length, 1))
        self.reward = np.zeros((session_length, 1))
        self.reward_side = np.zeros((session_length, 1))
        self.stim_signed = np.zeros((session_length, 1))
        self.stim1 = np.zeros((session_length, 1))
        self.trial_times = []
        self.trial_correct_side = []
        self.last_lick_time = 0.0
        self.last_reward_time = 0.0
        self.last_stim_time = 0.0
        self.last_trial_time = 0.0
        self.last_trial_kind = 0
        self.last_trial_correct_side = 0


@dataclass
class DualMouseSessionState:
    """Equivalent dual de MouseSessionState : deux Expectations independantes."""
    expectation_right: np.ndarray = field(default_factory=lambda: np.array([]))
    expectation_left: np.ndarray = field(default_factory=lambda: np.array([]))
    motivation: np.ndarray = field(default_factory=lambda: np.array([]))
    rpe: np.ndarray = field(default_factory=lambda: np.array([]))
    p_lick: np.ndarray = field(default_factory=lambda: np.array([]))       # D_final signee : >0 valeur cote droite, <0 valeur cote gauche
    lick: np.ndarray = field(default_factory=lambda: np.array([]))         # 0/1, un lick a eu lieu (peu importe le cote)
    lick_side: np.ndarray = field(default_factory=lambda: np.array([]))    # +1/-1/0 : quel cote a ete leche
    eligibility_right: np.ndarray = field(default_factory=lambda: np.array([]))
    eligibility_left: np.ndarray = field(default_factory=lambda: np.array([]))
    uncertainty: np.ndarray = field(default_factory=lambda: np.array([]))
    # Voie No-Go independante par cote — memes regles que la version mono (expectation_nogo),
    # appliquee separement a droite et a gauche pour qu'un cote qui cesse d'etre recompense
    # s'eteigne sans affecter l'autre.
    expectation_nogo_right: np.ndarray = field(default_factory=lambda: np.array([]))
    expectation_nogo_left: np.ndarray = field(default_factory=lambda: np.array([]))
    non_rew_lick_cnt: int = 0
    # Compteurs de sensibilisation separes par cote (sinon l'extinction d'un cote ferait
    # grandir le tau No-Go de l'autre cote encore recompense).
    nogo_streak_cnt_right: int = 0
    nogo_streak_cnt_left: int = 0

    def initialize(self, session_length: int, init_motivation: float,
                   init_expectation_right: float = 0.0, init_expectation_left: float = 0.0,
                   init_uncertainty: float = 0.0,
                   init_expectation_nogo_right: float = 0.0, init_expectation_nogo_left: float = 0.0):
        self.expectation_right = np.full((session_length, 1), float(init_expectation_right))
        self.expectation_left = np.full((session_length, 1), float(init_expectation_left))
        self.motivation = np.zeros((session_length, 1))
        self.motivation[0, 0] = init_motivation
        self.rpe = np.zeros((session_length, 1))
        self.p_lick = np.zeros((session_length, 1))
        self.lick = np.zeros((session_length, 1))
        self.lick_side = np.zeros((session_length, 1))
        self.eligibility_right = np.zeros((session_length, 1))
        self.eligibility_left = np.zeros((session_length, 1))
        self.uncertainty = np.full((session_length, 1), float(init_uncertainty))
        self.expectation_nogo_right = np.full((session_length, 1), float(init_expectation_nogo_right))
        self.expectation_nogo_left = np.full((session_length, 1), float(init_expectation_nogo_left))
        self.non_rew_lick_cnt = 0
        self.nogo_streak_cnt_right = 0
        self.nogo_streak_cnt_left = 0