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
    exp_update_stim: tuple = (1, 0.1)           # (tau, gain) for update after a stimulus
    learning_nonrew_lick: tuple = (2, 1)        # (threshold of non-reward licks, tau increment)
    learning_stim: float = 0.004                # learning rate for stimulus gain (scaled by RPE * eligibility)
    tau_eligibility: float = 2.0                # eligibility trace time constant (seconds)

    reward_value: float = 1.0                   # V: value of the reward, gates the decision (V*M - C)
    cost: float = 0.0                           # C: cost of licking, gates the decision (V*M - C)

    # Decision function D = [E/(1+e^{A_D.(U-U0)}) + Nd/(1+e^{-A_D.(U-U0)})] . [V.M - C]
    # A_D separe par contexte (comme lick_thrs/lick_thrs_wdt): en Free Licking on veut que E
    # domine fort des qu'un signal net existe (transition raide) ; en WDT on veut une transition
    # douce pour un apprentissage progressif sur plusieurs sessions plutot qu'un "tout ou rien".
    decision_slope: float = 5.0                 # "A_D" pour Free Licking: pente de transition exploit/explore
    decision_slope_wdt: float = 3.0             # "A_D" pour WDT: pente plus douce, apprentissage progressif
    uncertainty_offset: float = 0.5             # "U0": point neutre de la sigmoide — a U=U0, poids 50/50.
                                                 # A U=0 (souris naive/pas de surprise), E domine (w_exploit haut,
                                                 # w_explore quasi nul) au lieu d'un 50/50 qui laisserait le bruit
                                                 # seul faire licker meme sans aucune expectation.
    uncertainty_gain: float = 0.1               # "A_U": facteur d'echelle de la mise a jour de U, dans [0,1]
    uncertainty_max: float = 1.0                # U_max: borne haute de U (U reste dans [0, U_max])

    # Architecture Go/No-Go (D1R/D2R), fait partie de la decision de base — pas optionnelle.
    # En plus de l'Expectation habituelle (E_go, dynamique rapide : monte a la recompense,
    # redescend a chaque lick non recompense via exp_update_no_reward), une voie separee
    # E_nogo accumule un signal d'inhibition plus lent et plus persistant. La decision
    # utilise E_go - E_nogo (function_mouse_lick / function_mouse_lick_dual) au lieu de E_go
    # seule. Mise a jour comme motivation/uncertainty (recurrence a chaque bin, pas injection
    # d'un noyau futur) : ce paradigme genere des dizaines de bouts de lechage non recompenses
    # par seconde une fois la souris confiante, une injection future ferait saturer E_nogo en
    # quelques secondes. Le kick est saturant (gain*(1-E_nogo), pas +gain) donc borne par
    # construction.
    exp_update_nogo: tuple = (8.0, 0.15)        # (tau, gain) pour la voie No-Go (modele mono-stimulus)
    # Dual stim : tau/gain separes par cote, pour que la sensibilisation (croissance du tau,
    # voir nogo_streak_incr) declenchee par l'extinction d'un cote ne "fuite" jamais vers
    # l'autre cote encore recompense. Initialises aux memes valeurs que exp_update_nogo.
    exp_update_nogo_right: tuple = (8.0, 0.15)
    exp_update_nogo_left: tuple = (8.0, 0.15)
    nogo_relief: float = 0.4                    # fraction de E_nogo effacee a chaque reward confirme
    nogo_streak_incr: tuple = (2, 0.0)          # (seuil de lechages non recompenses consecutifs, increment de tau)
    nogo_tau_max: float = 8.0                   # plafond du tau_nogo (= tau initial, pas de croissance par defaut)
                                                 # (reacquisition rapide apres extinction, cf. litterature)

    # Dual stim uniquement (voir function_mouse_lick_dual) : bruit Gumbel independant ajoute
    # a E_droite et E_gauche separement puis argmax (Gumbel-max trick, echelle 1/side_readout_slope) —
    # equivalent statistique exact d'une sigmoide sur (E_droite-E_gauche), mais le bruit vit sur
    # les deux Expectations plutot qu'au moment du choix (comme dans Lak et al. 2020). Donc meme
    # quand un cote est tres confiant, il reste une probabilite residuelle de lire l'autre.
    side_readout_slope: float = 5.0

    # Dual stim uniquement : deux Expectations independantes (droite/gauche), chacune se
    # comporte comme l'Expectation du modele mono-stimulus (meme mise a jour reward/no-reward,
    # meme bornes [0,1]). Le gain d'anticipation stimulus est aussi separe par cote (comme dans
    # Code_dual_stim_v2 : stim_gain1/stim_gain2), pour que l'apprentissage d'un cote n'influence
    # jamais l'autre. tau et exp_update_reward/exp_update_no_reward restent partages (mouse.exp_update_stim[0]
    # pour le tau, mouse.exp_update_reward/no_reward pour la vitesse de mise a jour a la reward/no-reward).
    stim_gain_right: float = 0.1
    stim_gain_left: float = 0.1
    stim_gain_max: float = 0.6                  # plafond de croissance des deux gains ci-dessus (evite qu'un
                                                 # seul stimulus finisse par saturer E instantanement, cf. grid search)

    # Dual stim uniquement : au debut de l'apprentissage, une souris naive ne discrimine pas
    # parfaitement les deux stimuli (generalisation sensorielle) — un stimulus pousse un peu
    # le canal oppose en plus du sien. Comme stim_gain_right/left grandissent avec l'experience
    # alors que ce terme reste fixe, son poids relatif — et donc la confusion — diminue tout
    # seul avec l'entrainement, sans mecanisme de decroissance separe. Valide par grid search
    # multi-seeds : ~0.05 donne un Mismatch WDT1 ~7% qui retombe a 0 vers WDT10, sans biais de cote.
    cross_stim_gain: float = 0.05

    # --- Dual whisker/auditif uniquement (voir function_update_mouse_state_wa) ---------------
    # Reprend le paradigme historique (switch de contingence WDT->AUD) mais avec UNE SEULE
    # Expectation partagee (comme le mono-stimulus), alimentee par deux canaux de stimulus
    # (whisker=1, auditif=2), chacun avec son propre gain Go et son propre gain No-Go appris.
    # Tous les champs ci-dessous sont suffixes _wa et n'entrent JAMAIS en jeu pour le mono ou
    # le dual gauche/droite — aucun risque de collision avec exp_update_stim/learning_stim
    # (mono) ou stim_gain_max (dual gauche/droite, valeur differente : 0.6 vs 8.0 ici).
    exp_update_stim1_wa: tuple = (1, 0.1)       # (tau, gain Go) whisker
    exp_update_stim2_wa: tuple = (1, 0.1)       # (tau, gain Go) auditif
    stim1_nogo_gain_wa: float = 0.0             # gain No-Go whisker (mutable, comme exp_update_stim1_wa[1])
    stim2_nogo_gain_wa: float = 0.0             # gain No-Go auditif
    stim_gain_max_wa: float = 8.0               # plafond commun aux 4 gains ci-dessus (go1/go2/nogo1/nogo2).
                                                 # Volontairement haut : un plafond bas fait saturer le gain Go
                                                 # des la 1ere session (recompensee), apres quoi il ne peut plus
                                                 # compenser l'accumulation lente du No-Go — le gain net declinerait
                                                 # des le debut au lieu de rester stable jusqu'au switch.
    learning_stim_wa: float = 0.1               # taux d'apprentissage du gain Go (whisker/auditif)
    learning_stim_nogo_wa: float = 0.02         # taux d'apprentissage du gain No-Go — volontairement 5x plus
                                                 # lent que le Go (sinon les deux saturent ensemble des la 2e
                                                 # session et le gain net s'effondre avant meme le switch).

    noise_adapt_enable: bool = True             # active/désactive la feature
    noise_target_rpe: float = 0.20              # |RPE| "attendu" (0..1)
    noise_window_bins: int = 5                  # moyenne sur les 5 derniers RPE
    noise_update_period_bins: int = 5           # met à jour tous les 5 bins
    noise_alpha: float = 0.3                    # pas d'adaptation (0.1..0.4 raisonnable)
    noise_gain_bounds: tuple = (0.01, 0.20)     # [gain_min, gain_max] pour rester stable



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

    # --- Dual whisker/auditif uniquement (function_wdt_session_wa) ---------------------------
    # Champs additifs, jamais lus par function_wdt_session (mono) ni par le dual gauche/droite.
    trial_kinds: list = field(default_factory=lambda: [0, 1, 2])    # 0=catch, 1=whisker, 2=auditif
    stim1_amp: float = 1.0                                          # amplitude du whisker quand present
    stim2_amp: float = 1.0                                          # amplitude de l'auditif quand present
    reward_stim: int = 1                                            # quel stimulus est actuellement recompense (1 ou 2)



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
    stim1: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))          # stimulus amplitude trace (mono) / whisker (dual whisker/auditif)
    # --- Dual whisker/auditif uniquement : jamais lu par function_wdt_session (mono) ---------
    stim2: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))          # auditory amplitude trace
    last_trial_kind: int = 0                                                     # 0/1/2 du dernier trial demarre

    def initialize(self, session_length: int, no_lick_range: tuple, iti_range: tuple):
        self.no_lick_wind = sample_uniform_range(*no_lick_range)    # sample initial no-lick window
        self.iti = sample_uniform_range(*iti_range)                 # sample initial ITI
        self.reward_window = np.zeros((session_length, 1))          # allocate response-window mask (0/1)
        self.reward = np.zeros((session_length, 1))                 # allocate reward trace
        self.stim1 = np.zeros((session_length, 1))                  # allocate stimulus trace
        self.stim2 = np.zeros((session_length, 1))                  # allocate auditory trace (dual whisker/auditif)
        self.last_trial_kind = 0
        self.trial_times = []                                       # reset trial onset log



@dataclass
class MouseSessionState:
    expectation: np.ndarray = field(default_factory=lambda: np.array([]))  # reward expectation (0–1) per time bin
    motivation: np.ndarray = field(default_factory=lambda: np.array([]))   # current motivation (scales p_lick)
    rpe: np.ndarray = field(default_factory=lambda: np.array([]))          # reward prediction error (reward - expectation)
    p_lick: np.ndarray = field(default_factory=lambda: np.array([]))       # lick drive / probability per bin
    lick: np.ndarray = field(default_factory=lambda: np.array([]))         # 0/1 lick emitted at this bin
    eligibility: np.ndarray = field(default_factory=lambda: np.array([]))  # eligibility trace (exponential decay)
    uncertainty: np.ndarray = field(default_factory=lambda: np.array([]))  # U: confidence in the internal model, [0, U_max]
    expectation_nogo: np.ndarray = field(default_factory=lambda: np.array([]))  # E_nogo: voie d'inhibition Go/No-Go
    # --- Dual whisker/auditif uniquement : jamais lues par le mono (une seule `eligibility`) -
    eligibility1: np.ndarray = field(default_factory=lambda: np.array([]))  # eligibility whisker
    eligibility2: np.ndarray = field(default_factory=lambda: np.array([]))  # eligibility auditif
    non_rew_lick_cnt: int = 0
    nogo_streak_cnt: int = 0  # lechages non recompenses consecutifs, pour la sensibilisation du tau No-Go

    def initialize(self, session_length: int, init_motivation: float, init_expectation: float = 0.0,
                   init_uncertainty: float = 0.0, init_expectation_nogo: float = 0.0):
        self.expectation = np.full((session_length, 1), float(init_expectation))    # set initial expectation constant
        self.motivation = np.zeros((session_length, 1))                             # allocate motivation curve
        self.motivation[0, 0] = init_motivation                                     # set motivation at t=0
        self.rpe = np.zeros((session_length, 1))                                    # allocate RPE curve
        self.p_lick = np.zeros((session_length, 1))                                 # allocate p(lick) curve
        self.lick = np.zeros((session_length, 1))                                   # allocate lick (0/1) curve
        self.eligibility = np.zeros((session_length, 1))                            # allocate eligibility trace
        self.eligibility1 = np.zeros((session_length, 1))                           # dual whisker/auditif seulement
        self.eligibility2 = np.zeros((session_length, 1))
        # U est un "state" (comme Expectation) : pas de forgetting entre sessions, on repart
        # du dernier niveau de confiance atteint (sauf pour la toute 1ere session, U=0).
        self.uncertainty = np.full((session_length, 1), float(init_uncertainty))
        # E_nogo persiste aussi entre sessions (meme logique que E et U).
        self.expectation_nogo = np.full((session_length, 1), float(init_expectation_nogo))
        self.non_rew_lick_cnt = 0                                                   # reset non-reward lick counter
        self.nogo_streak_cnt = 0                                                    # reset No-Go sensitization counter



# Regroupe les constantes qui etaient des globales de module dans main.py, pour que les
# fonctions d'orchestration (initialization/run_FL1/run_FL2/run_all_wdt/plot_all_results/
# save_run_parameters, dans functions.py) les recoivent en un seul argument plutot qu'une
# longue liste de parametres. Valeurs par defaut = valeurs qui etaient dans main.py.
@dataclass
class SimConfig:
    # switches plots
    PLOT_TRACES: bool = True
    PLOT_BLOCK_HRFA: bool = True
    MAX_TRIALS_BLOCKS: int = 400
    PLOT_SESSIONS_COMPARISON: bool = True
    PLOT_SINGLE_SESSION_ALL: bool = True
    PLOT_STOCHASTIC_IN_POPULATION: bool = True
    PLOT_RPE_ALL: bool = True
    PLOT_ABS_RPE_ALL: bool = True
    PLOT_PSYCHO_TEST: bool = True
    SAVE_PARAMETERS_TXT: bool = True

    # reglages generaux
    NUM_WDT: int = 10
    DEFAULT_TYPES: list = field(default_factory=lambda: [0.0, 1.0])
    WDT_TEST_TYPES: list = field(default_factory=lambda: [0, 0, 0, 0, 0, 0.2, 0.5, 0.7, 1.0, 1.5])

    # modulation du bruit par sigmoide sur le RPE — desactivee, c'est Uncertainty qui pilote l'exploration
    USE_SIGMOID_NOISE: bool = False
    SIG_GAIN_MIN: float = 0.08
    SIG_GAIN_MAX: float = 0.28
    SIG_X0: float = 0.4
    SIG_SLOPE: float = 14.0
    SIG_AVG_LICKS: int = 15
    SIG_UPDATE_EVERY: int = 1
    PLOT_SIGMOID_MAPPING: bool = False

    # simulation de cohorte (plusieurs souris)
    N_MICE: int = 10
    NOISE_GAIN: float = 0.06
    LEARNING_STIM: float = 0.02
    SESSION_NAME: str = "WDT10"

    # decision gate V*M - C
    REWARD_VALUE: float = 1.0
    COST: float = 0.25
    VC_VARY: bool = False
    V_RANGE: tuple = (0.5, 1.0)
    C_RANGE: tuple = (0.0, 0.5)
    VC_BUFFER_SIZE: int = 5

    # seuils de lechage, separes Free Licking / WDT
    LICK_THRS_FL: float = 0.3
    LICK_THRS_WDT: float = 0.40

    # deux stimuli / deux cotes (voir les classes Dual* plus bas)
    dual_stim: bool = False
    FL_FORCED_REWARD_RIGHT_T: float = 500.0
    FL_FORCED_REWARD_LEFT_T: float = 700.0
    V_RIGHT: float = 1.0
    V_LEFT: float = 1.0
    C_RIGHT: float = 0.0
    C_LEFT: float = 0.0
    DUAL_TRIAL_KINDS: list = field(default_factory=lambda: [0, 1, 2])  # 0=catch, 1=stim droite, 2=stim gauche
    DUAL_STIM_AMP: float = 1.0

    # Troisieme mode, independant de dual_stim (gauche/droite) : whisker/auditif avec UNE SEULE
    # Expectation partagee (comme le mono) et un switch de contingence WDT->AUD (paradigme
    # historique). Priorite sur dual_stim si les deux sont actives par erreur (voir main.py).
    WHISKER_AUD_STIM: bool = False
    WA_NUM_SESSIONS: int = 10
    WA_SWITCH_SESSION: int = 6           # sessions 1..N-1 : stim1 (whisker) recompense ; N..fin : stim2 (auditif)
    WA_TRIAL_KINDS: list = field(default_factory=lambda: [0, 1, 2])  # 0=catch, 1=whisker, 2=auditif
    WA_STIM1_AMP: float = 1.0
    WA_STIM2_AMP: float = 1.0

    # Architecture Go/No-Go : toujours active dans la decision (voir Mouse.exp_update_nogo),
    # ces trois valeurs pilotent juste la dynamique de la voie No-Go.
    NOGO_TAU: float = 8.0
    NOGO_GAIN: float = 0.15
    NOGO_RELIEF: float = 0.4
    # Sensibilisation : le tau (persistance) de la voie No-Go grandit petit a petit apres
    # des series de lechages non recompenses consecutifs (meme principe que
    # Mouse.learning_nonrew_lick pour exp_update_no_reward, applique ici a exp_update_nogo).
    # Plafonne par NOGO_TAU_MAX pour eviter une derive sans fin.
    NOGO_STREAK_THRESHOLD: int = 4      # nb de bouts non recompenses consecutifs DANS un essai avant increment
    NOGO_TAU_GROWTH: float = 1.0        # increment de tau_nogo a chaque seuil atteint (0 = desactive)
    NOGO_TAU_MAX: float = 60.0          # plafond du tau_nogo

    # Desapprentissage integre au pipeline normal (pas un scenario a part) : si renseigne
    # (ex. 6), la probabilite de reward est forcee a 0 a partir de WDT{ce numero} inclus,
    # jusqu'a la fin de l'entrainement WDT (WDT_TEST n'est pas concerne). None = comportement
    # actuel, aucune extinction. Declenche en plus les 3 plots de diagnostic (taux de leche,
    # E_go vs E_nogo, barres par session) sur le run reel.
    DELEARNING_FROM_SESSION: Optional[int] = None
    # Dual stim uniquement : si renseigne (+1 = droite, -1 = gauche), seul ce cote perd sa
    # recompense a partir de DELEARNING_FROM_SESSION ; l'autre cote continue normalement.
    # None (par defaut) = comportement existant, les deux cotes perdent la recompense ensemble.
    DELEARNING_SIDE: Optional[int] = None
    # Reapprentissage : si renseigne (ex. 8), la recompense revient a la normale (reward_prob=1)
    # a partir de WDT{ce numero} inclus, sur le(s) cote(s) concerne(s) par DELEARNING_SIDE (ou
    # les deux si None). None = pas de reapprentissage, l'extinction dure jusqu'a la fin.
    DELEARNING_UNTIL_SESSION: Optional[int] = None

    # Population : au lieu d'un run normal (avec tous les plots individuels), fait tourner
    # POPULATION_N_MICE souris dont learning_stim et l'echelle du bruit (mouse.noise[1]) sont
    # tirees uniformement a +/- POPULATION_PARAM_SPREAD autour des valeurs de base (Mouse()),
    # puis compare juste leurs courbes d'apprentissage (voir run_population). Pas d'effet si False.
    POPULATION_RANGE: bool = False
    POPULATION_N_MICE: int = 10
    POPULATION_PARAM_SPREAD: float = 0.10
    # Si True, ignore POPULATION_PARAM_SPREAD et refait tourner la population pour chaque
    # pourcentage de POPULATION_SWEEP_VALUES (un jeu de plots par pourcentage, le pourcentage
    # apparait dans le titre et le nom de fichier), toujours dans le meme dossier population/.
    POPULATION_SPREAD_SWEEP: bool = False
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
    stim1_nogo_gain: float = 0.0
    stim2_nogo_gain: float = 0.0


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
    no_lick_wind: tuple = (3, 3)
    response_wind: tuple = (1, 1)
    reward_prob: float = 1.0
    reward_size: float = 1.0
    forced_reward_right: tuple = (1, 500.0)   # (on/off, temps en s) recompense forcee a droite
    forced_reward_left: tuple = (1, 700.0)    # (on/off, temps en s) recompense forcee a gauche


@dataclass
class DualWDTSessionParams:
    no_lick_wind: tuple = (3, 3)
    iti: tuple = (6, 12)
    response_wind: float = 1.0
    reward_prob: float = 1.0
    # Surcharge de reward_prob par cote (None = utilise reward_prob) : permet de couper la
    # recompense sur un seul cote (desapprentissage cible), l'autre restant a reward_prob.
    reward_prob_right: Optional[float] = None
    reward_prob_left: Optional[float] = None
    reward_size: float = 1.0
    trial_kinds: list = field(default_factory=lambda: [0, 1, 2])   # 0=catch, 1=stim droite, 2=stim gauche
    stim_amp: float = 1.0
    block_numb: int = 5


@dataclass
class DualFreeLickingSessionState:
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
    # Deux Expectations independantes, chacune dans [0, 1] comme dans le modele mono-stimulus —
    # au lieu d'une seule E signee qui doit partager sa capacite entre les deux cotes.
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