import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
from typing import Any, Optional, Sequence, Set, List, Dict


from models import (
    Mouse,
    SessionBaseInfo,
    FreeLickingSessionParams,
    MouseSessionState,
    WDTSesssionParams,
    WDTSesssionState,
    FreeLickingSessionState,
    SimConfig,
    FLSessionResult,
    WDTRunResult,
    WDTBundle,
    DualFreeLickingSessionParams,
    DualFreeLickingSessionState,
    DualWDTSessionParams,
    DualWDTSessionState,
    DualMouseSessionState,
)

from plotting import (
    plot_population_hr_fa,
    plot_traces,
    plot_wdt_block_rates,
    plot_session_rates,
    plot_single_session_rates,
    plot_rpe_per_lick,
    plot_noise_gain_sigmoid,
    plot_abs_rpe_per_lick,
    plot_value_cost_trials,
    save_parameters_txt,
    init_results_dir,
    _save_fig,
    session_rates_dual,
)




def sample_uniform_range(min_val, max_val):
    return min_val + (max_val - min_val) * random.random()


def function_mouse_lick(
    mouse: Mouse,
    mouse_session: MouseSessionState,
    i: int,
    threshold: Optional[float] = None,
    decision_slope: Optional[float] = None,
) -> tuple[int, MouseSessionState]:
    """
    Decision function D:
        D = [E/(1+e^{A_D.(U-U0)}) + Nd/(1+e^{-A_D.(U-U0)})] . [V.M - C]
        Mouse licks if D >= threshold.

    Le signe est bien +A_D.(U-U0) sous l'exponentielle de E (poids qui decroit avec U)
    et -A_D.(U-U0) sous celle de Nd (poids qui croit avec U) : quand l'incertitude U
    augmente au-dela du point neutre U0, le poids sur le bruit Nd augmente (exploration)
    et le poids sur l'expectation E diminue (moins d'exploitation). En dessous de U0
    (souris confiante / naive sans surprise), E domine et le bruit est quasi-nul.
    """
    lick = 0

    motivation_t = mouse_session.motivation[i - 1, 0]
    expectation_t = mouse_session.expectation[i - 1, 0]
    uncertainty_t = mouse_session.uncertainty[i - 1, 0]

    gamma_noise = np.random.gamma(mouse.noise[0], mouse.noise[1])

    A = mouse.decision_slope if decision_slope is None else decision_slope
    u_rel = uncertainty_t - mouse.uncertainty_offset
    w_exploit = 1.0 / (1.0 + np.exp(A * u_rel))     # poids de E, decroit avec U
    w_explore = 1.0 / (1.0 + np.exp(-A * u_rel))    # poids de Nd, croit avec U

    drive = expectation_t * w_exploit + gamma_noise * w_explore

    # decision gate: V*M - C (reward value * motivation, minus cost of licking)
    decision_gate = mouse.reward_value * motivation_t - mouse.cost
    p_lick_t = drive * decision_gate

    lick_thrs = mouse.lick_thrs if threshold is None else threshold
    if p_lick_t >= lick_thrs:
        lick = 1

    mouse_session.p_lick[i, 0] = p_lick_t
    mouse_session.lick[i, 0] = lick

    return lick, mouse_session



def function_update_mouse_state(
    mouse: Mouse,
    mouse_session: MouseSessionState,
    session_info: SessionBaseInfo,
    i: int,
    stim_t: float,
    reward_t: float
) -> tuple[Mouse, MouseSessionState]:

    t = i * session_info.resolution
    time_vector = session_info.time_vector

    if reward_t > 0:
        mouse_session.motivation[i, 0] = (
            mouse_session.motivation[i - 1, 0] - mouse.motivation[1] * reward_t
        )
    else:
        mouse_session.motivation[i, 0] = mouse_session.motivation[i - 1, 0]

    rpe_t = reward_t - mouse_session.expectation[i - 1, 0]
    mouse_session.rpe[i, 0] = rpe_t

    # Uncertainty: U_{t+1} = U_t + A_U*(2*|RPE_t|-1)^3, sur tout evenement saillant
    # (lick, recompense ou non ; OU reward recu meme sans lick, ex: forced reward) —
    # aligne avec Expectation, qui se met deja a jour sans condition de lick des qu'il
    # y a un reward (ligne ~102). Sans cette extension, un forced reward (le plus gros
    # RPE de toute la session) ne modifiait jamais U puisqu'aucun lick ne l'accompagne.
    mouse_session.uncertainty[i, 0] = mouse_session.uncertainty[i - 1, 0]
    if mouse_session.lick[i - 1, 0] == 1 or reward_t > 0:
        u_prev = mouse_session.uncertainty[i - 1, 0]
        u_new = u_prev + mouse.uncertainty_gain * (2.0 * abs(rpe_t) - 1.0) ** 3
        mouse_session.uncertainty[i, 0] = float(np.clip(u_new, 0.0, mouse.uncertainty_max))

    if reward_t > 0:
        update = (
            rpe_t
            * mouse.exp_update_reward[1]
            * np.exp(-np.maximum(time_vector - t, 0.0) / mouse.exp_update_reward[0])
        )
        mouse_session.expectation[i:, 0] += update[i:]
        mouse_session.expectation = np.clip(mouse_session.expectation, 0, 1)

    if reward_t == 0 and mouse_session.lick[i - 1, 0] == 1:
        update = (
            rpe_t
            * mouse.exp_update_no_reward[1]
            * np.exp(-np.maximum(time_vector - t, 0.0) / mouse.exp_update_no_reward[0])
        )
        mouse_session.expectation[i:, 0] += update[i:]
        mouse_session.expectation = np.clip(mouse_session.expectation, 0, 1)
        mouse_session.non_rew_lick_cnt += 1
    elif reward_t > 0 and mouse_session.lick[i - 1, 0] == 1:
        mouse_session.non_rew_lick_cnt = 0

    if stim_t > 0:
        stim_gain = mouse.exp_update_stim[1]
        stim_tau = mouse.exp_update_stim[0]

        update_exp = stim_t * stim_gain * np.exp(-np.maximum(time_vector - t, 0.0) / stim_tau)
        mouse_session.expectation[i:, 0] += update_exp[i:]
        mouse_session.expectation = np.clip(mouse_session.expectation, 0, 1)

        update_elig = stim_t * np.exp(-np.maximum(time_vector - t, 0.0) / mouse.tau_eligibility)
        mouse_session.eligibility[i:, 0] += update_elig[i:]
        mouse_session.eligibility = np.clip(mouse_session.eligibility, 0, 1)

    if mouse_session.non_rew_lick_cnt >= mouse.learning_nonrew_lick[0]:
        old_tau, gain = mouse.exp_update_no_reward
        mouse.exp_update_no_reward = (old_tau + mouse.learning_nonrew_lick[1], gain)
        mouse_session.non_rew_lick_cnt = 0

    if reward_t > 0:
        new_gain = mouse.exp_update_stim[1] + rpe_t * mouse.learning_stim * mouse_session.eligibility[i, 0]
        mouse.exp_update_stim = (mouse.exp_update_stim[0], new_gain)
        
    return mouse, mouse_session


def function_fl_session(
    session_param: FreeLickingSessionParams,
    fl_session,
    session_info: SessionBaseInfo,
    lick: int,
    i: int) -> tuple[float, float, Any]:
    t = i * session_info.resolution
    reward = 0.0
    stim = 0.0  

    if session_param.forced_reward[0] == 1 and np.isclose(t, session_param.forced_reward[1], atol=1e-6):
        reward = session_param.reward_size
        fl_session.last_reward_time = t

    # Avant la forced reward (si programmee), un lick spontane n'est jamais recompense :
    # l'expectation est censee etre a 0, donc il n'y a aucune raison biologique qu'un lick
    # "au hasard" (bruit) declenche une recompense avant que la souris ait decouvert la tache.
    reward_eligible = (
        session_param.forced_reward[0] == 0
        or t >= session_param.forced_reward[1]
    )

    if lick == 1:
        if reward_eligible and (t - fl_session.last_lick_time) > fl_session.no_lick_wind:
            fl_session.no_lick_wind = sample_uniform_range(*session_param.no_lick_wind)
            if random.random() <= session_param.reward_prob:
                reward = session_param.reward_size
                fl_session.last_reward_time = t
        fl_session.last_lick_time = t

    fl_session.reward[i, 0] = reward
    fl_session.stim1[i, 0] = stim

    return reward, stim, fl_session


def function_wdt_session(
    session_param: WDTSesssionParams,
    wdt_session: WDTSesssionState,
    session_info: SessionBaseInfo,
    lick: int,
    i: int
) -> tuple[float, float, WDTSesssionState, bool]:
    t = i * session_info.resolution
    reward = 0.0
    stim = 0.0
    new_trial = False


    if (
        (t - wdt_session.last_lick_time) > wdt_session.no_lick_wind
        and (t - wdt_session.last_trial_time) > wdt_session.iti
    ):
        wdt_session.trial_times.append(t)
        wdt_session.last_trial_time = t
        wdt_session.no_lick_wind = sample_uniform_range(*session_param.no_lick_wind)
        wdt_session.iti = sample_uniform_range(*session_param.iti)
        new_trial = True

        trial_type = random.choice(session_param.trial_types)
        stim = trial_type

        if stim > 0:
            wdt_session.last_stim_time = t
            window_length = int(round(session_param.response_wind / session_info.resolution))
            end_idx = min(i + window_length, session_info.number_bin)
            wdt_session.reward_window[i+1:end_idx, 0] = 1  # exclude current bin

    if lick == 1:
        wdt_session.last_lick_time = t
        if (
            wdt_session.reward_window[i, 0] > 0.5 and
            (t - wdt_session.last_reward_time) > 2
        ):
            if random.random() <= session_param.reward_prob:
                reward = session_param.reward_size
                wdt_session.last_reward_time = t

    wdt_session.reward[i, 0] = reward
    wdt_session.stim1[i, 0] = stim

    return reward, stim, wdt_session, new_trial


def sample_trial_value_cost(
    mouse: Mouse,
    buf_v,
    buf_c,
    v_range: tuple[float, float] = (0.01, 1.0),
    c_range: tuple[float, float] = (0.0, 1.0),
) -> tuple[float, float, float, float]:
    """
    A appeler quand un nouveau trial WDT demarre (new_trial=True).
    Tire une Value (taille de goutte) et un Cost (distance/difficulte) instantanes
    pour ce trial, les pousse dans les buffers glissants, et regle
    mouse.reward_value / mouse.cost sur la moyenne des derniers trials
    (expected_V, expected_C) utilisee par la fonction de decision.

    Retourne (v_trial, c_trial, expected_v, expected_c) — la valeur brute de
    ce trial ET la moyenne glissante, pour permettre de tracer les deux.
    """
    v_trial = random.uniform(*v_range)
    c_trial = random.uniform(*c_range)
    buf_v.append(v_trial)
    buf_c.append(c_trial)

    expected_v = float(np.mean(buf_v))
    expected_c = float(np.mean(buf_c))
    mouse.reward_value = expected_v
    mouse.cost = expected_c

    return v_trial, c_trial, expected_v, expected_c


def function_performance_wdt(
    wdt_params: WDTSesssionParams,
    session_info: SessionBaseInfo,
    wdt_session: WDTSesssionState,
    mouse_session: MouseSessionState
) -> np.ndarray:
    sr = int(round(1 / session_info.resolution))
    trials = [t for t in wdt_session.trial_times if t < (session_info.duration * 60 - wdt_params.response_wind)]
    
    performance = []

    for t in trials:
        pt1 = int(round(t * sr))
        pt2 = pt1 + int(round(wdt_params.response_wind * sr))
        pt2 = min(pt2, session_info.number_bin)

        stim_segment = wdt_session.stim1[pt1:pt2, 0]
        lick_segment = mouse_session.lick[pt1:pt2, 0]
        reward_segment = wdt_session.reward[pt1:pt2, 0]

        stim_amp = np.max(stim_segment)
        lick_detected = np.any(lick_segment)
        reward_detected = np.any(reward_segment)

        # Latency (index of first lick in sec)
        latency = np.argmax(lick_segment) / sr if lick_detected else np.nan

        # Trial outcome
        if stim_amp > 0 and not lick_detected:
            outcome = 0  # Miss
        elif stim_amp > 0 and lick_detected:
            outcome = 1  # Hit
        elif stim_amp == 0 and not lick_detected:
            outcome = 2  # Correct Rejection
        elif stim_amp == 0 and lick_detected:
            outcome = 3  # False Alarm

        performance.append([t, stim_amp, int(lick_detected), latency, int(reward_detected), outcome])

    return np.array(performance)


def simulate_mouse_and_get_session_perf(
    noise_gain: float,
    learning_stim: float,
    session_name: str = "WDT3",
    seed: Optional[int] = None,
    num_wdt_sessions: int = 10,
    multiamp_sessions: Optional[Set[int]] = None,      # <- Optional
    default_types: Optional[Sequence[float]] = None,   # <- Optional
    multiamp_types: Optional[Sequence[float]] = None,  # <- Optional
    include_wdt_test: bool = True
) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    if multiamp_sessions is None:
        multiamp_sessions = set()
    if default_types is None:
        default_types = [0.0, 1.0]
    if multiamp_types is None:
        multiamp_types = [0, 0, 0, 0, 0, 0.2, 0.5, 0.7, 1.0, 1.5]

    si = SessionBaseInfo()
    mouse = Mouse(noise=(1.2, 3.0, float(noise_gain)), learning_stim=float(learning_stim))

    # FL1
    flp = FreeLickingSessionParams()
    fls = FreeLickingSessionState(); fls.initialize(si.number_bin, flp.no_lick_wind)
    ms  = MouseSessionState();       ms.initialize(si.number_bin, mouse.motivation[0])
    for i in range(1, si.number_bin):
        r, s, fls = function_fl_session(flp, fls, si, int(ms.lick[i-1, 0]), i)
        _, ms = function_mouse_lick(mouse, ms, i)
        mouse, ms = function_update_mouse_state(mouse, ms, si, i, s, r)

    # FL2
    init_expect = (np.sum(fls.reward) / np.sum(ms.lick)) if np.sum(ms.lick) > 0 else 0.0
    flp2 = FreeLickingSessionParams(); flp2.forced_reward = (0, 0)
    fls2 = FreeLickingSessionState();  fls2.initialize(si.number_bin, flp2.no_lick_wind)
    ms2  = MouseSessionState();        ms2.initialize(si.number_bin, mouse.motivation[0], init_expect)
    for i in range(1, si.number_bin):
        r, s, fls2 = function_fl_session(flp2, fls2, si, int(ms2.lick[i-1, 0]), i)
        _, ms2 = function_mouse_lick(mouse, ms2, i)
        mouse, ms2 = function_update_mouse_state(mouse, ms2, si, i, s, r)

    prev_ms, prev_rw = ms2, fls2.reward
    perfs: dict[str, np.ndarray] = {}

    # WDT1 a WDT{num_wdt_sessions}
    for k in range(1, int(num_wdt_sessions) + 1):
        wp = build_wdt_params_for_session(
            k,
            default_types=default_types,
            multiamp_sessions=multiamp_sessions,
            multiamp_types=multiamp_types,
        )
        init_expect = (np.sum(prev_rw) / np.sum(prev_ms.lick)) if np.sum(prev_ms.lick) > 0 else 0.0

        ws = WDTSesssionState(); ws.initialize(si.number_bin, wp.no_lick_wind, wp.iti)
        ms = MouseSessionState(); ms.initialize(si.number_bin, mouse.motivation[0], init_expect)

        for i in range(1, si.number_bin):
            r, s, ws, _ = function_wdt_session(wp, ws, si, int(ms.lick[i-1, 0]), i)
            _, ms = function_mouse_lick(mouse, ms, i)
            mouse, ms = function_update_mouse_state(mouse, ms, si, i, s, r)

        perfs[f"WDT{k}"] = function_performance_wdt(wp, si, ws, ms)
        prev_ms, prev_rw = ms, ws.reward

    # WDT_TEST, optionnel
    if include_wdt_test:
        wp_test = WDTSesssionParams()
        wp_test.trial_types = list(multiamp_types) if multiamp_types is not None else \
        [0, 0, 0, 0, 0, 0.2, 0.5, 0.7, 1.0, 1.5]
        init_expect = (np.sum(prev_rw) / np.sum(prev_ms.lick)) if np.sum(prev_ms.lick) > 0 else 0.0

        ws_test = WDTSesssionState(); ws_test.initialize(si.number_bin, wp_test.no_lick_wind, wp_test.iti)
        ms_test = MouseSessionState(); ms_test.initialize(si.number_bin, mouse.motivation[0], init_expect)

        for i in range(1, si.number_bin):
            r, s, ws_test, _ = function_wdt_session(wp_test, ws_test, si, int(ms_test.lick[i-1, 0]), i)
            _, ms_test = function_mouse_lick(mouse, ms_test, i)
            mouse, ms_test = function_update_mouse_state(mouse, ms_test, si, i, s, r)

        perfs["WDT_TEST"] = function_performance_wdt(wp_test, si, ws_test, ms_test)

    if session_name not in perfs:
        raise KeyError(
            f"Session inconnue: {session_name} (attendu: "
            f"WDT1..WDT{num_wdt_sessions}{' ou WDT_TEST' if include_wdt_test else ''})"
        )
    return perfs[session_name]




def extract_rpe_per_lick(mouse_session: MouseSessionState,
                         reward_trace: np.ndarray,
                         stim_trace: Optional[np.ndarray],
                         session_info: SessionBaseInfo,
                         ignore_last_without_next: bool = True
                        ) -> np.ndarray:
    """
    Construit un tableau (N_licks x 6):
      [lick_idx, t_sec, rpe_at_lick, reward_flag, stim_amp, bin_index]
    - rpe_at_lick = reward[i+1] - expectation[i]
    - reward_flag = 1 si reward[i+1] > 0 sinon 0
    - stim_amp = stim_trace[i] (0 si FL ou stim_trace None)
    - on ignore le dernier lick si i+1 sort du vecteur (paramètre)
    Hypothèse: arrays (T,1). On gère via ravel().
    """
    lick = np.asarray(mouse_session.lick).ravel()
    expc = np.asarray(mouse_session.expectation).ravel()
    rew  = np.asarray(reward_trace).ravel()
    stim = np.asarray(stim_trace).ravel() if stim_trace is not None else None

    idxs = np.flatnonzero(lick == 1)

    if ignore_last_without_next:
        idxs = idxs[idxs + 1 < len(rew)]

    if idxs.size == 0:
        return np.zeros((0, 6), dtype=float)

    # rpe = reward[i+1] - expectation[i]
    rpe_vals = rew[idxs + 1] - expc[idxs]
    reward_flag = (rew[idxs + 1] > 0).astype(float)
    stim_amp = (stim[idxs] if stim is not None else np.zeros_like(idxs, dtype=float)).astype(float)
    t_sec = idxs.astype(float) * float(session_info.resolution)
    lick_idx = np.arange(1, len(idxs) + 1, dtype=float)

    out = np.column_stack([
        lick_idx,          # 0
        t_sec,             # 1
        rpe_vals.astype(float),  # 2
        reward_flag,       # 3
        stim_amp,          # 4
        idxs.astype(float) # 5 bin index
    ])
    return out


def build_wdt_params_for_session(
    session_idx: int,
    default_types: Optional[Sequence[float]] = None,   
    multiamp_sessions: Optional[Set[int]] = None,      
    multiamp_types: Optional[Sequence[float]] = None
) -> WDTSesssionParams:
    """
    Fabrique un WDTSesssionParams pour une session donnée.
    - Par défaut: binaire [0, 1.0]
    - Si session_idx ∈ multiamp_sessions: utilise multiamp_types
    """
    if default_types is None:
        default_types = [0.0, 1.0]
    if multiamp_sessions is None:
        multiamp_sessions = set()
    if multiamp_types is None:
        multiamp_types = [0, 0, 0, 0, 0, 0.2, 0.5, 0.7, 1.0, 1.5]

    p = WDTSesssionParams()
    p.trial_types = list(multiamp_types) if session_idx in multiamp_sessions else list(default_types)
    return p



# Aides pour la modulation du bruit par sigmoide, sur la moyenne des |RPE| aux licks recents
import numpy as _np
from collections import deque as _deque

def sigmoid_gain(m: float, m0: float, k: float, gmin: float, gmax: float) -> float:
    """g(m) = gmin + (gmax-gmin)/(1 + exp(-k*(m-m0)))."""
    return float(gmin + (gmax - gmin) * (1.0 / (1.0 + _np.exp(-k * (m - m0)))))

def new_rpe_buffer(maxlen: int):
    """Crée un buffer (deque) pour |RPE| aux licks + un compteur de licks."""
    return _deque(maxlen=int(maxlen)), 0

def online_sigmoid_update(
    mouse,
    mouse_session,
    i: int,
    buf,
    lick_counter: int,
    enabled: bool,
    m0: float,
    k: float,
    gmin: float,
    gmax: float,
    update_every: int,
) -> int:
    """
    Pousse |RPE[i]| si lick à i-1, et toutes les 'update_every' licks,
    met à jour le gain de bruit via la sigmoïde g(|RPE|_moy).
    Ne touche qu'au 'gain' (mouse.noise[2]).
    """
    if not enabled:
        return lick_counter

    if mouse_session.lick[i - 1, 0] == 1:
        buf.append(abs(mouse_session.rpe[i, 0]))
        lick_counter += 1

        if (lick_counter % int(update_every)) == 0 and len(buf) > 0:
            m = float(_np.mean(buf))
            new_gain = sigmoid_gain(m, m0=m0, k=k, gmin=gmin, gmax=gmax)
            # mouse.noise = (shape, scale, gain)
            mouse.noise = (mouse.noise[0], mouse.noise[1], float(new_gain))

    return lick_counter



from typing import Iterable, Optional, Sequence, Set, Tuple, List, Dict

def population_hr_fa_plot(
    n_mice: int = 10,
    noise_gain: float = 0.10,
    learning_stim: float = 0.02,
    session_name: str = "WDT1",
    seed: Optional[int] = None,
    num_wdt_sessions: int = 10,
    multiamp_sessions: Optional[Set[int]] = None,
    default_types: Optional[Sequence[float]] = None,
    multiamp_types: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Lance n_mice souris avec les mêmes paramètres, récupère la performance de `session_name`,
    puis trace HR/FA façon cohorte. Retourne (hr_arr, fa_arr, dp_arr).
    """
    perfs: List[np.ndarray] = []
    labels: List[str] = []

    base_seed = seed if seed is not None else None
    for k in range(n_mice):
        this_seed = None if base_seed is None else base_seed + k
        perf_k = simulate_mouse_and_get_session_perf(
            noise_gain=noise_gain,
            learning_stim=learning_stim,
            session_name=session_name,
            seed=this_seed,
            num_wdt_sessions=num_wdt_sessions,
            multiamp_sessions=multiamp_sessions,
            default_types=default_types,
            multiamp_types=multiamp_types,
            include_wdt_test=True,
        )
        perfs.append(perf_k)
        labels.append(f"M{k+1}")

    hr_arr, fa_arr, dp_arr = plot_population_hr_fa(
        perfs,
        labels,
        session_name=session_name,
        noise_gain=float(noise_gain),
        learning_stim=float(learning_stim),
    )
    return hr_arr, fa_arr, dp_arr


# Fonctions d'orchestration appelees par main.py. Elles reproduisent exactement l'ancien
# main.py (meme ordre d'appel, donc meme ordre de tirages aleatoires), juste rangees en
# fonctions au lieu d'etre ecrites a plat dans le script.

def initialization(config: SimConfig) -> tuple[SessionBaseInfo, Mouse]:
    """Cree le dossier de resultats (nomme d'apres V/C/seuil WDT, + dualstim si actif) et l'etat global (session_info, mouse)."""
    suffix = (
        ("VCvary" if config.VC_VARY else f"V{config.REWARD_VALUE:g}_C{config.COST:g}")
        + f"_wdtThrs{config.LICK_THRS_WDT:g}"
        + ("_dualstim" if config.dual_stim else "")
    )
    init_results_dir(suffix=suffix)

    session_info = SessionBaseInfo()
    mouse = Mouse(
        reward_value=config.REWARD_VALUE,
        cost=config.COST,
        lick_thrs=config.LICK_THRS_FL,
        lick_thrs_wdt=config.LICK_THRS_WDT,
    )
    return session_info, mouse


def run_FL1(mouse: Mouse, session_info: SessionBaseInfo, config: SimConfig) -> FLSessionResult:
    """Free Licking — Session 1 (avec forced reward a t=500s)."""
    fl_params = FreeLickingSessionParams()
    fl_session = FreeLickingSessionState()
    fl_session.initialize(session_info.number_bin, fl_params.no_lick_wind)
    mouse_session = MouseSessionState()
    mouse_session.initialize(session_info.number_bin, mouse.motivation[0])

    buf_fl1, cnt_fl1 = new_rpe_buffer(config.SIG_AVG_LICKS)

    for i in range(1, session_info.number_bin):
        reward_t, stim_t, fl_session = function_fl_session(
            fl_params, fl_session, session_info, int(mouse_session.lick[i - 1, 0]), i
        )
        _, mouse_session = function_mouse_lick(mouse, mouse_session, i)
        mouse, mouse_session = function_update_mouse_state(mouse, mouse_session, session_info, i, stim_t, reward_t)

        cnt_fl1 = online_sigmoid_update(
            mouse, mouse_session, i,
            buf_fl1, cnt_fl1,
            enabled=config.USE_SIGMOID_NOISE,
            m0=config.SIG_X0, k=config.SIG_SLOPE,
            gmin=config.SIG_GAIN_MIN, gmax=config.SIG_GAIN_MAX,
            update_every=config.SIG_UPDATE_EVERY,
        )

    return FLSessionResult(session=fl_session, mouse_session=mouse_session)


def run_FL2(mouse: Mouse, session_info: SessionBaseInfo, config: SimConfig, log_fl1: FLSessionResult) -> FLSessionResult:
    """Free Licking — Session 2 (pas de forced reward ; Expectation/Uncertainty heritees de FL1)."""
    fl_params = FreeLickingSessionParams()
    fl_params.forced_reward = (0, 0)

    lick_sum = float(np.sum(log_fl1.mouse_session.lick))
    reward_sum = float(np.sum(log_fl1.session.reward))
    init_expect = (reward_sum / lick_sum) if lick_sum > 0 else 0.0
    init_uncert = float(log_fl1.mouse_session.uncertainty[-1, 0])

    fl_session = FreeLickingSessionState()
    fl_session.initialize(session_info.number_bin, fl_params.no_lick_wind)
    mouse_session = MouseSessionState()
    mouse_session.initialize(session_info.number_bin, mouse.motivation[0], init_expect, init_uncert)

    buf_fl2, cnt_fl2 = new_rpe_buffer(config.SIG_AVG_LICKS)

    for i in range(1, session_info.number_bin):
        reward_t, stim_t, fl_session = function_fl_session(
            fl_params, fl_session, session_info, int(mouse_session.lick[i - 1, 0]), i
        )
        _, mouse_session = function_mouse_lick(mouse, mouse_session, i)
        mouse, mouse_session = function_update_mouse_state(mouse, mouse_session, session_info, i, stim_t, reward_t)

        cnt_fl2 = online_sigmoid_update(
            mouse, mouse_session, i,
            buf_fl2, cnt_fl2,
            enabled=config.USE_SIGMOID_NOISE,
            m0=config.SIG_X0, k=config.SIG_SLOPE,
            gmin=config.SIG_GAIN_MIN, gmax=config.SIG_GAIN_MAX,
            update_every=config.SIG_UPDATE_EVERY,
        )

    return FLSessionResult(session=fl_session, mouse_session=mouse_session)


def run_wdt_session(
    mouse: Mouse,
    session_info: SessionBaseInfo,
    config: SimConfig,
    wdt_params: WDTSesssionParams,
    prev_mouse_session: MouseSessionState,
    prev_reward: np.ndarray,
    label: str,
) -> WDTRunResult:
    """
    Corps d'une session WDT (une seule fonction, reutilisee pour WDT1..N ET pour WDT_TEST —
    seul `wdt_params.trial_types` et `label` different entre les deux usages).
    """
    lick_sum = float(np.sum(prev_mouse_session.lick))
    reward_sum = float(np.sum(prev_reward))
    init_expect = (reward_sum / lick_sum) if lick_sum > 0 else 0.0
    init_uncert = float(prev_mouse_session.uncertainty[-1, 0])

    wdt_session = WDTSesssionState()
    wdt_session.initialize(session_info.number_bin, wdt_params.no_lick_wind, wdt_params.iti)
    mouse_session = MouseSessionState()
    mouse_session.initialize(session_info.number_bin, mouse.motivation[0], init_expect, init_uncert)

    buf_wdt, cnt_wdt = new_rpe_buffer(config.SIG_AVG_LICKS)
    buf_v, buf_c = deque(maxlen=config.VC_BUFFER_SIZE), deque(maxlen=config.VC_BUFFER_SIZE)
    vc_log = {"trial_times": [], "v": [], "c": [], "expected_v": [], "expected_c": []}

    for i in range(1, session_info.number_bin):
        reward_t, stim_t, wdt_session, new_trial = function_wdt_session(
            wdt_params, wdt_session, session_info, int(mouse_session.lick[i - 1, 0]), i
        )
        if config.VC_VARY and new_trial:
            v_trial, c_trial, ev, ec = sample_trial_value_cost(mouse, buf_v, buf_c, config.V_RANGE, config.C_RANGE)
            vc_log["trial_times"].append(i * session_info.resolution)
            vc_log["v"].append(v_trial)
            vc_log["c"].append(c_trial)
            vc_log["expected_v"].append(ev)
            vc_log["expected_c"].append(ec)

        _, mouse_session = function_mouse_lick(
            mouse, mouse_session, i, threshold=mouse.lick_thrs_wdt, decision_slope=mouse.decision_slope_wdt
        )
        mouse, mouse_session = function_update_mouse_state(mouse, mouse_session, session_info, i, stim_t, reward_t)

        cnt_wdt = online_sigmoid_update(
            mouse, mouse_session, i,
            buf_wdt, cnt_wdt,
            enabled=config.USE_SIGMOID_NOISE,
            m0=config.SIG_X0, k=config.SIG_SLOPE,
            gmin=config.SIG_GAIN_MIN, gmax=config.SIG_GAIN_MAX,
            update_every=config.SIG_UPDATE_EVERY,
        )

    perf = function_performance_wdt(
        wdt_params=wdt_params,
        session_info=session_info,
        wdt_session=wdt_session,
        mouse_session=mouse_session,
    )

    return WDTRunResult(label=label, session=wdt_session, mouse_session=mouse_session, perf=perf, vc_log=vc_log)


def run_all_wdt(mouse: Mouse, session_info: SessionBaseInfo, config: SimConfig, log_fl2: FLSessionResult) -> WDTBundle:
    """Enchaine WDT1..WDT{config.NUM_WDT}, Expectation/Uncertainty heritees de la session precedente."""
    bundle = WDTBundle()
    prev_mouse_session = log_fl2.mouse_session
    prev_reward = log_fl2.session.reward

    for s_idx in range(1, config.NUM_WDT + 1):
        wdt_params = WDTSesssionParams()
        wdt_params.trial_types = config.DEFAULT_TYPES[:]

        label = f"WDT{s_idx}"
        result = run_wdt_session(mouse, session_info, config, wdt_params, prev_mouse_session, prev_reward, label)

        if result.perf.size > 0:
            stim_mask = result.perf[:, 1] > 0
            catch_mask = ~stim_mask
            hr = (np.sum((result.perf[:, 5] == 1) & stim_mask) / np.sum(stim_mask)) if np.sum(stim_mask) > 0 else 0.0
            fa = (np.sum((result.perf[:, 5] == 3) & catch_mask) / np.sum(catch_mask)) if np.sum(catch_mask) > 0 else 0.0
        else:
            hr, fa = 0.0, 0.0

        print(f"Hit Rate {label} : {hr:.2f}")
        print(f"FA  Rate {label} : {fa:.2f}")

        bundle.results.append(result)
        bundle.labels.append(label)

        prev_mouse_session = result.mouse_session
        prev_reward = result.session.reward

    return bundle


def run_wdt_test(mouse: Mouse, session_info: SessionBaseInfo, config: SimConfig, wdt_bundle: WDTBundle) -> WDTRunResult:
    """WDT_TEST — courbe psychometrique multi-amp, heritee de la derniere session WDT."""
    wdt_test_params = WDTSesssionParams()
    wdt_test_params.trial_types = config.WDT_TEST_TYPES[:]

    return run_wdt_session(
        mouse, session_info, config, wdt_test_params,
        wdt_bundle.last_mouse_session, wdt_bundle.last_reward, "WDT_TEST",
    )


def plot_all_results(
    session_info: SessionBaseInfo,
    mouse: Mouse,
    config: SimConfig,
    log_fl1: FLSessionResult,
    log_fl2: FLSessionResult,
    wdt_bundle: WDTBundle,
    wdt_test: WDTRunResult,
) -> None:
    """Tous les plots, dans le meme ordre que l'ancien main.py."""

    if config.PLOT_TRACES:
        plot_traces(session_info.time_vector, log_fl1.mouse_session, log_fl1.session.reward,
                    stim_array=None, title="Free Licking — Session 1",
                    save_name="free_licking_1", threshold=mouse.lick_thrs)
        plot_traces(session_info.time_vector, log_fl2.mouse_session, log_fl2.session.reward,
                    stim_array=None, title="Free Licking — Session 2",
                    save_name="free_licking_2", threshold=mouse.lick_thrs)
        for lbl, result in zip(wdt_bundle.labels, wdt_bundle.results):
            plot_traces(session_info.time_vector, result.mouse_session, result.session.reward,
                        stim_array=result.session.stim1, title=f"Whisker Detection Task — {lbl}",
                        save_name=lbl.lower(), threshold=mouse.lick_thrs_wdt)
        plot_traces(session_info.time_vector, wdt_test.mouse_session, wdt_test.session.reward,
                    stim_array=wdt_test.session.stim1, title="WDT Test (multi-amp)",
                    save_name="wdt_test", threshold=mouse.lick_thrs_wdt)

        if config.VC_VARY:
            for lbl, result in zip(wdt_bundle.labels, wdt_bundle.results):
                vc = result.vc_log
                if vc["trial_times"]:
                    plot_value_cost_trials(
                        vc["trial_times"], vc["v"], vc["c"], vc["expected_v"], vc["expected_c"],
                        title=f"Whisker Detection Task — {lbl} — Value/Cost",
                        save_name=f"{lbl.lower()}_value_cost",
                    )
            if wdt_test.vc_log["trial_times"]:
                vc = wdt_test.vc_log
                plot_value_cost_trials(
                    vc["trial_times"], vc["v"], vc["c"], vc["expected_v"], vc["expected_c"],
                    title="WDT Test (multi-amp) — Value/Cost",
                    save_name="wdt_test_value_cost",
                )

    if config.PLOT_BLOCK_HRFA:
        _wdt_params = WDTSesssionParams()
        for lbl, result in zip(wdt_bundle.labels, wdt_bundle.results):
            plot_wdt_block_rates(result.perf, _wdt_params, max_trials=config.MAX_TRIALS_BLOCKS,
                                 title=f"{lbl} — HR/FA by block", save_name=lbl.lower())

    if config.PLOT_SESSIONS_COMPARISON:
        wdt_perfs = [r.perf for r in wdt_bundle.results]
        plot_session_rates(wdt_perfs, wdt_bundle.labels, title="Learning — HR, FA & d′ (WDT1..WDT10)",
                            save_name="hr_fa_dprime_all_sessions")

    if config.PLOT_SINGLE_SESSION_ALL:
        for lbl, result in zip(wdt_bundle.labels, wdt_bundle.results):
            plot_single_session_rates(result.perf, title=f"{lbl} — HR vs FA", save_name=lbl.lower())

    if config.PLOT_RPE_ALL:
        tbl_fl1 = extract_rpe_per_lick(log_fl1.mouse_session, log_fl1.session.reward, None, session_info)
        plot_rpe_per_lick(tbl_fl1, title="FL1 — RPE per lick", save_name="free_licking_1_rpe")
        tbl_fl2 = extract_rpe_per_lick(log_fl2.mouse_session, log_fl2.session.reward, None, session_info)
        plot_rpe_per_lick(tbl_fl2, title="FL2 — RPE per lick", save_name="free_licking_2_rpe")
        for lbl, result in zip(wdt_bundle.labels, wdt_bundle.results):
            tbl = extract_rpe_per_lick(result.mouse_session, result.session.reward, result.session.stim1, session_info)
            plot_rpe_per_lick(tbl, title=f"{lbl} — RPE per lick", save_name=f"{lbl.lower()}_rpe")

    if config.PLOT_ABS_RPE_ALL:
        tbl_fl1 = extract_rpe_per_lick(log_fl1.mouse_session, log_fl1.session.reward, None, session_info)
        plot_abs_rpe_per_lick(tbl_fl1, title="FL1 — |RPE| per lick", save_name="free_licking_1_abs_rpe")
        tbl_fl2 = extract_rpe_per_lick(log_fl2.mouse_session, log_fl2.session.reward, None, session_info)
        plot_abs_rpe_per_lick(tbl_fl2, title="FL2 — |RPE| per lick", save_name="free_licking_2_abs_rpe")
        for lbl, result in zip(wdt_bundle.labels, wdt_bundle.results):
            tbl = extract_rpe_per_lick(result.mouse_session, result.session.reward, result.session.stim1, session_info)
            plot_abs_rpe_per_lick(tbl, title=f"{lbl} — |RPE| per lick", save_name=f"{lbl.lower()}_abs_rpe")

    if config.PLOT_PSYCHO_TEST and (wdt_test.perf is not None and wdt_test.perf.size):
        perf = wdt_test.perf
        stim_amps = perf[:, 1]
        outcomes = perf[:, 5]
        unique_stims = np.unique(stim_amps)
        hit_rates = []
        for s in unique_stims:
            m = stim_amps == s
            hit_rates.append(np.sum(outcomes[m] == 1) / np.sum(m) if np.sum(m) > 0 else np.nan)
        fig = plt.figure(figsize=(8, 5))
        plt.plot(unique_stims, hit_rates, 'o-', label='Hit rate')
        plt.xlabel('Stimulus amplitude'); plt.ylabel('Hit rate')
        plt.title('Psychometric curve — WDT Test (multi-amp)')
        plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
        _save_fig(fig, category="psychometric", name="hit_rate_vs_stimulus_amplitude")

    if config.PLOT_SIGMOID_MAPPING:
        plot_noise_gain_sigmoid(
            m0=config.SIG_X0,
            k=config.SIG_SLOPE,
            gmin=config.SIG_GAIN_MIN,
            gmax=config.SIG_GAIN_MAX,
            title="Noise gain mapping g(m)",
            save_name="noise_gain_mapping",
        )

    if config.PLOT_STOCHASTIC_IN_POPULATION:
        population_hr_fa_plot(
            n_mice=config.N_MICE,
            noise_gain=config.NOISE_GAIN,
            learning_stim=config.LEARNING_STIM,
            session_name=config.SESSION_NAME,
        )


def save_run_parameters(config: SimConfig, mouse: Mouse, session_info: SessionBaseInfo) -> None:
    """Ecrit parameters.txt — recapitulatif de tous les parametres numeriques du run."""
    _ref_mouse = Mouse()
    _ref_session = SessionBaseInfo()
    _ref_fl = FreeLickingSessionParams()
    _ref_wdt = WDTSesssionParams()

    param_rows = [
        ("SESSION TIMING", None),
        ("Session duration (min)", _ref_session.duration),
        ("Time resolution (s)", _ref_session.resolution),
        ("Number of time bins", _ref_session.number_bin),

        ("MOUSE — NOISE & DECISION", None),
        ("Noise gamma shape", _ref_mouse.noise[0]),
        ("Noise gamma scale", _ref_mouse.noise[1]),
        ("Lick threshold — Free Licking", config.LICK_THRS_FL),
        ("Lick threshold — WDT", config.LICK_THRS_WDT),

        ("MOUSE — MOTIVATION", None),
        ("Motivation (initial)", _ref_mouse.motivation[0]),
        ("Motivation loss per reward", _ref_mouse.motivation[1]),

        ("MOUSE — UNCERTAINTY (DECISION BLEND E/Nd)", None),
        ("Decision slope — Free Licking (A_D)", _ref_mouse.decision_slope),
        ("Decision slope — WDT (A_D)", _ref_mouse.decision_slope_wdt),
        ("Uncertainty gain (A_U)", _ref_mouse.uncertainty_gain),
        ("Uncertainty max (U_max)", _ref_mouse.uncertainty_max),

        ("MOUSE — VALUE & COST (DECISION GATE: V·M − C)", None),
        ("Reward value (V)", config.REWARD_VALUE),
        ("Cost (C)", config.COST),
        ("Per-trial random V/C (WDT only)", config.VC_VARY),
        ("Value range (droplet size)", f"{config.V_RANGE[0]}–{config.V_RANGE[1]}" if config.VC_VARY else "n/a"),
        ("Cost range (spout distance)", f"{config.C_RANGE[0]}–{config.C_RANGE[1]}" if config.VC_VARY else "n/a"),
        ("Expected V/C averaging window (trials)", config.VC_BUFFER_SIZE if config.VC_VARY else "n/a"),

        ("MOUSE — EXPECTATION UPDATE (REWARD)", None),
        ("Tau (reward, s)", _ref_mouse.exp_update_reward[0]),
        ("Gain (reward)", _ref_mouse.exp_update_reward[1]),

        ("MOUSE — EXPECTATION UPDATE (NO REWARD)", None),
        ("Tau (no reward, initial, s)", _ref_mouse.exp_update_no_reward[0]),
        ("Gain (no reward)", _ref_mouse.exp_update_no_reward[1]),

        ("MOUSE — EXPECTATION UPDATE (STIMULUS)", None),
        ("Tau (stimulus, s)", _ref_mouse.exp_update_stim[0]),
        ("Gain (stimulus, initial)", _ref_mouse.exp_update_stim[1]),

        ("MOUSE — LEARNING RULES", None),
        ("Consecutive non-rewarded licks threshold", _ref_mouse.learning_nonrew_lick[0]),
        ("Tau (no reward) increment", _ref_mouse.learning_nonrew_lick[1]),
        ("Stimulus gain learning rate", _ref_mouse.learning_stim),
        ("Eligibility trace tau (s)", _ref_mouse.tau_eligibility),

        ("NOISE GAIN MODULATION (SIGMOID)", None),
        ("Enabled", config.USE_SIGMOID_NOISE),
        ("Gain min", config.SIG_GAIN_MIN),
        ("Gain max", config.SIG_GAIN_MAX),
        ("Inflection point (m0)", config.SIG_X0),
        ("Slope (k)", config.SIG_SLOPE),
        ("Averaging window (licks)", config.SIG_AVG_LICKS),
        ("Update every (licks)", config.SIG_UPDATE_EVERY),

        ("FREE LICKING SESSIONS", None),
        ("No-lick window (s)", f"{_ref_fl.no_lick_wind[0]}–{_ref_fl.no_lick_wind[1]}"),
        ("Reward probability", _ref_fl.reward_prob),
        ("Reward size", _ref_fl.reward_size),
        ("Forced reward — Session 1 (s)", _ref_fl.forced_reward[1]),
        ("Forced reward — Session 2", "none"),

        ("WHISKER DETECTION TASK (WDT1–10)", None),
        ("Number of WDT sessions", config.NUM_WDT),
        ("No-lick window (s)", f"{_ref_wdt.no_lick_wind[0]}–{_ref_wdt.no_lick_wind[1]}"),
        ("Inter-trial interval (s)", f"{_ref_wdt.iti[0]}–{_ref_wdt.iti[1]}"),
        ("Response window (s)", _ref_wdt.response_wind),
        ("Reward probability", _ref_wdt.reward_prob),
        ("Reward size", _ref_wdt.reward_size),
        ("Trial types (stimulus amplitudes)", config.DEFAULT_TYPES),
        ("Block size (trials)", _ref_wdt.block_numb),

        ("WDT PSYCHOMETRIC TEST", None),
        ("Trial types (stimulus amplitudes)", config.WDT_TEST_TYPES),

        ("POPULATION / COHORT SIMULATION", None),
        ("Number of mice", config.N_MICE),
        ("Noise gain", config.NOISE_GAIN),
        ("Stimulus learning rate", config.LEARNING_STIM),
    ]

    save_parameters_txt(param_rows)


# Deux stimuli, chacun menant a une recompense a un endroit different (droite/gauche).
# Rien ci-dessus n'est modifie ; ces fonctions sont utilisees a la place des precedentes
# uniquement quand config.dual_stim=True (voir main.py).
#
# Expectation (E) est signee, dans [-1, 1] : le signe dit quel cote est actuellement
# attendu, la magnitude dit a quel point c'est confiant. Le cote de la decision est
# tranche par le signe de E AVANT de calculer le "drive" — le bruit et l'Uncertainty
# ne font donc jamais changer de cote, ils modulent seulement si la souris repond ou
# pas sur le cote que E designe deja (meme role que dans le modele a un seul stimulus,
# ou le bruit ne "choisit" jamais rien, il fait juste agir ou pas).
#
# Motivation et Uncertainty restent globales (pas dupliquees par cote), pour permettre
# une vraie interference entre les deux apprentissages plutot que deux souris independantes.


def function_mouse_lick_dual(
    mouse: Mouse,
    mouse_session: DualMouseSessionState,
    i: int,
    threshold: Optional[float] = None,
    decision_slope: Optional[float] = None,
    v_right: Optional[float] = None,
    v_left: Optional[float] = None,
    c_right: Optional[float] = None,
    c_left: Optional[float] = None,
) -> tuple[int, int, DualMouseSessionState]:
    """
    Compare E_droite et E_gauche pour trancher le cote, puis reutilise la meme mecanique
    de decision a seuil unique que le modele a un seul stimulus, appliquee a ce cote avec
    sa propre Expectation.

    Le cote n'est pas lu par une comparaison dure : P(cote=droite) suit une sigmoide de
    (E_droite - E_gauche) (mouse.side_readout_slope). Meme quand un cote est tres confiant,
    il reste une probabilite residuelle de lire l'autre — pas un mecanisme separe "au cas
    ou", ca vient directement du bruit de lecture, comme un decodage bruite d'un code de
    population neuronal (le meme principe qu'une asymptote de lapse en psychophysique,
    mais qui emerge de la lecture plutot que d'etre rajoute a part).
    """
    lick = 0
    lick_side = 0

    motivation_t = mouse_session.motivation[i - 1, 0]
    e_right_t = mouse_session.expectation_right[i - 1, 0]
    e_left_t = mouse_session.expectation_left[i - 1, 0]
    uncertainty_t = mouse_session.uncertainty[i - 1, 0]

    gamma_noise = np.random.gamma(mouse.noise[0], mouse.noise[1])

    A = mouse.decision_slope if decision_slope is None else decision_slope
    u_rel = uncertainty_t - mouse.uncertainty_offset
    w_exploit = 1.0 / (1.0 + np.exp(A * u_rel))
    w_explore = 1.0 / (1.0 + np.exp(-A * u_rel))

    p_right = 1.0 / (1.0 + np.exp(-mouse.side_readout_slope * (e_right_t - e_left_t)))
    side = 1 if random.random() < p_right else -1
    e_side = e_right_t if side == 1 else e_left_t

    v_r = mouse.reward_value if v_right is None else v_right
    v_l = mouse.reward_value if v_left is None else v_left
    c_r = mouse.cost if c_right is None else c_right
    c_l = mouse.cost if c_left is None else c_left
    v_side = v_r if side == 1 else v_l
    c_side = c_r if side == 1 else c_l

    drive = e_side * w_exploit + gamma_noise * w_explore
    decision_gate = v_side * motivation_t - c_side
    d_t = drive * decision_gate

    lick_thrs = mouse.lick_thrs if threshold is None else threshold
    if d_t >= lick_thrs:
        lick = 1
        lick_side = side

    mouse_session.p_lick[i, 0] = float(d_t) * side
    mouse_session.lick[i, 0] = lick
    mouse_session.lick_side[i, 0] = lick_side

    return lick, lick_side, mouse_session


def function_update_mouse_state_dual(
    mouse: Mouse,
    mouse_session: DualMouseSessionState,
    session_info: SessionBaseInfo,
    i: int,
    stim_signed_t: float,
    reward_t: float,
    reward_side_t: int,
) -> tuple[Mouse, DualMouseSessionState]:
    """
    Meme structure que function_update_mouse_state, appliquee independamment au canal
    (droite ou gauche) concerne par l'evenement — reward_side_t pour un reward, le cote
    leche pour un lick non recompense, le cote du stimulus pour l'anticipation. L'autre
    canal n'est jamais touche : chacun se comporte comme l'Expectation du modele mono-
    stimulus, sans jamais partager sa capacite avec l'autre cote.
    """
    t = i * session_info.resolution
    time_vector = session_info.time_vector

    if reward_t > 0:
        mouse_session.motivation[i, 0] = (
            mouse_session.motivation[i - 1, 0] - mouse.motivation[1] * reward_t
        )
    else:
        mouse_session.motivation[i, 0] = mouse_session.motivation[i - 1, 0]

    lick_side_prev = int(mouse_session.lick_side[i - 1, 0])
    ref_side_t = reward_side_t if reward_t > 0 else lick_side_prev

    if ref_side_t == 1:
        e_ref_prev = mouse_session.expectation_right[i - 1, 0]
    elif ref_side_t == -1:
        e_ref_prev = mouse_session.expectation_left[i - 1, 0]
    else:
        e_ref_prev = 0.0
    rpe_t = reward_t - e_ref_prev
    mouse_session.rpe[i, 0] = rpe_t

    mouse_session.uncertainty[i, 0] = mouse_session.uncertainty[i - 1, 0]
    if mouse_session.lick[i - 1, 0] == 1 or reward_t > 0:
        u_prev = mouse_session.uncertainty[i - 1, 0]
        u_new = u_prev + mouse.uncertainty_gain * (2.0 * abs(rpe_t) - 1.0) ** 3
        mouse_session.uncertainty[i, 0] = float(np.clip(u_new, 0.0, mouse.uncertainty_max))

    if reward_t > 0 and ref_side_t != 0:
        update = (
            rpe_t
            * mouse.exp_update_reward[1]
            * np.exp(-np.maximum(time_vector - t, 0.0) / mouse.exp_update_reward[0])
        )
        if ref_side_t == 1:
            mouse_session.expectation_right[i:, 0] += update[i:]
            mouse_session.expectation_right = np.clip(mouse_session.expectation_right, 0.0, 1.0)
        else:
            mouse_session.expectation_left[i:, 0] += update[i:]
            mouse_session.expectation_left = np.clip(mouse_session.expectation_left, 0.0, 1.0)

    if reward_t == 0 and mouse_session.lick[i - 1, 0] == 1:
        update = (
            rpe_t
            * mouse.exp_update_no_reward[1]
            * np.exp(-np.maximum(time_vector - t, 0.0) / mouse.exp_update_no_reward[0])
        )
        if ref_side_t == 1:
            mouse_session.expectation_right[i:, 0] += update[i:]
            mouse_session.expectation_right = np.clip(mouse_session.expectation_right, 0.0, 1.0)
        elif ref_side_t == -1:
            mouse_session.expectation_left[i:, 0] += update[i:]
            mouse_session.expectation_left = np.clip(mouse_session.expectation_left, 0.0, 1.0)
        mouse_session.non_rew_lick_cnt += 1
    elif reward_t > 0 and mouse_session.lick[i - 1, 0] == 1:
        mouse_session.non_rew_lick_cnt = 0

    if stim_signed_t != 0:
        stim_tau = mouse.exp_update_stim[0]
        update_elig = abs(stim_signed_t) * np.exp(-np.maximum(time_vector - t, 0.0) / mouse.tau_eligibility)

        if stim_signed_t > 0:
            update_exp = abs(stim_signed_t) * mouse.stim_gain_right * np.exp(-np.maximum(time_vector - t, 0.0) / stim_tau)
            mouse_session.expectation_right[i:, 0] += update_exp[i:]
            mouse_session.expectation_right = np.clip(mouse_session.expectation_right, 0.0, 1.0)
            mouse_session.eligibility_right[i:, 0] += update_elig[i:]
            mouse_session.eligibility_right = np.clip(mouse_session.eligibility_right, 0, 1)

            # generalisation sensorielle precoce : un stim droite pousse aussi (faiblement) le canal gauche
            update_cross = abs(stim_signed_t) * mouse.cross_stim_gain * np.exp(-np.maximum(time_vector - t, 0.0) / stim_tau)
            mouse_session.expectation_left[i:, 0] += update_cross[i:]
            mouse_session.expectation_left = np.clip(mouse_session.expectation_left, 0.0, 1.0)
        else:
            update_exp = abs(stim_signed_t) * mouse.stim_gain_left * np.exp(-np.maximum(time_vector - t, 0.0) / stim_tau)
            mouse_session.expectation_left[i:, 0] += update_exp[i:]
            mouse_session.expectation_left = np.clip(mouse_session.expectation_left, 0.0, 1.0)
            mouse_session.eligibility_left[i:, 0] += update_elig[i:]
            mouse_session.eligibility_left = np.clip(mouse_session.eligibility_left, 0, 1)

            update_cross = abs(stim_signed_t) * mouse.cross_stim_gain * np.exp(-np.maximum(time_vector - t, 0.0) / stim_tau)
            mouse_session.expectation_right[i:, 0] += update_cross[i:]
            mouse_session.expectation_right = np.clip(mouse_session.expectation_right, 0.0, 1.0)

    if mouse_session.non_rew_lick_cnt >= mouse.learning_nonrew_lick[0]:
        old_tau, gain = mouse.exp_update_no_reward
        mouse.exp_update_no_reward = (old_tau + mouse.learning_nonrew_lick[1], gain)
        mouse_session.non_rew_lick_cnt = 0

    if reward_t > 0 and ref_side_t == 1:
        mouse.stim_gain_right = min(mouse.stim_gain_max, max(0.0, mouse.stim_gain_right + rpe_t * mouse.learning_stim * mouse_session.eligibility_right[i, 0]))
    elif reward_t > 0 and ref_side_t == -1:
        mouse.stim_gain_left = min(mouse.stim_gain_max, max(0.0, mouse.stim_gain_left + rpe_t * mouse.learning_stim * mouse_session.eligibility_left[i, 0]))

    return mouse, mouse_session


def function_fl_dualstim_session(
    session_param: DualFreeLickingSessionParams,
    fl_session: DualFreeLickingSessionState,
    session_info: SessionBaseInfo,
    lick: int,
    lick_side: int,
    i: int,
) -> tuple[float, int, DualFreeLickingSessionState]:
    """
    Comme function_fl_session, mais avec deux forced rewards (un par cote) pour que
    la souris connaisse deja les deux spouts avant le WDT. Un lick spontane n'est
    recompense qu'une fois les DEUX forced rewards passes (meme logique que le
    modele a un seul stimulus, etendue aux deux cotes).
    """
    t = i * session_info.resolution
    reward = 0.0
    reward_side = 0

    if session_param.forced_reward_right[0] == 1 and np.isclose(t, session_param.forced_reward_right[1], atol=1e-6):
        reward = session_param.reward_size
        reward_side = 1
        fl_session.last_reward_time = t

    if session_param.forced_reward_left[0] == 1 and np.isclose(t, session_param.forced_reward_left[1], atol=1e-6):
        reward = session_param.reward_size
        reward_side = -1
        fl_session.last_reward_time = t

    reward_eligible = (
        (session_param.forced_reward_right[0] == 0 or t >= session_param.forced_reward_right[1])
        and (session_param.forced_reward_left[0] == 0 or t >= session_param.forced_reward_left[1])
    )

    if lick == 1:
        if reward_eligible and (t - fl_session.last_lick_time) > fl_session.no_lick_wind:
            fl_session.no_lick_wind = sample_uniform_range(*session_param.no_lick_wind)
            if random.random() <= session_param.reward_prob:
                reward = session_param.reward_size
                reward_side = lick_side
                fl_session.last_reward_time = t
        fl_session.last_lick_time = t

    fl_session.reward[i, 0] = reward
    fl_session.reward_side[i, 0] = reward_side

    return reward, reward_side, fl_session


def function_wdt_dualstim_session(
    session_param: DualWDTSessionParams,
    wdt_session: DualWDTSessionState,
    session_info: SessionBaseInfo,
    lick: int,
    lick_side: int,
    i: int,
) -> tuple[float, int, float, DualWDTSessionState, bool]:
    """
    Comme function_wdt_session, avec un cote correct tire par trial (kind 1=droite,
    2=gauche, 0=catch). Un lick sur le mauvais cote pendant la fenetre de reponse
    n'est jamais recompense (mismatch), exactement comme un lick non recompense
    classique — pas de consequence a part.
    """
    t = i * session_info.resolution
    reward = 0.0
    reward_side = 0
    stim_signed = 0.0
    new_trial = False

    if (
        (t - wdt_session.last_lick_time) > wdt_session.no_lick_wind
        and (t - wdt_session.last_trial_time) > wdt_session.iti
    ):
        wdt_session.trial_times.append(t)
        wdt_session.last_trial_time = t
        wdt_session.no_lick_wind = sample_uniform_range(*session_param.no_lick_wind)
        wdt_session.iti = sample_uniform_range(*session_param.iti)
        new_trial = True

        kind = random.choice(session_param.trial_kinds)
        wdt_session.last_trial_kind = int(kind)

        if kind == 1:
            stim_signed = session_param.stim_amp
            wdt_session.last_trial_correct_side = 1
            wdt_session.last_stim_time = t
        elif kind == 2:
            stim_signed = -session_param.stim_amp
            wdt_session.last_trial_correct_side = -1
            wdt_session.last_stim_time = t
        else:
            wdt_session.last_trial_correct_side = 0

        wdt_session.trial_correct_side.append(wdt_session.last_trial_correct_side)

        if kind in (1, 2):
            window_len = int(round(session_param.response_wind / session_info.resolution))
            end_idx = min(i + window_len, session_info.number_bin)
            wdt_session.reward_window[i + 1:end_idx, 0] = 1

    if lick == 1:
        wdt_session.last_lick_time = t
        if wdt_session.reward_window[i, 0] > 0.5 and (t - wdt_session.last_reward_time) > 2:
            if lick_side != 0 and lick_side == wdt_session.last_trial_correct_side:
                if random.random() <= session_param.reward_prob:
                    reward = session_param.reward_size
                    reward_side = lick_side
                    wdt_session.last_reward_time = t

    wdt_session.reward[i, 0] = reward
    wdt_session.reward_side[i, 0] = reward_side
    if stim_signed != 0:
        wdt_session.stim_signed[i, 0] = stim_signed
        wdt_session.stim1[i, 0] = abs(stim_signed)

    return reward, reward_side, stim_signed, wdt_session, new_trial


def function_performance_wdt_dual(
    wdt_params: DualWDTSessionParams,
    session_info: SessionBaseInfo,
    wdt_session: DualWDTSessionState,
    mouse_session: DualMouseSessionState,
) -> np.ndarray:
    """
    Comme function_performance_wdt, avec une categorie de resultat en plus :
    outcome 0=Miss, 1=Hit, 2=Correct Rejection, 3=False Alarm, 4=Mismatch
    (elle a leche, mais du mauvais cote).
    Colonnes : [t, correct_side, lick_detected, latency, reward_detected, outcome, lick_side_detected]
    """
    sr = int(round(1 / session_info.resolution))
    trials = [t for t in wdt_session.trial_times if t < (session_info.duration * 60 - wdt_params.response_wind)]

    performance = []
    for idx, t in enumerate(trials):
        pt1 = int(round(t * sr))
        pt2 = min(pt1 + int(round(wdt_params.response_wind * sr)), session_info.number_bin)

        correct_side = wdt_session.trial_correct_side[idx]
        lick_segment = mouse_session.lick[pt1:pt2, 0]
        lick_side_segment = mouse_session.lick_side[pt1:pt2, 0]
        reward_segment = wdt_session.reward[pt1:pt2, 0]

        lick_detected = np.any(lick_segment)
        reward_detected = np.any(reward_segment)

        if lick_detected:
            first_idx = int(np.argmax(lick_segment))
            lick_side_detected = int(lick_side_segment[first_idx])
            latency = first_idx / sr
        else:
            lick_side_detected = 0
            latency = np.nan

        if correct_side != 0 and not lick_detected:
            outcome = 0  # Miss
        elif correct_side != 0 and lick_detected and lick_side_detected == correct_side:
            outcome = 1  # Hit
        elif correct_side != 0 and lick_detected:
            outcome = 4  # Mismatch : elle a leche, mauvais cote
        elif correct_side == 0 and not lick_detected:
            outcome = 2  # Correct Rejection
        else:
            outcome = 3  # False Alarm

        performance.append([t, correct_side, int(lick_detected), latency, int(reward_detected), outcome, lick_side_detected])

    return np.array(performance)


def _init_expect_dual_channels(mouse_session: DualMouseSessionState, prev_session) -> tuple[float, float]:
    """
    Generalisation directe de (reward_sum/lick_sum) du modele a un seul stimulus, calculee
    separement par canal : chaque reward compte pour le cote ou il a ete delivre uniquement
    (y compris les forced rewards, un par cote en FL). Retourne (init_droite, init_gauche).
    """
    lick_sum = float(np.sum(mouse_session.lick))
    if lick_sum <= 0:
        return 0.0, 0.0
    reward_side = prev_session.reward_side.astype(float)
    reward = prev_session.reward
    right_sum = float(np.sum(np.where(reward_side == 1, reward, 0.0)))
    left_sum = float(np.sum(np.where(reward_side == -1, reward, 0.0)))
    return right_sum / lick_sum, left_sum / lick_sum


def run_FL1_dualstim(mouse: Mouse, session_info: SessionBaseInfo, config: SimConfig) -> FLSessionResult:
    """Free Licking — Session 1, en dual stim : forced reward a droite puis a gauche."""
    fl_params = DualFreeLickingSessionParams(
        forced_reward_right=(1, config.FL_FORCED_REWARD_RIGHT_T),
        forced_reward_left=(1, config.FL_FORCED_REWARD_LEFT_T),
    )
    fl_session = DualFreeLickingSessionState()
    fl_session.initialize(session_info.number_bin, fl_params.no_lick_wind)
    mouse_session = DualMouseSessionState()
    mouse_session.initialize(session_info.number_bin, mouse.motivation[0])

    for i in range(1, session_info.number_bin):
        reward_t, reward_side_t, fl_session = function_fl_dualstim_session(
            fl_params, fl_session, session_info,
            int(mouse_session.lick[i - 1, 0]), int(mouse_session.lick_side[i - 1, 0]), i,
        )
        _, _, mouse_session = function_mouse_lick_dual(mouse, mouse_session, i)
        mouse, mouse_session = function_update_mouse_state_dual(
            mouse, mouse_session, session_info, i, 0.0, reward_t, reward_side_t
        )

    return FLSessionResult(session=fl_session, mouse_session=mouse_session)


def run_FL2_dualstim(mouse: Mouse, session_info: SessionBaseInfo, config: SimConfig, log_fl1: FLSessionResult) -> FLSessionResult:
    """Free Licking — Session 2, en dual stim : plus de forced reward, les deux cotes restent actifs."""
    fl_params = DualFreeLickingSessionParams(forced_reward_right=(0, 0), forced_reward_left=(0, 0))

    init_expect_right, init_expect_left = _init_expect_dual_channels(log_fl1.mouse_session, log_fl1.session)
    init_uncert = float(log_fl1.mouse_session.uncertainty[-1, 0])

    fl_session = DualFreeLickingSessionState()
    fl_session.initialize(session_info.number_bin, fl_params.no_lick_wind)
    mouse_session = DualMouseSessionState()
    mouse_session.initialize(session_info.number_bin, mouse.motivation[0], init_expect_right, init_expect_left, init_uncert)

    for i in range(1, session_info.number_bin):
        reward_t, reward_side_t, fl_session = function_fl_dualstim_session(
            fl_params, fl_session, session_info,
            int(mouse_session.lick[i - 1, 0]), int(mouse_session.lick_side[i - 1, 0]), i,
        )
        _, _, mouse_session = function_mouse_lick_dual(mouse, mouse_session, i)
        mouse, mouse_session = function_update_mouse_state_dual(
            mouse, mouse_session, session_info, i, 0.0, reward_t, reward_side_t
        )

    return FLSessionResult(session=fl_session, mouse_session=mouse_session)


def run_wdt_dualstim_session(
    mouse: Mouse,
    session_info: SessionBaseInfo,
    config: SimConfig,
    wdt_params: DualWDTSessionParams,
    prev_mouse_session: DualMouseSessionState,
    prev_session: DualFreeLickingSessionState,
    label: str,
) -> WDTRunResult:
    """Corps d'une session WDT en dual stim, reutilisee pour WDT1..N et pour WDT_TEST."""
    init_expect_right, init_expect_left = _init_expect_dual_channels(prev_mouse_session, prev_session)
    init_uncert = float(prev_mouse_session.uncertainty[-1, 0])

    wdt_session = DualWDTSessionState()
    wdt_session.initialize(session_info.number_bin, wdt_params.no_lick_wind, wdt_params.iti)
    mouse_session = DualMouseSessionState()
    mouse_session.initialize(session_info.number_bin, mouse.motivation[0], init_expect_right, init_expect_left, init_uncert)

    for i in range(1, session_info.number_bin):
        reward_t, reward_side_t, stim_signed_t, wdt_session, _ = function_wdt_dualstim_session(
            wdt_params, wdt_session, session_info,
            int(mouse_session.lick[i - 1, 0]), int(mouse_session.lick_side[i - 1, 0]), i,
        )
        _, _, mouse_session = function_mouse_lick_dual(
            mouse, mouse_session, i,
            threshold=mouse.lick_thrs_wdt, decision_slope=mouse.decision_slope_wdt,
            v_right=config.V_RIGHT, v_left=config.V_LEFT, c_right=config.C_RIGHT, c_left=config.C_LEFT,
        )
        mouse, mouse_session = function_update_mouse_state_dual(
            mouse, mouse_session, session_info, i, stim_signed_t, reward_t, reward_side_t
        )

    perf = function_performance_wdt_dual(wdt_params, session_info, wdt_session, mouse_session)

    return WDTRunResult(label=label, session=wdt_session, mouse_session=mouse_session, perf=perf, vc_log={})


def run_all_wdt_dualstim(mouse: Mouse, session_info: SessionBaseInfo, config: SimConfig, log_fl2: FLSessionResult) -> WDTBundle:
    """Enchaine WDT1..WDT{config.NUM_WDT} en dual stim."""
    bundle = WDTBundle()
    prev_mouse_session = log_fl2.mouse_session
    prev_session = log_fl2.session

    for s_idx in range(1, config.NUM_WDT + 1):
        wdt_params = DualWDTSessionParams(trial_kinds=config.DUAL_TRIAL_KINDS[:], stim_amp=config.DUAL_STIM_AMP)
        label = f"WDT{s_idx}"
        result = run_wdt_dualstim_session(mouse, session_info, config, wdt_params, prev_mouse_session, prev_session, label)

        hr, hr_right, hr_left, mismatch, fa, _ = session_rates_dual(result.perf)
        print(f"Hit Rate {label} : {hr:.2f} (droite {hr_right:.2f} / gauche {hr_left:.2f})")
        print(f"Mismatch {label} : {mismatch:.2f}")
        print(f"FA  Rate {label} : {fa:.2f}")

        bundle.results.append(result)
        bundle.labels.append(label)

        prev_mouse_session = result.mouse_session
        prev_session = result.session

    return bundle


def run_wdt_test_dualstim(mouse: Mouse, session_info: SessionBaseInfo, config: SimConfig, wdt_bundle: WDTBundle) -> WDTRunResult:
    """WDT_TEST en dual stim, heritee de la derniere session WDT."""
    wdt_test_params = DualWDTSessionParams(trial_kinds=config.DUAL_TRIAL_KINDS[:], stim_amp=config.DUAL_STIM_AMP)
    last = wdt_bundle.results[-1]
    return run_wdt_dualstim_session(
        mouse, session_info, config, wdt_test_params, last.mouse_session, last.session, "WDT_TEST",
    )


def plot_all_results_dualstim(
    session_info: SessionBaseInfo,
    mouse: Mouse,
    config: SimConfig,
    log_fl1: FLSessionResult,
    log_fl2: FLSessionResult,
    wdt_bundle: WDTBundle,
    wdt_test: WDTRunResult,
) -> None:
    """Version dual stim de plot_all_results : memes fonctions de plot, appelees avec dual_stim=True."""

    if config.PLOT_TRACES:
        plot_traces(session_info.time_vector, log_fl1.mouse_session, log_fl1.session.reward,
                    stim_array=None, title="Free Licking — Session 1 (dual stim)",
                    save_name="free_licking_1", threshold=mouse.lick_thrs, dual_stim=True)
        plot_traces(session_info.time_vector, log_fl2.mouse_session, log_fl2.session.reward,
                    stim_array=None, title="Free Licking — Session 2 (dual stim)",
                    save_name="free_licking_2", threshold=mouse.lick_thrs, dual_stim=True)
        for lbl, result in zip(wdt_bundle.labels, wdt_bundle.results):
            plot_traces(session_info.time_vector, result.mouse_session, result.session.reward,
                        stim_array=result.session.stim1, title=f"Two-Choice Detection Task — {lbl}",
                        save_name=lbl.lower(), threshold=mouse.lick_thrs_wdt, dual_stim=True)
        plot_traces(session_info.time_vector, wdt_test.mouse_session, wdt_test.session.reward,
                    stim_array=wdt_test.session.stim1, title="WDT Test (dual stim)",
                    save_name="wdt_test", threshold=mouse.lick_thrs_wdt, dual_stim=True)

    if config.PLOT_BLOCK_HRFA:
        _wdt_params = DualWDTSessionParams()
        for lbl, result in zip(wdt_bundle.labels, wdt_bundle.results):
            plot_wdt_block_rates(result.perf, _wdt_params, max_trials=config.MAX_TRIALS_BLOCKS,
                                 title=f"{lbl} — HR/Mismatch/FA by block", save_name=lbl.lower(), dual_stim=True)

    if config.PLOT_SESSIONS_COMPARISON:
        wdt_perfs = [r.perf for r in wdt_bundle.results]
        plot_session_rates(wdt_perfs, wdt_bundle.labels, title="Learning — HR, Mismatch, FA & d′ (WDT1..WDT10)",
                            save_name="hr_fa_dprime_all_sessions", dual_stim=True)

    if config.PLOT_SINGLE_SESSION_ALL:
        for lbl, result in zip(wdt_bundle.labels, wdt_bundle.results):
            plot_single_session_rates(result.perf, title=f"{lbl} — HR vs Mismatch vs FA", save_name=lbl.lower(), dual_stim=True)
