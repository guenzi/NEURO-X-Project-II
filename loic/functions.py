import numpy as np
import random
from typing import Any, Optional, Sequence, Set, List, Dict


from models import (
    Mouse,
    SessionBaseInfo,
    FreeLickingSessionParams,
    MouseSessionState,
    WDTSesssionParams,
    WDTSesssionState,
    FreeLickingSessionState,
)

from plotting import plot_population_hr_fa




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

    # --- FL1 ---
    flp = FreeLickingSessionParams()
    fls = FreeLickingSessionState(); fls.initialize(si.number_bin, flp.no_lick_wind)
    ms  = MouseSessionState();       ms.initialize(si.number_bin, mouse.motivation[0])
    for i in range(1, si.number_bin):
        r, s, fls = function_fl_session(flp, fls, si, int(ms.lick[i-1, 0]), i)
        _, ms = function_mouse_lick(mouse, ms, i)
        mouse, ms = function_update_mouse_state(mouse, ms, si, i, s, r)

    # --- FL2 ---
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

    # --- WDT1..WDT{num_wdt_sessions} ---
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

    # --- WDT_TEST optionnel ---
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



# === Noise modulation helpers (sigmoïde sur |RPE| aux licks récents) ===
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
