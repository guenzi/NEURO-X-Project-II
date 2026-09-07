import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
from dataclasses import replace
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
    plot_delearning_diagnostics,
    plot_population_learning_curves,
    plot_population_session_progression,
    plot_wdt_block_rates_wa,
    plot_session_rates_wa,
    plot_single_session_rates_wa,
    plot_stim_gains,
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
    Decision function D, avec E = E_go - E_nogo (architecture Go/No-Go, D1R/D2R) :
        D = [E/(1+e^{A_D.(U-U0)}) + Nd/(1+e^{-A_D.(U-U0)})] . [V.M - C]
        Mouse licks if D >= threshold.

    Le signe est bien +A_D.(U-U0) sous l'exponentielle de E (poids qui decroit avec U)
    et -A_D.(U-U0) sous celle de Nd (poids qui croit avec U) : quand l'incertitude U
    augmente au-dela du point neutre U0, le poids sur le bruit Nd augmente (exploration)
    et le poids sur l'expectation E diminue (moins d'exploitation). En dessous de U0
    (souris confiante / naive sans surprise), E domine et le bruit est quasi-nul.

    E_go garde exactement sa dynamique existante (monte a la recompense, redescend a
    chaque lick non recompense) ; E_nogo est une voie separee, plus lente et plus
    persistante, qui s'y oppose au moment de la decision (voir function_update_mouse_state).
    """
    lick = 0

    motivation_t = mouse_session.motivation[i - 1, 0]
    expectation_t = mouse_session.expectation[i - 1, 0] - mouse_session.expectation_nogo[i - 1, 0]
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
    reward_t: float,
    in_trial_window_prev: int = 0,
    reward_denied_by_prob: bool = False,
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

    # E_nogo (voie No-Go, architecture Go/No-Go) : s'ajoute a la mise a jour de E_go
    # ci-dessous, ne la remplace pas. E_go garde exactement sa dynamique rapide ; E_nogo
    # est un signal supplementaire, plus lent et plus persistant, qui vient s'y opposer
    # au moment de la decision (E_go - E_nogo, voir function_mouse_lick). Mise a jour
    # comme motivation/uncertainty (recurrence a chaque bin), pas comme expectation
    # (injection d'un noyau futur) : ce paradigme genere des dizaines de bouts de
    # lechage non recompenses par seconde une fois la souris confiante, une injection
    # future ferait saturer E_nogo en quelques secondes. Le kick est saturant
    # (gain*(1-E_nogo), pas +gain) donc borne par construction.
    tau_nogo, gain_nogo = mouse.exp_update_nogo
    decay_nogo = np.exp(-session_info.resolution / tau_nogo)
    e_nogo_new = mouse_session.expectation_nogo[i - 1, 0] * decay_nogo

    is_new_unrewarded_bout = (
        reward_t == 0
        and mouse_session.lick[i - 1, 0] == 1
        and (i < 2 or mouse_session.lick[i - 2, 0] == 0)
    )
    if is_new_unrewarded_bout:
        e_nogo_new = e_nogo_new + gain_nogo * (1.0 - e_nogo_new)

    if reward_t > 0:
        # "Soulagement" : une recompense confirmee efface une partie de la mefiance deja
        # accumulee (reacquisition rapide apres extinction, cf. litterature comportementale).
        e_nogo_new = e_nogo_new * (1.0 - mouse.nogo_relief)

    mouse_session.expectation_nogo[i, 0] = float(np.clip(e_nogo_new, 0.0, 1.0))

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
        # Sensibilisation No-Go : ne compte que les lechages DANS un essai (fenetre de
        # reponse a un trial, ex. une FA sur un catch) et seulement au debut d'un bout
        # (is_new_unrewarded_bout), pas le lechage spontane entre les essais ni chaque
        # bin d'un meme bout — sinon le compteur explose des le premier bout de lechage.
        if is_new_unrewarded_bout and in_trial_window_prev:
            mouse_session.nogo_streak_cnt += 1
    elif reward_t > 0 and mouse_session.lick[i - 1, 0] == 1:
        mouse_session.non_rew_lick_cnt = 0
        mouse_session.nogo_streak_cnt = 0

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

    # Sensibilisation No-Go : apres une serie de lechages non recompenses consecutifs (DANS un
    # essai), le tau_nogo peut grandir (desactive par standardisation, NOGO_TAU_GROWTH=0) et/ou
    # le gain No-Go apprend a la place — meme regle delta que le gain Go (learning_stim), sur
    # -rpe_t (plus la non-recompense est surprenante, plus le gain grandit). Meme cadence que
    # l'ancienne sensibilisation du tau pour ne pas contaminer l'apprentissage normal.
    if mouse_session.nogo_streak_cnt >= mouse.nogo_streak_incr[0]:
        old_tau_nogo, gain_nogo_cur = mouse.exp_update_nogo
        new_tau_nogo = min(old_tau_nogo + mouse.nogo_streak_incr[1], mouse.nogo_tau_max)
        new_gain_nogo = gain_nogo_cur
        if mouse.learning_nogo_gain > 0:
            new_gain_nogo = float(np.clip(
                gain_nogo_cur + (-rpe_t) * mouse.learning_nogo_gain, 0.0, mouse.nogo_gain_max
            ))
        mouse.exp_update_nogo = (new_tau_nogo, new_gain_nogo)
        mouse_session.nogo_streak_cnt = 0

    if reward_t > 0:
        new_gain = mouse.exp_update_stim[1] + rpe_t * mouse.learning_stim * mouse_session.eligibility[i, 0]
        mouse.exp_update_stim = (mouse.exp_update_stim[0], new_gain)
    elif mouse.stim_gain_noreward_active and is_new_unrewarded_bout and reward_denied_by_prob:
        # Baisse symetrique du gain de stimulus (mono), meme principe que le correctif dual
        # (stim_gain_right/left) : UNIQUEMENT quand ce lick a atteint le tirage de probabilite
        # de recompense (dans la fenetre, cooldown ecoule : "aurait du" etre recompense) et que
        # ce tirage a echoue. reward_prob=1.0 (defaut hors desapprentissage) rend ca
        # structurellement impossible, donc inerte en entrainement normal.
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
) -> tuple[float, float, WDTSesssionState, bool, bool]:
    t = i * session_info.resolution
    reward = 0.0
    stim = 0.0
    new_trial = False
    reward_denied_by_prob = False


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
            else:
                reward_denied_by_prob = True

    wdt_session.reward[i, 0] = reward
    wdt_session.stim1[i, 0] = stim

    return reward, stim, wdt_session, new_trial, reward_denied_by_prob


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
            r, s, ws, _, rdp = function_wdt_session(wp, ws, si, int(ms.lick[i-1, 0]), i)
            _, ms = function_mouse_lick(mouse, ms, i)
            mouse, ms = function_update_mouse_state(
                mouse, ms, si, i, s, r,
                in_trial_window_prev=int(ws.reward_window[i - 1, 0] > 0.5),
                reward_denied_by_prob=rdp,
            )

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
            r, s, ws_test, _, rdp = function_wdt_session(wp_test, ws_test, si, int(ms_test.lick[i-1, 0]), i)
            _, ms_test = function_mouse_lick(mouse, ms_test, i)
            mouse, ms_test = function_update_mouse_state(
                mouse, ms_test, si, i, s, r,
                in_trial_window_prev=int(ws_test.reward_window[i - 1, 0] > 0.5),
                reward_denied_by_prob=rdp,
            )

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

def _build_mouse(config: SimConfig, learning_stim: Optional[float] = None, noise: Optional[tuple] = None) -> Mouse:
    """Construit un Mouse a partir de config, avec overrides optionnels (utilise par run_population
    pour faire varier learning_stim / noise d'une souris a l'autre sans dupliquer cette liste)."""
    kwargs = dict(
        reward_value=config.REWARD_VALUE,
        cost=config.COST,
        lick_thrs=config.LICK_THRS_FL,
        lick_thrs_wdt=config.LICK_THRS_WDT,
        exp_update_nogo=(config.NOGO_TAU, config.NOGO_GAIN),
        exp_update_nogo_right=(config.NOGO_TAU, config.NOGO_GAIN),
        exp_update_nogo_left=(config.NOGO_TAU, config.NOGO_GAIN),
        nogo_relief=config.NOGO_RELIEF,
        nogo_streak_incr=(config.NOGO_STREAK_THRESHOLD, config.NOGO_TAU_GROWTH),
        nogo_tau_max=config.NOGO_TAU_MAX,
        learning_nogo_gain=config.LEARNING_NOGO_GAIN,
        nogo_gain_max=config.NOGO_GAIN_MAX,
        stim_gain_noreward_active=config.STIM_GAIN_NOREWARD_ACTIVE,
    )
    if learning_stim is not None:
        kwargs["learning_stim"] = learning_stim
    if noise is not None:
        kwargs["noise"] = noise
    return Mouse(**kwargs)


def initialization(config: SimConfig) -> tuple[SessionBaseInfo, Mouse]:
    """Cree le dossier de resultats (nomme mono/dual/whiskAud, + delearning si actif) et l'etat global (session_info, mouse)."""
    side_tag = {1: "droite", -1: "gauche"}.get(config.DELEARNING_SIDE)
    mode_tag = "whiskAud" if config.WHISKER_AUD_STIM else ("dual" if config.dual_stim else "mono")
    suffix = (
        mode_tag
        + (f"_delearnFrom{config.DELEARNING_FROM_SESSION}" if config.DELEARNING_FROM_SESSION is not None else "")
        + (f"_{side_tag}" if side_tag and config.DELEARNING_FROM_SESSION is not None else "")
        + (f"_until{config.DELEARNING_UNTIL_SESSION}" if config.DELEARNING_UNTIL_SESSION is not None else "")
    )
    init_results_dir(suffix=suffix)

    session_info = SessionBaseInfo()
    mouse = _build_mouse(config)
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
    init_nogo = float(log_fl1.mouse_session.expectation_nogo[-1, 0])

    fl_session = FreeLickingSessionState()
    fl_session.initialize(session_info.number_bin, fl_params.no_lick_wind)
    mouse_session = MouseSessionState()
    mouse_session.initialize(session_info.number_bin, mouse.motivation[0], init_expect, init_uncert, init_nogo)

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
    init_nogo = float(prev_mouse_session.expectation_nogo[-1, 0])

    wdt_session = WDTSesssionState()
    wdt_session.initialize(session_info.number_bin, wdt_params.no_lick_wind, wdt_params.iti)
    mouse_session = MouseSessionState()
    mouse_session.initialize(session_info.number_bin, mouse.motivation[0], init_expect, init_uncert, init_nogo)

    buf_wdt, cnt_wdt = new_rpe_buffer(config.SIG_AVG_LICKS)
    buf_v, buf_c = deque(maxlen=config.VC_BUFFER_SIZE), deque(maxlen=config.VC_BUFFER_SIZE)
    vc_log = {"trial_times": [], "v": [], "c": [], "expected_v": [], "expected_c": []}

    for i in range(1, session_info.number_bin):
        reward_t, stim_t, wdt_session, new_trial, reward_denied_by_prob = function_wdt_session(
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
        mouse, mouse_session = function_update_mouse_state(
            mouse, mouse_session, session_info, i, stim_t, reward_t,
            in_trial_window_prev=int(wdt_session.reward_window[i - 1, 0] > 0.5),
            reward_denied_by_prob=reward_denied_by_prob,
        )

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
        if (
            config.DELEARNING_FROM_SESSION is not None
            and s_idx >= config.DELEARNING_FROM_SESSION
            and (config.DELEARNING_UNTIL_SESSION is None or s_idx < config.DELEARNING_UNTIL_SESSION)
        ):
            # Desapprentissage integre au pipeline normal : plus aucune recompense a partir
            # de cette session (et jusqu'a DELEARNING_UNTIL_SESSION si renseigne, sinon jusqu'a
            # la fin), les stimuli/trials continuent normalement.
            wdt_params.reward_prob = 0.0

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

        # Meme courbe, mais recalculee a Motivation=1.0 tout du long (annule l'effet de
        # satiete intra-session sur V*M-C) pour voir la "vraie" performance (ce que E encode).
        wdt_perfs_norm = [_normalize_perf_for_motivation(r, False, config) for r in wdt_bundle.results]
        plot_session_rates(wdt_perfs_norm, wdt_bundle.labels,
                            title="Learning — HR, FA & d′ (WDT1..WDT10) — normalisé Motivation=1.0",
                            save_name="hr_fa_dprime_all_sessions_normalized_motivation")

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

    if config.DELEARNING_FROM_SESSION is not None:
        plot_delearning_diagnostics(
            session_info,
            [r.mouse_session for r in wdt_bundle.results],
            wdt_bundle.labels,
            config.DELEARNING_FROM_SESSION,
            perfs=[r.perf for r in wdt_bundle.results],
            dual_stim=False,
            delearning_until_session=config.DELEARNING_UNTIL_SESSION,
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

        ("MOUSE — ARCHITECTURE GO/NO-GO (E_go - E_nogo, base)", None),
        ("Tau (no-go, s)", config.NOGO_TAU),
        ("Gain (no-go)", config.NOGO_GAIN),
        ("Relief (fraction effacee par reward)", config.NOGO_RELIEF),
        ("Sensibilisation : seuil (lechages non recompenses consecutifs)", config.NOGO_STREAK_THRESHOLD),
        ("Sensibilisation : increment de tau", config.NOGO_TAU_GROWTH),
        ("Sensibilisation : tau max", config.NOGO_TAU_MAX),
        ("Desapprentissage integre a partir de la session", config.DELEARNING_FROM_SESSION or "n/a"),
        ("Desapprentissage cote cible (dual)", {1: "droite", -1: "gauche"}.get(config.DELEARNING_SIDE, "les deux")),
        ("Reapprentissage a partir de la session", config.DELEARNING_UNTIL_SESSION or "n/a"),

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


# Dual whisker/auditif : reprend le paradigme historique (switch de contingence WDT->AUD)
# mais avec UNE SEULE Expectation partagee (comme le mono), alimentee par deux canaux de
# stimulus, chacun avec son propre gain Go et son propre gain No-Go (desapprentissage
# specifique au stimulus). Rien ci-dessus n'est modifie ; ces fonctions sont utilisees a la
# place des precedentes uniquement quand config.WHISKER_AUD_STIM=True (voir main.py).
# function_mouse_lick et function_fl_session (mono, inchangees) sont reutilisees telles
# quelles : la decision et le Free Licking ne dependent jamais du nombre de canaux de
# stimulus, seule la session WDT/AUD en a un.

def function_update_mouse_state_wa(
    mouse: Mouse,
    mouse_session: MouseSessionState,
    session_info: SessionBaseInfo,
    i: int,
    stim1_t: float,
    stim2_t: float,
    reward_t: float,
    in_trial_window_prev: int = 0,
    reward_denied_stim: int = 0,
) -> tuple[Mouse, MouseSessionState]:
    """
    Comme function_update_mouse_state (mono), etendue a DEUX canaux de stimulus qui
    alimentent la MEME Expectation partagee. Motivation, RPE, Uncertainty, E_nogo generale
    et la mise a jour reward/no-reward de `expectation` sont identiques au mono (memes
    lignes) ; seule la section stimulus change (stim1_t/stim2_t au lieu de stim_t).
    """
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

    mouse_session.uncertainty[i, 0] = mouse_session.uncertainty[i - 1, 0]
    if mouse_session.lick[i - 1, 0] == 1 or reward_t > 0:
        u_prev = mouse_session.uncertainty[i - 1, 0]
        u_new = u_prev + mouse.uncertainty_gain * (2.0 * abs(rpe_t) - 1.0) ** 3
        mouse_session.uncertainty[i, 0] = float(np.clip(u_new, 0.0, mouse.uncertainty_max))

    # E_nogo generale : identique au mono, capte l'impulsivite independamment du stimulus.
    tau_nogo, gain_nogo = mouse.exp_update_nogo
    decay = np.exp(-session_info.resolution / tau_nogo)
    e_nogo_new = mouse_session.expectation_nogo[i - 1, 0] * decay

    is_new_unrewarded_bout = (
        reward_t == 0
        and mouse_session.lick[i - 1, 0] == 1
        and (i < 2 or mouse_session.lick[i - 2, 0] == 0)
    )
    if is_new_unrewarded_bout:
        e_nogo_new = e_nogo_new + gain_nogo * (1.0 - e_nogo_new)
    if reward_t > 0:
        e_nogo_new = e_nogo_new * (1.0 - mouse.nogo_relief)
    mouse_session.expectation_nogo[i, 0] = float(np.clip(e_nogo_new, 0.0, 1.0))

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
        if is_new_unrewarded_bout and in_trial_window_prev:
            mouse_session.nogo_streak_cnt += 1
    elif reward_t > 0 and mouse_session.lick[i - 1, 0] == 1:
        mouse_session.non_rew_lick_cnt = 0
        mouse_session.nogo_streak_cnt = 0

    # --- Contribution des DEUX stimuli a l'Expectation partagee -----------------------------
    if stim1_t > 0.0 or stim2_t > 0.0:
        tau1, net_gain1 = mouse.exp_update_stim1_wa
        tau2, net_gain2 = mouse.exp_update_stim2_wa

        if stim1_t > 0.0:
            update_exp1 = stim1_t * net_gain1 * np.exp(-np.maximum(time_vector - t, 0.0) / tau1)
            mouse_session.expectation[i:, 0] += update_exp1[i:]
            update_elig1 = stim1_t * np.exp(-np.maximum(time_vector - t, 0.0) / mouse.tau_eligibility)
            mouse_session.eligibility1[i:, 0] += update_elig1[i:]
            mouse_session.eligibility1 = np.clip(mouse_session.eligibility1, 0, 1)

        if stim2_t > 0.0:
            update_exp2 = stim2_t * net_gain2 * np.exp(-np.maximum(time_vector - t, 0.0) / tau2)
            mouse_session.expectation[i:, 0] += update_exp2[i:]
            update_elig2 = stim2_t * np.exp(-np.maximum(time_vector - t, 0.0) / mouse.tau_eligibility)
            mouse_session.eligibility2[i:, 0] += update_elig2[i:]
            mouse_session.eligibility2 = np.clip(mouse_session.eligibility2, 0, 1)

        mouse_session.expectation = np.clip(mouse_session.expectation, 0, 1)

    if mouse_session.non_rew_lick_cnt >= mouse.learning_nonrew_lick[0]:
        old_tau, gain = mouse.exp_update_no_reward
        mouse.exp_update_no_reward = (old_tau + mouse.learning_nonrew_lick[1], gain)
        mouse_session.non_rew_lick_cnt = 0

    # Sensibilisation No-Go generale : meme principe et meme standardisation que
    # function_update_mouse_state (mono) — tau ET gain No-Go grandissent ensemble a ce
    # declenchement (le gain apprend selon la meme regle delta que le gain Go, sur -rpe_t).
    if mouse_session.nogo_streak_cnt >= mouse.nogo_streak_incr[0]:
        old_tau_nogo, gain_nogo_cur = mouse.exp_update_nogo
        new_tau_nogo = min(old_tau_nogo + mouse.nogo_streak_incr[1], mouse.nogo_tau_max)
        new_gain_nogo = gain_nogo_cur
        if mouse.learning_nogo_gain > 0:
            new_gain_nogo = float(np.clip(
                gain_nogo_cur + (-rpe_t) * mouse.learning_nogo_gain, 0.0, mouse.nogo_gain_max
            ))
        mouse.exp_update_nogo = (new_tau_nogo, new_gain_nogo)
        mouse_session.nogo_streak_cnt = 0

    # --- Apprentissage des gains, a la recompense --------------------------------------------
    if reward_t > 0.0:
        elig1 = float(mouse_session.eligibility1[i, 0])
        elig2 = float(mouse_session.eligibility2[i, 0])
        tau1, g1 = mouse.exp_update_stim1_wa
        tau2, g2 = mouse.exp_update_stim2_wa
        g1_new = g1 + rpe_t * mouse.learning_stim_wa * elig1
        g2_new = g2 + rpe_t * mouse.learning_stim_wa * elig2
        mouse.exp_update_stim1_wa = (tau1, float(np.clip(g1_new, 0.0, mouse.stim_gain_max_wa)))
        mouse.exp_update_stim2_wa = (tau2, float(np.clip(g2_new, 0.0, mouse.stim_gain_max_wa)))
    elif is_new_unrewarded_bout and reward_denied_stim in (1, 2):
        # Standardisation : meme principe que Mouse.stim_gain_noreward_active (mono/dual) — le
        # gain du stimulus concerne descend directement, avec exactement la meme formule que sa
        # propre montee (pas de voie No-Go competitrice separee). Ici toujours actif (voir
        # commentaire sur Mouse.exp_update_stim1_wa/2_wa) : declenche par le mismatch de
        # contingence plutot qu'un tirage de probabilite refuse, gate bout-onset pour ne pas
        # re-declencher a chaque bin d'un meme bout soutenu.
        if reward_denied_stim == 1:
            elig1 = float(mouse_session.eligibility1[i, 0])
            tau1, g1 = mouse.exp_update_stim1_wa
            g1_new = g1 + rpe_t * mouse.learning_stim_wa * elig1
            mouse.exp_update_stim1_wa = (tau1, float(np.clip(g1_new, 0.0, mouse.stim_gain_max_wa)))
        else:
            elig2 = float(mouse_session.eligibility2[i, 0])
            tau2, g2 = mouse.exp_update_stim2_wa
            g2_new = g2 + rpe_t * mouse.learning_stim_wa * elig2
            mouse.exp_update_stim2_wa = (tau2, float(np.clip(g2_new, 0.0, mouse.stim_gain_max_wa)))

    return mouse, mouse_session


def function_wdt_session_wa(
    session_param: WDTSesssionParams,
    wdt_session: WDTSesssionState,
    session_info: SessionBaseInfo,
    lick: int,
    i: int,
) -> tuple[float, float, float, WDTSesssionState, bool, int]:
    """Comme function_wdt_session (mono), avec 3 types de trial (0=catch, 1=whisker,
    2=auditif) au lieu de 2. Seul un lick pendant la fenetre de reponse ET dont le type de
    trial correspond a `session_param.reward_stim` est recompense — un lick sur l'AUTRE
    stimulus n'est jamais recompense, exactement comme un lick de catch trial."""
    t = i * session_info.resolution
    reward = 0.0
    stim1 = 0.0
    stim2 = 0.0
    new_trial = False
    # 0=aucun, 1/2=whisker/auditif : stimulus dont ce lick, dans sa propre fenetre de reponse,
    # n'a pas ete recompense parce que ce n'est pas actuellement le stimulus recompense
    # (mismatch de contingence) — l'equivalent whisker/auditif de reward_denied_by_prob
    # (mono/dual), sauf que ce n'est jamais du a un tirage de probabilite : c'est certain des
    # que le mauvais stimulus est presente, puisque le paradigme a toujours un cote "faux".
    reward_denied_stim = 0

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
            stim1 = session_param.stim1_amp
            wdt_session.last_stim_time = t
        elif kind == 2:
            stim2 = session_param.stim2_amp
            wdt_session.last_stim_time = t

        if kind in (1, 2):
            window_length = int(round(session_param.response_wind / session_info.resolution))
            end_idx = min(i + window_length, session_info.number_bin)
            wdt_session.reward_window[i + 1:end_idx, 0] = 1

    if lick == 1:
        wdt_session.last_lick_time = t
        if (
            wdt_session.reward_window[i, 0] > 0.5
            and (t - wdt_session.last_reward_time) > 2
        ):
            if wdt_session.last_trial_kind == session_param.reward_stim:
                if random.random() <= session_param.reward_prob:
                    reward = session_param.reward_size
                    wdt_session.last_reward_time = t
            elif wdt_session.last_trial_kind in (1, 2):
                reward_denied_stim = wdt_session.last_trial_kind

    wdt_session.reward[i, 0] = reward
    if stim1 > 0:
        wdt_session.stim1[i, 0] = stim1
    if stim2 > 0:
        wdt_session.stim2[i, 0] = stim2

    return reward, stim1, stim2, wdt_session, new_trial, reward_denied_stim


def function_performance_wdt_wa(
    wdt_params: WDTSesssionParams,
    session_info: SessionBaseInfo,
    wdt_session: WDTSesssionState,
    mouse_session: MouseSessionState,
) -> np.ndarray:
    """Comme function_performance_wdt (mono), avec une colonne stim_code (0/1/2) au lieu
    d'une amplitude, pour calculer un Hit Rate separe par stimulus (plot_session_rates_wa
    s'en sert pour montrer le switch de contingence)."""
    sr = int(round(1 / session_info.resolution))
    trials = [t for t in wdt_session.trial_times if t < (session_info.duration * 60 - wdt_params.response_wind)]

    performance = []
    for t in trials:
        pt1 = int(round(t * sr))
        pt2 = min(pt1 + int(round(wdt_params.response_wind * sr)), session_info.number_bin)

        s1 = float(np.max(wdt_session.stim1[pt1:pt2, 0]))
        s2 = float(np.max(wdt_session.stim2[pt1:pt2, 0]))
        stim_code = 1.0 if s1 > 0.0 else (2.0 if s2 > 0.0 else 0.0)

        lick_segment = mouse_session.lick[pt1:pt2, 0]
        reward_segment = wdt_session.reward[pt1:pt2, 0]

        lick_detected = bool(np.any(lick_segment))
        reward_detected = bool(np.any(reward_segment))
        latency = float(np.argmax(lick_segment) / sr) if lick_detected else float("nan")

        if stim_code > 0 and not lick_detected:
            outcome = 0  # Miss
        elif stim_code > 0 and lick_detected:
            outcome = 1  # Hit
        elif stim_code == 0 and not lick_detected:
            outcome = 2  # Correct Rejection
        else:
            outcome = 3  # False Alarm

        performance.append([t, stim_code, int(lick_detected), latency, int(reward_detected), outcome])

    return np.array(performance, dtype=float)


def run_wdt_session_wa(
    mouse: Mouse,
    session_info: SessionBaseInfo,
    config: SimConfig,
    wdt_params: WDTSesssionParams,
    prev_mouse_session: MouseSessionState,
    prev_reward: np.ndarray,
    label: str,
) -> WDTRunResult:
    """Corps d'une session WDT/AUD (whisker/auditif, une seule Expectation partagee)."""
    lick_sum = float(np.sum(prev_mouse_session.lick))
    reward_sum = float(np.sum(prev_reward))
    init_expect = (reward_sum / lick_sum) if lick_sum > 0 else 0.0
    init_uncert = float(prev_mouse_session.uncertainty[-1, 0])
    init_nogo = float(prev_mouse_session.expectation_nogo[-1, 0])

    wdt_session = WDTSesssionState()
    wdt_session.initialize(session_info.number_bin, wdt_params.no_lick_wind, wdt_params.iti)
    mouse_session = MouseSessionState()
    mouse_session.initialize(session_info.number_bin, mouse.motivation[0], init_expect, init_uncert, init_nogo)

    for i in range(1, session_info.number_bin):
        reward_t, s1, s2, wdt_session, _, reward_denied_stim = function_wdt_session_wa(
            wdt_params, wdt_session, session_info, int(mouse_session.lick[i - 1, 0]), i
        )
        _, mouse_session = function_mouse_lick(
            mouse, mouse_session, i, threshold=mouse.lick_thrs_wdt, decision_slope=mouse.decision_slope_wdt
        )
        mouse, mouse_session = function_update_mouse_state_wa(
            mouse, mouse_session, session_info, i, s1, s2, reward_t,
            in_trial_window_prev=int(wdt_session.reward_window[i - 1, 0] > 0.5),
            reward_denied_stim=reward_denied_stim,
        )

    perf = function_performance_wdt_wa(wdt_params, session_info, wdt_session, mouse_session)

    return WDTRunResult(
        label=label, session=wdt_session, mouse_session=mouse_session, perf=perf,
        stim1_gain=float(mouse.exp_update_stim1_wa[1]), stim2_gain=float(mouse.exp_update_stim2_wa[1]),
    )


def run_all_wdt_wa(mouse: Mouse, session_info: SessionBaseInfo, config: SimConfig, log_fl2: FLSessionResult) -> WDTBundle:
    """Enchaine WA1..WA{config.WA_NUM_SESSIONS}. Sessions 1..WA_SWITCH_SESSION-1 : stim1
    (whisker) recompense, etiquette "WDT{k}". Sessions WA_SWITCH_SESSION..fin : stim2
    (auditif) recompense, etiquette "AUD{k}" — paradigme historique du projet."""
    bundle = WDTBundle()
    prev_mouse_session = log_fl2.mouse_session
    prev_reward = log_fl2.session.reward

    for s_idx in range(1, config.WA_NUM_SESSIONS + 1):
        wdt_params = WDTSesssionParams(
            trial_kinds=config.WA_TRIAL_KINDS[:],
            stim1_amp=config.WA_STIM1_AMP,
            stim2_amp=config.WA_STIM2_AMP,
            reward_stim=1 if s_idx < config.WA_SWITCH_SESSION else 2,
        )
        label = f"WDT{s_idx}" if s_idx < config.WA_SWITCH_SESSION else f"AUD{s_idx}"
        result = run_wdt_session_wa(mouse, session_info, config, wdt_params, prev_mouse_session, prev_reward, label)

        if result.perf.size > 0:
            s1_mask = result.perf[:, 1] == 1
            s2_mask = result.perf[:, 1] == 2
            catch_mask = result.perf[:, 1] == 0
            hr1 = (np.sum((result.perf[:, 5] == 1) & s1_mask) / np.sum(s1_mask)) if np.sum(s1_mask) > 0 else 0.0
            hr2 = (np.sum((result.perf[:, 5] == 1) & s2_mask) / np.sum(s2_mask)) if np.sum(s2_mask) > 0 else 0.0
            fa = (np.sum((result.perf[:, 5] == 3) & catch_mask) / np.sum(catch_mask)) if np.sum(catch_mask) > 0 else 0.0
        else:
            hr1, hr2, fa = 0.0, 0.0, 0.0

        print(f"Hit Rate {label} : stim1(whisker)={hr1:.2f}  stim2(auditif)={hr2:.2f}  FA={fa:.2f}"
              f"  |  gains(stim1,stim2)=({result.stim1_gain:.3f},{result.stim2_gain:.3f})")

        bundle.results.append(result)
        bundle.labels.append(label)

        prev_mouse_session = result.mouse_session
        prev_reward = result.session.reward

    return bundle


def plot_all_results_wa(
    session_info: SessionBaseInfo,
    mouse: Mouse,
    config: SimConfig,
    log_fl1: FLSessionResult,
    log_fl2: FLSessionResult,
    wdt_bundle: WDTBundle,
) -> None:
    """Version whisker/auditif de plot_all_results (mono) : reutilise plot_traces (mono,
    en passant les deux canaux stim1/stim2), plus les plots dedies HR(stim1/stim2)/FA."""

    if config.PLOT_TRACES:
        plot_traces(session_info.time_vector, log_fl1.mouse_session, log_fl1.session.reward,
                    stim_array=None, title="Free Licking — Session 1",
                    save_name="free_licking_1", threshold=mouse.lick_thrs)
        plot_traces(session_info.time_vector, log_fl2.mouse_session, log_fl2.session.reward,
                    stim_array=None, title="Free Licking — Session 2",
                    save_name="free_licking_2", threshold=mouse.lick_thrs)
        for lbl, result in zip(wdt_bundle.labels, wdt_bundle.results):
            plot_traces(session_info.time_vector, result.mouse_session, result.session.reward,
                        stim_array=result.session.stim1, stim2_array=result.session.stim2,
                        title=f"Two-Stimulus Detection — {lbl}",
                        save_name=lbl.lower(), threshold=mouse.lick_thrs_wdt)

    if config.PLOT_BLOCK_HRFA:
        _wdt_params = WDTSesssionParams()
        for lbl, result in zip(wdt_bundle.labels, wdt_bundle.results):
            plot_wdt_block_rates_wa(result.perf, _wdt_params, max_trials=config.MAX_TRIALS_BLOCKS,
                                    title=f"{lbl} — HR (stim1/stim2) & FA by block", save_name=lbl.lower())

    if config.PLOT_SESSIONS_COMPARISON:
        wdt_perfs = [r.perf for r in wdt_bundle.results]
        plot_session_rates_wa(wdt_perfs, wdt_bundle.labels,
                              title="Learning — HR(stim1/stim2) & FA(catch) across sessions",
                              save_name="hr_fa_all_sessions_wa", switch_session=config.WA_SWITCH_SESSION)

    if config.PLOT_SINGLE_SESSION_ALL:
        for lbl, result in zip(wdt_bundle.labels, wdt_bundle.results):
            plot_single_session_rates_wa(result.perf, title=f"{lbl} — HR(stim1/stim2) vs FA", save_name=lbl.lower())

    plot_stim_gains(wdt_bundle, config, title="Evolution des gains de stimuli",
                    save_name="stim_gains_evolution")

    if config.PLOT_RPE_ALL:
        tbl_fl1 = extract_rpe_per_lick(log_fl1.mouse_session, log_fl1.session.reward, None, session_info)
        plot_rpe_per_lick(tbl_fl1, title="FL1 — RPE per lick", save_name="free_licking_1_rpe")
        tbl_fl2 = extract_rpe_per_lick(log_fl2.mouse_session, log_fl2.session.reward, None, session_info)
        plot_rpe_per_lick(tbl_fl2, title="FL2 — RPE per lick", save_name="free_licking_2_rpe")
        for lbl, result in zip(wdt_bundle.labels, wdt_bundle.results):
            stim_vis = result.session.stim1 + 2.0 * result.session.stim2
            tbl = extract_rpe_per_lick(result.mouse_session, result.session.reward, stim_vis, session_info)
            plot_rpe_per_lick(tbl, title=f"{lbl} — RPE per lick", save_name=f"{lbl.lower()}_rpe")

    if config.PLOT_ABS_RPE_ALL:
        for lbl, result in zip(wdt_bundle.labels, wdt_bundle.results):
            stim_vis = result.session.stim1 + 2.0 * result.session.stim2
            tbl = extract_rpe_per_lick(result.mouse_session, result.session.reward, stim_vis, session_info)
            plot_abs_rpe_per_lick(tbl, title=f"{lbl} — |RPE| per lick", save_name=f"{lbl.lower()}_abs_rpe")


def save_run_parameters_wa(config: SimConfig, mouse: Mouse, session_info: SessionBaseInfo) -> None:
    """Ecrit parameters.txt pour le mode whisker/auditif — variante de save_run_parameters (mono)."""
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
        ("Lick threshold — WDT/AUD", config.LICK_THRS_WDT),

        ("MOUSE — MOTIVATION", None),
        ("Motivation (initial)", _ref_mouse.motivation[0]),
        ("Motivation loss per reward", _ref_mouse.motivation[1]),

        ("MOUSE — VALUE & COST (DECISION GATE: V·M − C)", None),
        ("Reward value (V)", config.REWARD_VALUE),
        ("Cost (C)", config.COST),

        ("MOUSE — STIMULI (WHISKER=1, AUDITIF=2)", None),
        ("Tau (stimulus, s)", _ref_mouse.exp_update_stim1_wa[0]),
        ("Gain initial (stim1, whisker)", _ref_mouse.exp_update_stim1_wa[1]),
        ("Gain initial (stim2, auditif)", _ref_mouse.exp_update_stim2_wa[1]),
        ("Stimulus gain learning rate (monte a la recompense, descend au mismatch)", _ref_mouse.learning_stim_wa),
        ("Stimulus gain cap", _ref_mouse.stim_gain_max_wa),
        ("Eligibility trace tau (s)", _ref_mouse.tau_eligibility),

        ("MOUSE — ARCHITECTURE GO/NO-GO GENERALE (E_go - E_nogo, base)", None),
        ("Tau (no-go, s)", config.NOGO_TAU),
        ("Gain (no-go)", config.NOGO_GAIN),
        ("Relief (fraction effacee par reward)", config.NOGO_RELIEF),
        ("Sensibilisation : seuil (lechages non recompenses consecutifs)", config.NOGO_STREAK_THRESHOLD),
        ("Sensibilisation : increment de tau", config.NOGO_TAU_GROWTH),
        ("Sensibilisation : tau max", config.NOGO_TAU_MAX),

        ("TWO-STIMULUS DETECTION TASK (SWITCH DE CONTINGENCE)", None),
        ("Number of sessions", config.WA_NUM_SESSIONS),
        ("Sessions recompensant stim1 (whisker)", config.WA_SWITCH_SESSION - 1),
        ("Sessions recompensant stim2 (auditif)", config.WA_NUM_SESSIONS - config.WA_SWITCH_SESSION + 1),
        ("Stim1 (whisker) amplitude", config.WA_STIM1_AMP),
        ("Stim2 (auditif) amplitude", config.WA_STIM2_AMP),
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

    Le cote n'est pas lu par une comparaison dure : un bruit independant (Gumbel, echelle
    1/side_readout_slope) est ajoute a E_droite et a E_gauche separement, puis on prend le
    max (comme dans Lak et al. 2020 — le bruit vit sur les deux Expectations, pas au moment
    du choix). Ce mecanisme est l'equivalent exact (Gumbel-max trick) de l'ancienne sigmoide
    P(droite) = sigmoide(side_readout_slope . (E_droite - E_gauche)) : meme statistique de
    choix agregee, mais le bruit est maintenant sur les deux representations plutot qu'une
    probabilite calculee sur des valeurs propres. Bruit fixe, independant de l'Uncertainty
    (qui ne module que la decision Go/No-Go plus bas, pas ce choix de cote).
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

    beta_side = 1.0 / mouse.side_readout_slope
    eps_right = np.random.gumbel(0.0, beta_side)
    eps_left = np.random.gumbel(0.0, beta_side)
    side = 1 if (e_right_t + eps_right) >= (e_left_t + eps_left) else -1
    e_side = e_right_t if side == 1 else e_left_t

    # Architecture Go/No-Go : le choix du cote (ci-dessus) reste base sur E_right/E_left
    # brutes — c'est une decision "lequel", separee de "est-ce que j'y vais". La voie
    # No-Go ne module que le cote deja selectionne, une fois qu'il s'agit d'agir ou non.
    e_nogo_side = (
        mouse_session.expectation_nogo_right[i - 1, 0] if side == 1
        else mouse_session.expectation_nogo_left[i - 1, 0]
    )
    e_side = e_side - e_nogo_side

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
    in_trial_window_prev: int = 0,
    reward_denied_by_prob: bool = False,
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

    # E_nogo par cote (voie No-Go, architecture Go/No-Go) : meme logique que la version
    # mono-stimulus, appliquee independamment a chaque cote via ref_side_t (le cote du
    # reward, ou le cote du dernier lick si pas de reward — meme reference que pour la
    # mise a jour de l'expectation ci-dessous). Un cote qui cesse d'etre recompense
    # accumule sa propre mefiance sans jamais affecter l'autre cote — y compris le tau/gain
    # (exp_update_nogo_right/left separes), sinon la sensibilisation d'un cote en extinction
    # ferait aussi ralentir la decroissance d'E_nogo de l'autre cote encore recompense.
    tau_nogo_r, gain_nogo_r = mouse.exp_update_nogo_right
    tau_nogo_l, gain_nogo_l = mouse.exp_update_nogo_left
    decay_nogo_r = np.exp(-session_info.resolution / tau_nogo_r)
    decay_nogo_l = np.exp(-session_info.resolution / tau_nogo_l)

    e_nogo_r_new = mouse_session.expectation_nogo_right[i - 1, 0] * decay_nogo_r
    e_nogo_l_new = mouse_session.expectation_nogo_left[i - 1, 0] * decay_nogo_l

    is_new_unrewarded_bout = (
        reward_t == 0
        and mouse_session.lick[i - 1, 0] == 1
        and ref_side_t != 0
        and (i < 2 or not (mouse_session.lick[i - 2, 0] == 1 and mouse_session.lick_side[i - 2, 0] == ref_side_t))
    )
    if is_new_unrewarded_bout:
        if ref_side_t == 1:
            e_nogo_r_new = e_nogo_r_new + gain_nogo_r * (1.0 - e_nogo_r_new)
        else:
            e_nogo_l_new = e_nogo_l_new + gain_nogo_l * (1.0 - e_nogo_l_new)

    if reward_t > 0 and ref_side_t != 0:
        # "Soulagement" localise : une recompense confirmee sur ce cote efface une partie
        # de la mefiance deja accumulee sur CE cote uniquement (reacquisition rapide).
        if ref_side_t == 1:
            e_nogo_r_new = e_nogo_r_new * (1.0 - mouse.nogo_relief)
        else:
            e_nogo_l_new = e_nogo_l_new * (1.0 - mouse.nogo_relief)

    mouse_session.expectation_nogo_right[i, 0] = float(np.clip(e_nogo_r_new, 0.0, 1.0))
    mouse_session.expectation_nogo_left[i, 0] = float(np.clip(e_nogo_l_new, 0.0, 1.0))

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
        if is_new_unrewarded_bout and in_trial_window_prev:
            if ref_side_t == 1:
                mouse_session.nogo_streak_cnt_right += 1
            elif ref_side_t == -1:
                mouse_session.nogo_streak_cnt_left += 1
    elif reward_t > 0 and mouse_session.lick[i - 1, 0] == 1:
        mouse_session.non_rew_lick_cnt = 0
        if ref_side_t == 1:
            mouse_session.nogo_streak_cnt_right = 0
        elif ref_side_t == -1:
            mouse_session.nogo_streak_cnt_left = 0

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

    # Standardisation : tau fixe par defaut, le gain No-Go apprend a la place, separement par
    # cote (meme regle et meme cadence que la version mono).
    if mouse_session.nogo_streak_cnt_right >= mouse.nogo_streak_incr[0]:
        old_tau_nogo_r, gain_nogo_r_cur = mouse.exp_update_nogo_right
        new_tau_nogo_r = min(old_tau_nogo_r + mouse.nogo_streak_incr[1], mouse.nogo_tau_max)
        new_gain_nogo_r = gain_nogo_r_cur
        if mouse.learning_nogo_gain > 0:
            new_gain_nogo_r = float(np.clip(
                gain_nogo_r_cur + (-rpe_t) * mouse.learning_nogo_gain, 0.0, mouse.nogo_gain_max
            ))
        mouse.exp_update_nogo_right = (new_tau_nogo_r, new_gain_nogo_r)
        mouse_session.nogo_streak_cnt_right = 0

    if mouse_session.nogo_streak_cnt_left >= mouse.nogo_streak_incr[0]:
        old_tau_nogo_l, gain_nogo_l_cur = mouse.exp_update_nogo_left
        new_tau_nogo_l = min(old_tau_nogo_l + mouse.nogo_streak_incr[1], mouse.nogo_tau_max)
        new_gain_nogo_l = gain_nogo_l_cur
        if mouse.learning_nogo_gain > 0:
            new_gain_nogo_l = float(np.clip(
                gain_nogo_l_cur + (-rpe_t) * mouse.learning_nogo_gain, 0.0, mouse.nogo_gain_max
            ))
        mouse.exp_update_nogo_left = (new_tau_nogo_l, new_gain_nogo_l)
        mouse_session.nogo_streak_cnt_left = 0

    if reward_t > 0 and ref_side_t == 1:
        mouse.stim_gain_right = min(mouse.stim_gain_max, max(0.0, mouse.stim_gain_right + rpe_t * mouse.learning_stim * mouse_session.eligibility_right[i, 0]))
    elif reward_t > 0 and ref_side_t == -1:
        mouse.stim_gain_left = min(mouse.stim_gain_max, max(0.0, mouse.stim_gain_left + rpe_t * mouse.learning_stim * mouse_session.eligibility_left[i, 0]))
    elif mouse.stim_gain_noreward_active and is_new_unrewarded_bout and reward_denied_by_prob:
        # Baisse symetrique de stim_gain, UNIQUEMENT quand reward_denied_by_prob est vrai —
        # c.a.d. ce lick a atteint le tirage de probabilite de recompense (bon cote, dans la
        # fenetre, cooldown ecoule : "aurait du" etre recompense) et ce tirage a echoue. Avec
        # reward_prob=1.0 (defaut hors desapprentissage cible), ce tirage ne peut structurellement
        # jamais echouer, donc ce bloc reste inerte en entrainement normal. Gate bout-onset
        # (is_new_unrewarded_bout) en plus, pour ne pas re-declencher a chaque bin d'un meme
        # bout soutenu pendant un desapprentissage prolonge.
        if ref_side_t == 1:
            mouse.stim_gain_right = min(mouse.stim_gain_max, max(0.0, mouse.stim_gain_right + rpe_t * mouse.learning_stim * mouse_session.eligibility_right[i, 0]))
        elif ref_side_t == -1:
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
) -> tuple[float, int, float, DualWDTSessionState, bool, bool]:
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
    # True uniquement quand ce lick a atteint le tirage de probabilite de recompense (bon
    # cote, dans la fenetre, cooldown ecoule — donc "aurait du" etre recompense) et que ce
    # tirage a echoue. reward_prob=1.0 (defaut, hors desapprentissage) rend ca structurellement
    # impossible ; ne devient vrai que si reward_prob/right/left < 1 (desapprentissage cible).
    reward_denied_by_prob = False

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
                if lick_side == 1 and session_param.reward_prob_right is not None:
                    eff_reward_prob = session_param.reward_prob_right
                elif lick_side == -1 and session_param.reward_prob_left is not None:
                    eff_reward_prob = session_param.reward_prob_left
                else:
                    eff_reward_prob = session_param.reward_prob
                if random.random() <= eff_reward_prob:
                    reward = session_param.reward_size
                    reward_side = lick_side
                    wdt_session.last_reward_time = t
                else:
                    reward_denied_by_prob = True

    wdt_session.reward[i, 0] = reward
    wdt_session.reward_side[i, 0] = reward_side
    if stim_signed != 0:
        wdt_session.stim_signed[i, 0] = stim_signed
        wdt_session.stim1[i, 0] = abs(stim_signed)

    return reward, reward_side, stim_signed, wdt_session, new_trial, reward_denied_by_prob


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
    init_nogo_right = float(log_fl1.mouse_session.expectation_nogo_right[-1, 0])
    init_nogo_left = float(log_fl1.mouse_session.expectation_nogo_left[-1, 0])

    fl_session = DualFreeLickingSessionState()
    fl_session.initialize(session_info.number_bin, fl_params.no_lick_wind)
    mouse_session = DualMouseSessionState()
    mouse_session.initialize(session_info.number_bin, mouse.motivation[0], init_expect_right, init_expect_left,
                              init_uncert, init_nogo_right, init_nogo_left)

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
    init_nogo_right = float(prev_mouse_session.expectation_nogo_right[-1, 0])
    init_nogo_left = float(prev_mouse_session.expectation_nogo_left[-1, 0])

    wdt_session = DualWDTSessionState()
    wdt_session.initialize(session_info.number_bin, wdt_params.no_lick_wind, wdt_params.iti)
    mouse_session = DualMouseSessionState()
    mouse_session.initialize(session_info.number_bin, mouse.motivation[0], init_expect_right, init_expect_left,
                              init_uncert, init_nogo_right, init_nogo_left)

    for i in range(1, session_info.number_bin):
        reward_t, reward_side_t, stim_signed_t, wdt_session, _, reward_denied_by_prob = function_wdt_dualstim_session(
            wdt_params, wdt_session, session_info,
            int(mouse_session.lick[i - 1, 0]), int(mouse_session.lick_side[i - 1, 0]), i,
        )
        _, _, mouse_session = function_mouse_lick_dual(
            mouse, mouse_session, i,
            threshold=mouse.lick_thrs_wdt, decision_slope=mouse.decision_slope_wdt,
            v_right=config.V_RIGHT, v_left=config.V_LEFT, c_right=config.C_RIGHT, c_left=config.C_LEFT,
        )
        mouse, mouse_session = function_update_mouse_state_dual(
            mouse, mouse_session, session_info, i, stim_signed_t, reward_t, reward_side_t,
            in_trial_window_prev=int(wdt_session.reward_window[i - 1, 0] > 0.5),
            reward_denied_by_prob=reward_denied_by_prob,
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
        if (
            config.DELEARNING_FROM_SESSION is not None
            and s_idx >= config.DELEARNING_FROM_SESSION
            and (config.DELEARNING_UNTIL_SESSION is None or s_idx < config.DELEARNING_UNTIL_SESSION)
        ):
            if config.DELEARNING_SIDE == 1:
                wdt_params.reward_prob_right = 0.0
            elif config.DELEARNING_SIDE == -1:
                wdt_params.reward_prob_left = 0.0
            else:
                wdt_params.reward_prob = 0.0

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
        plot_session_rates(wdt_perfs, wdt_bundle.labels, title="Learning — HR, Mismatch, FA & % essais réussis (WDT1..WDT10)",
                            save_name="hr_fa_dprime_all_sessions", dual_stim=True)

        # Meme courbe, mais recalculee a Motivation=1.0 tout du long (annule l'effet de
        # satiete intra-session sur V*M-C) pour voir la "vraie" performance (ce que E encode).
        wdt_perfs_norm = [_normalize_perf_for_motivation(r, True, config) for r in wdt_bundle.results]
        plot_session_rates(wdt_perfs_norm, wdt_bundle.labels,
                            title="Learning — HR, Mismatch, FA & % essais réussis (WDT1..WDT10) — normalisé Motivation=1.0",
                            save_name="hr_fa_dprime_all_sessions_normalized_motivation", dual_stim=True)

    if config.PLOT_SINGLE_SESSION_ALL:
        for lbl, result in zip(wdt_bundle.labels, wdt_bundle.results):
            plot_single_session_rates(result.perf, title=f"{lbl} — HR vs Mismatch vs FA", save_name=lbl.lower(), dual_stim=True)

    if config.DELEARNING_FROM_SESSION is not None:
        plot_delearning_diagnostics(
            session_info,
            [r.mouse_session for r in wdt_bundle.results],
            wdt_bundle.labels,
            config.DELEARNING_FROM_SESSION,
            perfs=[r.perf for r in wdt_bundle.results],
            dual_stim=True,
            delearning_until_session=config.DELEARNING_UNTIL_SESSION,
        )


def _normalize_perf_for_motivation(result: WDTRunResult, dual_stim: bool, quiet: SimConfig) -> np.ndarray:
    """
    Recalcule les issues des essais (perf) comme si la Motivation etait restee a 1.0 toute
    la session, pour annuler l'effet de satiete intra-session sur la porte V*M-C et isoler
    la "vraie" performance (ce que E encode) — cf. mecanisme identifie sur le plateau de
    Hit Rate: D = drive * (V*M - C), donc a seuil egal on peut retro-corriger D en le
    remultipliant par (V-C)/(V*M-C). Ne touche pas au choix du cote en dual (independant
    de M, deja tranche par le bruit Gumbel sur E_droite/E_gauche).
    """
    ms = result.mouse_session
    M = ms.motivation[:, 0]
    p = ms.p_lick[:, 0]

    if dual_stim:
        side = np.sign(p)
        d_actual = np.abs(p)
        v_side = np.where(side >= 0, quiet.V_RIGHT, quiet.V_LEFT)
        c_side = np.where(side >= 0, quiet.C_RIGHT, quiet.C_LEFT)
    else:
        d_actual = p
        v_side = quiet.REWARD_VALUE
        c_side = quiet.COST

    gate_actual = v_side * M - c_side
    gate_ref = v_side - c_side  # motivation de reference = 1.0
    safe_gate = np.where(np.abs(gate_actual) < 1e-6, 1e-6, gate_actual)
    d_norm = d_actual * gate_ref / safe_gate

    lick_norm = (d_norm >= quiet.LICK_THRS_WDT).astype(float).reshape(-1, 1)

    if dual_stim:
        lick_side_norm = (side * lick_norm[:, 0]).reshape(-1, 1)
        norm_ms = replace(ms, lick=lick_norm, lick_side=lick_side_norm)
        return function_performance_wdt_dual(DualWDTSessionParams(), SessionBaseInfo(), result.session, norm_ms)
    else:
        norm_ms = replace(ms, lick=lick_norm)
        return function_performance_wdt(WDTSesssionParams(), SessionBaseInfo(), result.session, norm_ms)


def _run_population_at_spread(quiet: SimConfig, n_mice: int, spread: float) -> tuple[list, list[str]]:
    """Lance n_mice souris a +/- spread autour des valeurs de base (Mouse()). Retourne
    (records, summary_lines) — utilise par run_population, une fois ou en boucle (sweep)."""
    base_learning_stim = Mouse().learning_stim
    base_noise_scale = Mouse().noise[1]

    records = []
    summary_lines = [f"Population — {n_mice} souris ({'dual' if quiet.dual_stim else 'mono'}), "
                      f"spread +/-{spread*100:.0f}% autour de learning_stim={base_learning_stim} "
                      f"et noise[1]={base_noise_scale}", ""]

    for k in range(n_mice):
        learning_stim_i = base_learning_stim * random.uniform(1 - spread, 1 + spread)
        noise_scale_i = base_noise_scale * random.uniform(1 - spread, 1 + spread)

        session_info = SessionBaseInfo()
        mouse = _build_mouse(quiet, learning_stim=learning_stim_i, noise=(1.2, noise_scale_i))

        label = f"M{k + 1}"
        if quiet.dual_stim:
            log_fl1 = run_FL1_dualstim(mouse, session_info, quiet)
            log_fl2 = run_FL2_dualstim(mouse, session_info, quiet, log_fl1)
            wdt_bundle = run_all_wdt_dualstim(mouse, session_info, quiet, log_fl2)
        else:
            log_fl1 = run_FL1(mouse, session_info, quiet)
            log_fl2 = run_FL2(mouse, session_info, quiet, log_fl1)
            wdt_bundle = run_all_wdt(mouse, session_info, quiet, log_fl2)

        records.append((label, learning_stim_i, noise_scale_i, wdt_bundle))
        summary_lines.append(f"{label}: learning_stim={learning_stim_i:.5f}  noise[1]={noise_scale_i:.4f}")

    return records, summary_lines


def run_population(config: SimConfig) -> None:
    """
    Lance config.POPULATION_N_MICE souris (mono ou dual selon config.dual_stim), chacune avec
    son propre learning_stim et sa propre echelle de bruit (mouse.noise[1]), tires uniformement
    a +/- config.POPULATION_PARAM_SPREAD autour des valeurs de base (Mouse()). Pas de plots
    individuels par souris (FL/WDT/traces...) : seulement une comparaison des courbes
    d'apprentissage (Hit Rate par session WDT) et de l'evolution du Hit Rate pendant WDT10,
    avec les parametres qui varient en legende.

    Si config.POPULATION_SPREAD_SWEEP est True, repete l'operation pour chaque pourcentage de
    config.POPULATION_SWEEP_VALUES (un jeu de plots par pourcentage, indique dans le titre et
    le nom de fichier), toujours dans le meme dossier results/.../population/.
    """
    n_mice = config.POPULATION_N_MICE

    suffix = ("dual" if config.dual_stim else "mono") + "_population"
    save_dir = init_results_dir(suffix=suffix)

    quiet = replace(
        config,
        PLOT_TRACES=False, PLOT_BLOCK_HRFA=False, PLOT_SESSIONS_COMPARISON=False,
        PLOT_SINGLE_SESSION_ALL=False, PLOT_STOCHASTIC_IN_POPULATION=False,
        PLOT_RPE_ALL=False, PLOT_ABS_RPE_ALL=False, PLOT_PSYCHO_TEST=False,
        SAVE_PARAMETERS_TXT=False,
    )

    spreads = config.POPULATION_SWEEP_VALUES if config.POPULATION_SPREAD_SWEEP else (config.POPULATION_PARAM_SPREAD,)

    all_summary_lines = []
    for spread in spreads:
        records, summary_lines = _run_population_at_spread(quiet, n_mice, spread)

        pct = round(spread * 100)
        title_suffix = f" — spread ±{pct}%" if config.POPULATION_SPREAD_SWEEP else ""
        category = f"population/p{pct}" if config.POPULATION_SPREAD_SWEEP else "population"

        plot_population_learning_curves(
            records, dual_stim=quiet.dual_stim,
            save_name="population_learning_curves", title_suffix=title_suffix, category=category,
        )

        # Meme courbe, mais avec les issues des essais recalculees a Motivation=1.0 tout du
        # long (annule l'effet de satiete intra-session identifie sur le plateau de Hit Rate),
        # pour voir la "vraie" performance (ce que E encode) independamment de la fatigue.
        records_norm = [
            (label, ls_i, ns_i, replace(wdt_bundle, results=[
                replace(r, perf=_normalize_perf_for_motivation(r, quiet.dual_stim, quiet))
                for r in wdt_bundle.results
            ]))
            for (label, ls_i, ns_i, wdt_bundle) in records
        ]
        plot_population_learning_curves(
            records_norm, dual_stim=quiet.dual_stim,
            save_name="population_learning_curves_normalized_motivation",
            title_suffix=f"{title_suffix} — normalisé Motivation=1.0", category=category,
        )

        n_sessions = min(len(r[3].results) for r in records) if records else 0
        for session_idx, session_tag in ((0, "wdt1"), (4, "wdt5"), (n_sessions - 1, "wdt10")):
            if not (0 <= session_idx < n_sessions):
                continue
            plot_population_session_progression(
                records, dual_stim=quiet.dual_stim, session_index=session_idx,
                save_name=f"population_session_progression_{session_tag}",
                title_suffix=title_suffix, category=category,
            )

        all_summary_lines.extend(summary_lines)
        all_summary_lines.append("")

    with open(f"{save_dir}/population_summary.txt", "w") as f:
        f.write("\n".join(all_summary_lines) + "\n")
