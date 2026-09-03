import numpy as np
import random
from typing import Any, Optional

from models import (
    Mouse,
    SessionBaseInfo,
    FreeLickingSessionParams,
    FreeLickingSessionState,
    MouseSessionState,
    WDTSesssionParams,
    WDTSesssionState,
    SimConfig,
    FLSessionResult,
    WDTRunResult,
    WDTBundle,
)

from plotting import (
    plot_traces,
    plot_wdt_block_rates,
    plot_session_rates,
    plot_single_session_rates,
    plot_rpe_per_lick,
    plot_abs_rpe_per_lick,
    plot_stim_gains,
    save_parameters_txt,
    init_results_dir,
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
    Decision function D — identique a Code_mono_stim_v2 (une seule Expectation partagee,
    que celle-ci soit alimentee par un ou deux stimuli ne change rien a la decision) :
        D = [E/(1+e^{A_D.(U-U0)}) + Nd/(1+e^{-A_D.(U-U0)})] . [V.M - C]
        Mouse licks if D >= threshold.
    """
    lick = 0

    motivation_t = mouse_session.motivation[i - 1, 0]
    expectation_t = mouse_session.expectation[i - 1, 0]
    if mouse.delearning_enable:
        # Drive net de la voie Go/No-Go generale : E_go - E_nogo. Si delearning_enable=False,
        # cette ligne n'est jamais executee et expectation_t reste E_go seule.
        expectation_t = expectation_t - mouse_session.expectation_nogo[i - 1, 0]
    uncertainty_t = mouse_session.uncertainty[i - 1, 0]

    gamma_noise = np.random.gamma(mouse.noise[0], mouse.noise[1])

    A = mouse.decision_slope if decision_slope is None else decision_slope
    u_rel = uncertainty_t - mouse.uncertainty_offset
    w_exploit = 1.0 / (1.0 + np.exp(A * u_rel))     # poids de E, decroit avec U
    w_explore = 1.0 / (1.0 + np.exp(-A * u_rel))    # poids de Nd, croit avec U

    drive = expectation_t * w_exploit + gamma_noise * w_explore

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
    stim1_t: float,
    stim2_t: float,
    reward_t: float,
) -> tuple[Mouse, MouseSessionState]:
    """
    Meme structure que Code_mono_stim_v2::function_update_mouse_state, etendue a DEUX canaux
    de stimulus (whisker=1, auditif=2) qui alimentent la MEME Expectation partagee, chacun
    avec sa propre eligibilite et son propre gain appris. Tout ce qui ne concerne pas les
    stimuli (motivation, RPE, Uncertainty, E_nogo general, mise a jour reward/no-reward de
    l'Expectation) est identique au mono, ligne pour ligne.
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

    # Uncertainty: identique au mono.
    mouse_session.uncertainty[i, 0] = mouse_session.uncertainty[i - 1, 0]
    if mouse_session.lick[i - 1, 0] == 1 or reward_t > 0:
        u_prev = mouse_session.uncertainty[i - 1, 0]
        u_new = u_prev + mouse.uncertainty_gain * (2.0 * abs(rpe_t) - 1.0) ** 3
        mouse_session.uncertainty[i, 0] = float(np.clip(u_new, 0.0, mouse.uncertainty_max))

    # --- E_nogo general (voie No-Go, desapprentissage) — identique au mono, capte
    # l'impulsivite generale independamment de quel stimulus (le cas echeant) etait present.
    if mouse.delearning_enable:
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
        # Comportement actuel (non modifie) : le lick non recompense soustrait directement
        # de l'expectation E_go — actif que delearning_enable soit True ou False.
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

    # --- Contribution des DEUX stimuli a l'Expectation partagee -----------------------------
    # Chaque stimulus injecte un "kick" decroissant proportionnel a son gain NET
    # (go - nogo si delearning_enable, go seul sinon) — exactement comme en mono, mais avec
    # deux canaux independants qui partagent la meme cible (expectation).
    if stim1_t > 0.0 or stim2_t > 0.0:
        tau1, gain1_go = mouse.exp_update_stim1
        tau2, gain2_go = mouse.exp_update_stim2
        net_gain1 = gain1_go - (mouse.stim1_nogo_gain if mouse.delearning_enable else 0.0)
        net_gain2 = gain2_go - (mouse.stim2_nogo_gain if mouse.delearning_enable else 0.0)

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

    # --- Apprentissage des gains de stimulus (Go), a la recompense --------------------------
    # Identique a l'ancien Code_dual_stim : chaque gain est mis a jour via SA PROPRE
    # eligibilite, donc seul le stimulus recemment presente apprend (l'autre a une
    # eligibilite quasi nulle et n'est quasiment pas affecte).
    if reward_t > 0.0:
        elig1 = float(mouse_session.eligibility1[i, 0])
        elig2 = float(mouse_session.eligibility2[i, 0])
        tau1, g1 = mouse.exp_update_stim1
        tau2, g2 = mouse.exp_update_stim2
        g1_new = g1 + rpe_t * mouse.learning_stim * elig1
        g2_new = g2 + rpe_t * mouse.learning_stim * elig2
        mouse.exp_update_stim1 = (tau1, float(np.clip(g1_new, 0.0, mouse.stim_gain_max)))
        mouse.exp_update_stim2 = (tau2, float(np.clip(g2_new, 0.0, mouse.stim_gain_max)))

    # --- Desapprentissage SPECIFIQUE au stimulus (No-Go), au lick non recompense ------------
    # Miroir exact de la regle ci-dessus, mais sur RPE negatif : un lick declenche par un
    # stimulus qui ne paie plus fait grimper LE No-Go de CE stimulus specifiquement (meme
    # eligibilite que pour le Go, donc meme "credit assignment"). C'est ce qui permet
    # d'eteindre stim1 sans toucher stim2 apres le switch de contingence.
    if mouse.delearning_enable and reward_t == 0.0 and mouse_session.lick[i - 1, 0] == 1:
        elig1 = float(mouse_session.eligibility1[i, 0])
        elig2 = float(mouse_session.eligibility2[i, 0])
        mouse.stim1_nogo_gain = float(np.clip(mouse.stim1_nogo_gain + (-rpe_t) * mouse.learning_stim_nogo * elig1, 0.0, mouse.stim_gain_max))
        mouse.stim2_nogo_gain = float(np.clip(mouse.stim2_nogo_gain + (-rpe_t) * mouse.learning_stim_nogo * elig2, 0.0, mouse.stim_gain_max))

    return mouse, mouse_session


def function_fl_session(
    session_param: FreeLickingSessionParams,
    fl_session: FreeLickingSessionState,
    session_info: SessionBaseInfo,
    lick: int,
    i: int,
) -> tuple[float, float, float, FreeLickingSessionState]:
    """Identique a Code_mono_stim_v2 : pas de stimulus en Free Licking, on renvoie (0, 0) pour
    stim1/stim2 afin de garder un pattern d'appel uniforme avec les sessions WDT/AUD."""
    t = i * session_info.resolution
    reward = 0.0

    if session_param.forced_reward[0] == 1 and np.isclose(t, session_param.forced_reward[1], atol=1e-6):
        reward = session_param.reward_size
        fl_session.last_reward_time = t

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
    fl_session.stim1[i, 0] = 0.0
    fl_session.stim2[i, 0] = 0.0

    return reward, 0.0, 0.0, fl_session


def function_wdt_session(
    session_param: WDTSesssionParams,
    wdt_session: WDTSesssionState,
    session_info: SessionBaseInfo,
    lick: int,
    i: int,
) -> tuple[float, float, float, WDTSesssionState, bool]:
    """
    Comme Code_mono_stim_v2::function_wdt_session, mais avec 3 types de trial (0=catch,
    1=whisker, 2=auditif) au lieu de 2. Seul le lick pendant la fenetre de reponse ET dont
    le type de trial correspond a `session_param.reward_stim` est recompense — un lick sur
    l'AUTRE stimulus (celui qui n'est plus recompense apres le switch de contingence) n'est
    jamais recompense, exactement comme un lick de catch trial.
    """
    t = i * session_info.resolution
    reward = 0.0
    stim1 = 0.0
    stim2 = 0.0
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
            stim1 = session_param.stim1_amp
            wdt_session.last_stim_time = t
        elif kind == 2:
            stim2 = session_param.stim2_amp
            wdt_session.last_stim_time = t

        if kind in (1, 2):
            window_length = int(round(session_param.response_wind / session_info.resolution))
            end_idx = min(i + window_length, session_info.number_bin)
            wdt_session.reward_window[i + 1:end_idx, 0] = 1  # exclude current bin

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

    wdt_session.reward[i, 0] = reward
    if stim1 > 0:
        wdt_session.stim1[i, 0] = stim1
    if stim2 > 0:
        wdt_session.stim2[i, 0] = stim2

    return reward, stim1, stim2, wdt_session, new_trial


def function_performance_wdt(
    wdt_params: WDTSesssionParams,
    session_info: SessionBaseInfo,
    wdt_session: WDTSesssionState,
    mouse_session: MouseSessionState,
) -> np.ndarray:
    """
    Comme Code_mono_stim_v2::function_performance_wdt, avec une colonne stim_code qui vaut
    0 (catch), 1 (whisker) ou 2 (auditif) au lieu d'une simple amplitude — pour pouvoir
    calculer un Hit Rate SEPARE par stimulus (plot_session_rates s'en sert pour montrer
    l'inversion de contingence). outcome reste Miss/Hit/CR/FA base sur la presence d'UN
    stimulus quelconque (comme l'ancien Code_dual_stim) ; c'est stim_code qui dit lequel.
    """
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
            outcome = 1  # Hit (repond au stimulus, qu'il soit recompense ou non ce moment-la)
        elif stim_code == 0 and not lick_detected:
            outcome = 2  # Correct Rejection
        else:
            outcome = 3  # False Alarm

        performance.append([t, stim_code, int(lick_detected), latency, int(reward_detected), outcome])

    return np.array(performance, dtype=float)


def extract_rpe_per_lick(mouse_session: MouseSessionState,
                         reward_trace: np.ndarray,
                         stim_trace: Optional[np.ndarray],
                         session_info: SessionBaseInfo,
                         ignore_last_without_next: bool = True
                        ) -> np.ndarray:
    """Identique a Code_mono_stim_v2 : stim_trace peut etre stim1, stim2, ou une combinaison
    (stim1 + 2*stim2) fournie par l'appelant pour visualiser les deux canaux sur un plot."""
    lick = np.asarray(mouse_session.lick).ravel()
    expc = np.asarray(mouse_session.expectation).ravel()
    rew = np.asarray(reward_trace).ravel()
    stim = np.asarray(stim_trace).ravel() if stim_trace is not None else None

    idxs = np.flatnonzero(lick == 1)
    if ignore_last_without_next:
        idxs = idxs[idxs + 1 < len(rew)]

    if idxs.size == 0:
        return np.zeros((0, 6), dtype=float)

    rpe_vals = rew[idxs + 1] - expc[idxs]
    reward_flag = (rew[idxs + 1] > 0).astype(float)
    stim_amp = (stim[idxs] if stim is not None else np.zeros_like(idxs, dtype=float)).astype(float)
    t_sec = idxs.astype(float) * float(session_info.resolution)
    lick_idx = np.arange(1, len(idxs) + 1, dtype=float)

    return np.column_stack([lick_idx, t_sec, rpe_vals.astype(float), reward_flag, stim_amp, idxs.astype(float)])


# =============================================================================================
# Fonctions d'orchestration appelees par main.py — meme pattern que Code_mono_stim_v2/functions.py
# (initialization/run_FL1/run_FL2/run_all_wdt/run_wdt_test/plot_all_results/save_run_parameters).
# =============================================================================================

def initialization(config: SimConfig) -> tuple[SessionBaseInfo, Mouse]:
    """Cree le dossier de resultats et l'etat global (session_info, mouse)."""
    suffix = (
        f"V{config.REWARD_VALUE:g}_C{config.COST:g}_wdtThrs{config.LICK_THRS_WDT:g}"
        + ("_delearn" if config.DELEARNING else "")
    )
    init_results_dir(suffix=suffix)

    session_info = SessionBaseInfo()
    mouse = Mouse(
        reward_value=config.REWARD_VALUE,
        cost=config.COST,
        lick_thrs=config.LICK_THRS_FL,
        lick_thrs_wdt=config.LICK_THRS_WDT,
        delearning_enable=config.DELEARNING,
        exp_update_nogo=(config.NOGO_TAU, config.NOGO_GAIN),
        nogo_relief=config.NOGO_RELIEF,
        learning_stim=config.LEARNING_STIM,
        learning_stim_nogo=config.LEARNING_STIM_NOGO,
    )
    return session_info, mouse


def run_FL1(mouse: Mouse, session_info: SessionBaseInfo, config: SimConfig) -> FLSessionResult:
    """Free Licking — Session 1 (avec forced reward a t=500s)."""
    fl_params = FreeLickingSessionParams()
    fl_session = FreeLickingSessionState()
    fl_session.initialize(session_info.number_bin, fl_params.no_lick_wind)
    mouse_session = MouseSessionState()
    mouse_session.initialize(session_info.number_bin, mouse.motivation[0])

    for i in range(1, session_info.number_bin):
        reward_t, s1, s2, fl_session = function_fl_session(
            fl_params, fl_session, session_info, int(mouse_session.lick[i - 1, 0]), i
        )
        _, mouse_session = function_mouse_lick(mouse, mouse_session, i)
        mouse, mouse_session = function_update_mouse_state(mouse, mouse_session, session_info, i, s1, s2, reward_t)

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

    for i in range(1, session_info.number_bin):
        reward_t, s1, s2, fl_session = function_fl_session(
            fl_params, fl_session, session_info, int(mouse_session.lick[i - 1, 0]), i
        )
        _, mouse_session = function_mouse_lick(mouse, mouse_session, i)
        mouse, mouse_session = function_update_mouse_state(mouse, mouse_session, session_info, i, s1, s2, reward_t)

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
    """Corps d'une session WDT/AUD (reutilisee pour WDT1..N ET pour WDT_TEST)."""
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
        reward_t, s1, s2, wdt_session, _ = function_wdt_session(
            wdt_params, wdt_session, session_info, int(mouse_session.lick[i - 1, 0]), i
        )
        _, mouse_session = function_mouse_lick(
            mouse, mouse_session, i, threshold=mouse.lick_thrs_wdt, decision_slope=mouse.decision_slope_wdt
        )
        mouse, mouse_session = function_update_mouse_state(mouse, mouse_session, session_info, i, s1, s2, reward_t)

    perf = function_performance_wdt(wdt_params, session_info, wdt_session, mouse_session)

    return WDTRunResult(
        label=label, session=wdt_session, mouse_session=mouse_session, perf=perf,
        stim1_gain=float(mouse.exp_update_stim1[1]), stim2_gain=float(mouse.exp_update_stim2[1]),
        stim1_nogo_gain=float(mouse.stim1_nogo_gain), stim2_nogo_gain=float(mouse.stim2_nogo_gain),
    )


def run_all_wdt(mouse: Mouse, session_info: SessionBaseInfo, config: SimConfig, log_fl2: FLSessionResult) -> WDTBundle:
    """
    Enchaine WDT1..WDT{config.NUM_WDT}. Sessions 1..SWITCH_SESSION-1 : stim1 (whisker)
    recompense, etiquette "WDT{k}". Sessions SWITCH_SESSION..NUM_WDT : stim2 (auditif)
    recompense, etiquette "AUD{k}" — exactement le paradigme de l'ancien Code_dual_stim.
    """
    bundle = WDTBundle()
    prev_mouse_session = log_fl2.mouse_session
    prev_reward = log_fl2.session.reward

    for s_idx in range(1, config.NUM_WDT + 1):
        wdt_params = WDTSesssionParams(
            trial_kinds=config.TRIAL_KINDS[:],
            stim1_amp=config.STIM1_AMP,
            stim2_amp=config.STIM2_AMP,
            reward_stim=1 if s_idx < config.SWITCH_SESSION else 2,
        )
        label = f"WDT{s_idx}" if s_idx < config.SWITCH_SESSION else f"AUD{s_idx}"
        result = run_wdt_session(mouse, session_info, config, wdt_params, prev_mouse_session, prev_reward, label)

        if result.perf.size > 0:
            stim1_mask = result.perf[:, 1] == 1
            stim2_mask = result.perf[:, 1] == 2
            catch_mask = result.perf[:, 1] == 0
            hr1 = (np.sum((result.perf[:, 5] == 1) & stim1_mask) / np.sum(stim1_mask)) if np.sum(stim1_mask) > 0 else 0.0
            hr2 = (np.sum((result.perf[:, 5] == 1) & stim2_mask) / np.sum(stim2_mask)) if np.sum(stim2_mask) > 0 else 0.0
            fa = (np.sum((result.perf[:, 5] == 3) & catch_mask) / np.sum(catch_mask)) if np.sum(catch_mask) > 0 else 0.0
        else:
            hr1, hr2, fa = 0.0, 0.0, 0.0

        print(f"Hit Rate {label} : stim1(whisker)={hr1:.2f}  stim2(auditif)={hr2:.2f}  FA={fa:.2f}"
              f"  |  gains go(stim1,stim2)=({result.stim1_gain:.3f},{result.stim2_gain:.3f})"
              + (f"  nogo(stim1,stim2)=({result.stim1_nogo_gain:.3f},{result.stim2_nogo_gain:.3f})" if config.DELEARNING else ""))

        bundle.results.append(result)
        bundle.labels.append(label)

        prev_mouse_session = result.mouse_session
        prev_reward = result.session.reward

    return bundle


def plot_all_results(
    session_info: SessionBaseInfo,
    mouse: Mouse,
    config: SimConfig,
    log_fl1: FLSessionResult,
    log_fl2: FLSessionResult,
    wdt_bundle: WDTBundle,
) -> None:
    """Tous les plots, meme pattern que Code_mono_stim_v2::plot_all_results."""

    if config.PLOT_TRACES:
        plot_traces(session_info.time_vector, log_fl1.mouse_session, log_fl1.session.reward,
                    stim1_array=None, stim2_array=None, title="Free Licking — Session 1",
                    save_name="free_licking_1", threshold=mouse.lick_thrs)
        plot_traces(session_info.time_vector, log_fl2.mouse_session, log_fl2.session.reward,
                    stim1_array=None, stim2_array=None, title="Free Licking — Session 2",
                    save_name="free_licking_2", threshold=mouse.lick_thrs)
        for lbl, result in zip(wdt_bundle.labels, wdt_bundle.results):
            plot_traces(session_info.time_vector, result.mouse_session, result.session.reward,
                        stim1_array=result.session.stim1, stim2_array=result.session.stim2,
                        title=f"Two-Stimulus Detection — {lbl}",
                        save_name=lbl.lower(), threshold=mouse.lick_thrs_wdt)

    if config.PLOT_BLOCK_HRFA:
        _wdt_params = WDTSesssionParams()
        for lbl, result in zip(wdt_bundle.labels, wdt_bundle.results):
            plot_wdt_block_rates(result.perf, _wdt_params, max_trials=config.MAX_TRIALS_BLOCKS,
                                 title=f"{lbl} — HR (stim1/stim2) & FA by block", save_name=lbl.lower())

    if config.PLOT_SESSIONS_COMPARISON:
        wdt_perfs = [r.perf for r in wdt_bundle.results]
        plot_session_rates(wdt_perfs, wdt_bundle.labels,
                            title="Learning — HR(stim1/stim2) & FA(catch) across sessions",
                            save_name="hr_fa_all_sessions", switch_session=config.SWITCH_SESSION)

    if config.PLOT_SINGLE_SESSION_ALL:
        for lbl, result in zip(wdt_bundle.labels, wdt_bundle.results):
            plot_single_session_rates(result.perf, title=f"{lbl} — HR(stim1/stim2) vs FA", save_name=lbl.lower())

    if config.PLOT_STIM_GAINS:
        plot_stim_gains(wdt_bundle, config, title="Evolution des gains de stimuli (Go — No-Go)",
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
        ("Lick threshold — WDT/AUD", config.LICK_THRS_WDT),

        ("MOUSE — MOTIVATION", None),
        ("Motivation (initial)", _ref_mouse.motivation[0]),
        ("Motivation loss per reward", _ref_mouse.motivation[1]),

        ("MOUSE — UNCERTAINTY (DECISION BLEND E/Nd)", None),
        ("Decision slope — Free Licking (A_D)", _ref_mouse.decision_slope),
        ("Decision slope — WDT/AUD (A_D)", _ref_mouse.decision_slope_wdt),
        ("Uncertainty gain (A_U)", _ref_mouse.uncertainty_gain),
        ("Uncertainty max (U_max)", _ref_mouse.uncertainty_max),

        ("MOUSE — VALUE & COST (DECISION GATE: V·M − C)", None),
        ("Reward value (V)", config.REWARD_VALUE),
        ("Cost (C)", config.COST),

        ("MOUSE — EXPECTATION UPDATE (REWARD)", None),
        ("Tau (reward, s)", _ref_mouse.exp_update_reward[0]),
        ("Gain (reward)", _ref_mouse.exp_update_reward[1]),

        ("MOUSE — EXPECTATION UPDATE (NO REWARD)", None),
        ("Tau (no reward, initial, s)", _ref_mouse.exp_update_no_reward[0]),
        ("Gain (no reward)", _ref_mouse.exp_update_no_reward[1]),

        ("MOUSE — STIMULI (WHISKER=1, AUDITIF=2)", None),
        ("Tau (stimulus, s)", _ref_mouse.exp_update_stim1[0]),
        ("Gain initial (stim1, whisker)", _ref_mouse.exp_update_stim1[1]),
        ("Gain initial (stim2, auditif)", _ref_mouse.exp_update_stim2[1]),
        ("Stimulus gain learning rate (Go)", config.LEARNING_STIM),
        ("Stimulus gain learning rate (No-Go)", config.LEARNING_STIM_NOGO),
        ("Stimulus gain cap (go et no-go)", _ref_mouse.stim_gain_max),
        ("Eligibility trace tau (s)", _ref_mouse.tau_eligibility),

        ("MOUSE — LEARNING RULES (GENERALES)", None),
        ("Consecutive non-rewarded licks threshold", _ref_mouse.learning_nonrew_lick[0]),
        ("Tau (no reward) increment", _ref_mouse.learning_nonrew_lick[1]),

        ("MOUSE — DESAPPRENTISSAGE GENERAL (GO/NO-GO)", None),
        ("Enabled", config.DELEARNING),
        ("Tau (no-go, s)", config.NOGO_TAU),
        ("Gain (no-go)", config.NOGO_GAIN),
        ("Relief (fraction effacee par recompense)", config.NOGO_RELIEF),

        ("MOUSE — DESAPPRENTISSAGE SPECIFIQUE AU STIMULUS", None),
        ("Enabled (meme flag que ci-dessus)", config.DELEARNING),
        ("Learning rate (meme que le gain Go)", config.LEARNING_STIM),

        ("FREE LICKING SESSIONS", None),
        ("No-lick window (s)", f"{_ref_fl.no_lick_wind[0]}–{_ref_fl.no_lick_wind[1]}"),
        ("Reward probability", _ref_fl.reward_prob),
        ("Reward size", _ref_fl.reward_size),
        ("Forced reward — Session 1 (s)", _ref_fl.forced_reward[1]),
        ("Forced reward — Session 2", "none"),

        ("TWO-STIMULUS DETECTION TASK (SWITCH DE CONTINGENCE)", None),
        ("Number of sessions", config.NUM_WDT),
        ("Sessions 1..N recompensant stim1 (whisker)", config.SWITCH_SESSION - 1),
        ("Sessions N..fin recompensant stim2 (auditif)", config.NUM_WDT - config.SWITCH_SESSION + 1),
        ("No-lick window (s)", f"{_ref_wdt.no_lick_wind[0]}–{_ref_wdt.no_lick_wind[1]}"),
        ("Inter-trial interval (s)", f"{_ref_wdt.iti[0]}–{_ref_wdt.iti[1]}"),
        ("Response window (s)", _ref_wdt.response_wind),
        ("Reward probability", _ref_wdt.reward_prob),
        ("Reward size", _ref_wdt.reward_size),
        ("Stim1 (whisker) amplitude", config.STIM1_AMP),
        ("Stim2 (auditif) amplitude", config.STIM2_AMP),
        ("Block size (trials)", _ref_wdt.block_numb),
    ]

    save_parameters_txt(param_rows)
