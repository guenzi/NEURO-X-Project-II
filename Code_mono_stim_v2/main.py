# Deroulement : 2 sessions Free Licking + 10 WDT + test psychometrique.
# Deux modes : dual_stim=False (un seul stimulus, modele valide) ou dual_stim=True
# (deux stimuli, deux cotes de recompense) selon config.dual_stim, tout en bas.
# Independamment : DELEARNING=True active l'architecture Go/No-Go (desapprentissage),
# qui fonctionne a l'identique en mono et en dual stim (une voie No-Go par cote en dual).

from models import SimConfig
from functions import (
    initialization,
    run_FL1,
    run_FL2,
    run_all_wdt,
    run_wdt_test,
    plot_all_results,
    run_FL1_dualstim,
    run_FL2_dualstim,
    run_all_wdt_dualstim,
    run_wdt_test_dualstim,
    plot_all_results_dualstim,
    save_run_parameters,
)

config = SimConfig(dual_stim=True, DELEARNING=False)

session_info, mouse = initialization(config)

if config.dual_stim:
    log_fl1 = run_FL1_dualstim(mouse, session_info, config)
    log_fl2 = run_FL2_dualstim(mouse, session_info, config, log_fl1)

    wdt_bundle = run_all_wdt_dualstim(mouse, session_info, config, log_fl2)
    wdt_test = run_wdt_test_dualstim(mouse, session_info, config, wdt_bundle)

    plot_all_results_dualstim(session_info, mouse, config, log_fl1, log_fl2, wdt_bundle, wdt_test)
else:
    log_fl1 = run_FL1(mouse, session_info, config)
    log_fl2 = run_FL2(mouse, session_info, config, log_fl1)

    wdt_bundle = run_all_wdt(mouse, session_info, config, log_fl2)
    wdt_test = run_wdt_test(mouse, session_info, config, wdt_bundle)

    plot_all_results(session_info, mouse, config, log_fl1, log_fl2, wdt_bundle, wdt_test)

if config.SAVE_PARAMETERS_TXT:
    save_run_parameters(config, mouse, session_info)
