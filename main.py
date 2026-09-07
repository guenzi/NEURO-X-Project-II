# Construit un SimConfig et appelle la fonction d'orchestration correspondante.
#
# Scenarios
#   mono                                SimConfig()
#   dual                                SimConfig(dual_stim=True)
#   whisker/auditif                     SimConfig(WHISKER_AUD_STIM=True)
#   mono, desapprentissage              SimConfig(DELEARNING_FROM_SESSION=6) --> choisir la session de début de desapprentissage (ici 6 par exemple)
#   dual, desapprentissage droite       SimConfig(dual_stim=True, DELEARNING_FROM_SESSION=6, DELEARNING_SIDE=1) --> sessions à choix
#   desapprentissage puis reapprentissage   + DELEARNING_UNTIL_SESSION=7 (avec DELEARNING_FROM_SESSION=5) --> sessions à choix
#   population mono                     SimConfig(POPULATION_RANGE=True, POPULATION_N_MICE=5) --> range de parametres sur 5 souris, parametres: 
#   population, balayage p0..p50        + POPULATION_SPREAD_SWEEP=True --> balayage de range de 0% à 50% sur 5 souris, avec 10 répétitions par point de balayage (50 simulations au total)

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
    run_all_wdt_wa,
    plot_all_results_wa,
    save_run_parameters,
    save_run_parameters_wa,
    run_population,
)

config = SimConfig(dual_stim=True, POPULATION_RANGE=True, POPULATION_N_MICE=5, POPULATION_SPREAD_SWEEP=True)

if config.POPULATION_RANGE:
    run_population(config)
else:
    session_info, mouse = initialization(config)

    if config.WHISKER_AUD_STIM:
        log_fl1 = run_FL1(mouse, session_info, config)
        log_fl2 = run_FL2(mouse, session_info, config, log_fl1)

        wdt_bundle = run_all_wdt_wa(mouse, session_info, config, log_fl2)

        plot_all_results_wa(session_info, mouse, config, log_fl1, log_fl2, wdt_bundle)

        if config.SAVE_PARAMETERS_TXT:
            save_run_parameters_wa(config, mouse, session_info)

    else:
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
