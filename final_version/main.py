# Deroulement : 2 sessions Free Licking + 10 WDT + test psychometrique (test psychometrique
# absent en whisker/auditif, qui a son propre switch de contingence, cf. WA_SWITCH_SESSION).
# Trois modes, mutuellement exclusifs (WHISKER_AUD_STIM a priorite sur dual_stim si les deux
# sont actives par erreur), selon les flags de SimConfig tout en bas :
#   - dual_stim=False, WHISKER_AUD_STIM=False : un seul stimulus (mono, modele valide).
#   - dual_stim=True                          : deux stimuli, deux cotes de recompense
#                                                (deux Expectations independantes, choix de cote).
#   - WHISKER_AUD_STIM=True                   : deux stimuli (whisker/auditif), UNE SEULE
#                                                Expectation partagee, switch de contingence
#                                                WDT->AUD (paradigme historique du projet).
# La decision de base utilise toujours E_go - E_nogo (architecture Go/No-Go), a l'identique
# dans les trois modes (une voie No-Go par cote en dual gauche/droite, une voie No-Go par
# stimulus en whisker/auditif, une voie generale en mono). Pour observer un desapprentissage
# sur un run normal (mono/dual), fixer DELEARNING_FROM_SESSION (ex. 6) : plus aucune recompense
# a partir de WDT6, et 3 plots de diagnostic en plus des plots habituels.
# Pour comparer une population de souris (mono/dual uniquement, learning_stim / noise tires
# +/- autour de la base, pas de plots individuels), fixer POPULATION_RANGE=True et
# POPULATION_N_MICE.

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
        log_fl1 = run_FL1(mouse, session_info, config)      # FL n'a pas de stimulus : la version
        log_fl2 = run_FL2(mouse, session_info, config, log_fl1)  # mono suffit, rien a adapter.

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
