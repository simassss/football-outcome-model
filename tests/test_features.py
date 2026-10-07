"""
Tests for the feature building.

The most important test: features of a match must NOT depend on that match's
own result (otherwise the model would "see the future").

Run:  python -m pytest
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main


def make_matches(last_home_goals):
    # 4 teams play each other, the last match is the one we change
    rows = []
    teams = ["A", "B", "C", "D"]
    day = 1
    for round_number in range(4):
        for k in range(0, 4, 2):
            home = teams[(k + round_number) % 4]
            away = teams[(k + 1 + round_number) % 4]
            rows.append({"HomeTeam": home, "AwayTeam": away,
                         "FTHG": 1, "FTAG": 1, "Date": day})
            day = day + 1
    rows[-1]["FTHG"] = last_home_goals
    return pd.DataFrame(rows)


def test_features_do_not_depend_on_own_result():
    data_a = make_matches(0)
    data_b = make_matches(5)
    features_a = main.build_features(data_a)
    features_b = main.build_features(data_b)
    last = len(data_a) - 1
    if last in features_a.index:
        assert features_a.loc[last].equals(features_b.loc[last])


def test_first_matches_have_no_features():
    # teams need MIN_MATCHES past matches before they get features
    data = make_matches(1)
    features = main.build_features(data)
    assert 0 not in features.index


def test_points():
    assert main.get_points(2, 1) == 3
    assert main.get_points(1, 1) == 1
    assert main.get_points(0, 3) == 0


def test_elo_is_zero_sum():
    new_home, new_away = main.elo_update(1500, 1500, 2, 0)
    assert abs((new_home + new_away) - 3000) < 1e-9
    assert new_home > 1500


def test_clubelo_name():
    assert main.clubelo_name("Man City") == "ManCity"
    assert main.clubelo_name("Crystal Palace") == "CrystalPalace"
    assert main.clubelo_name("Arsenal") == "Arsenal"


def test_clubelo_rating_uses_valid_date_range():
    table = pd.DataFrame({
        "Elo": [1600.0, 1650.0],
        "From": pd.to_datetime(["2022-01-01", "2022-02-01"]),
        "To": pd.to_datetime(["2022-01-31", "2022-02-28"]),
    })
    assert main.clubelo_rating(table, pd.Timestamp("2022-01-15")) == 1600.0
    assert main.clubelo_rating(table, pd.Timestamp("2022-02-10")) == 1650.0
    assert pd.isna(main.clubelo_rating(table, pd.Timestamp("2021-12-01")))


def test_poisson_probabilities_sum_to_one():
    outcome_probs, over_probs = main.poisson_probabilities(np.array([1.5, 0.8]), np.array([1.1, 2.0]))
    assert abs(outcome_probs[0].sum() - 1.0) < 1e-9
    assert abs(outcome_probs[1].sum() - 1.0) < 1e-9
    assert (over_probs > 0).all() and (over_probs < 1).all()


def test_poisson_equal_rates_are_symmetric():
    outcome_probs, over_probs = main.poisson_probabilities(np.array([1.3]), np.array([1.3]))
    assert abs(outcome_probs[0, 0] - outcome_probs[0, 2]) < 1e-9


def test_poisson_stronger_home_team_is_favourite():
    outcome_probs, over_probs = main.poisson_probabilities(np.array([2.5]), np.array([0.6]))
    assert outcome_probs[0, 0] > outcome_probs[0, 1]
    assert outcome_probs[0, 0] > outcome_probs[0, 2]


def test_elo_differences_match_build_features():
    # the fast Elo function and the feature builder must agree
    data = make_matches(1)
    features = main.build_features(data, 30, 80)
    diffs = main.elo_differences(data, 30, 80)
    for i in features.index:
        assert abs(features.loc[i, "elo_diff"] - diffs[i]) < 1e-9


def test_tune_elo_needs_two_earlier_seasons():
    data = make_matches(1)
    # "1112" is the second season, so only ONE season lies before it
    data["Season"] = main.SEASONS[0]
    data["FTR"] = "D"
    try:
        main.tune_elo(data, main.SEASONS[1])
        assert False
    except ValueError:
        pass


def test_get_test_seasons():
    assert main.get_test_seasons("2425") == ["2425"]
    assert main.get_test_seasons("2324") == ["2324", "2425"]
    assert main.get_test_seasons(main.SEASONS[2])[0] == main.SEASONS[2]
    for bad in ["1011", "1112", "9999"]:      # too early or unknown
        try:
            main.get_test_seasons(bad)
            assert False
        except ValueError:
            pass


def test_bookmaker_probabilities_sum_to_one():
    df = pd.DataFrame({"B365H": [2.0], "B365D": [3.5], "B365A": [4.0]})
    probs = main.bookmaker_probabilities(df)
    assert abs(probs.sum() - 1.0) < 1e-9


def test_bootstrap_identical_probabilities_give_zero_difference():
    probs = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5], [0.4, 0.4, 0.2]])
    y = np.array([0, 2, 1])
    mean_diff, low, high = main.bootstrap_loss_difference(probs, probs, y, n_resamples=100)
    assert mean_diff == 0
    assert low == 0 and high == 0


def test_bootstrap_interval_excludes_zero_for_clearly_worse_model():
    # the "bookmaker" is always confident and right, the "model" always says 1/3 each
    n = 200
    y = np.array([0, 1, 2, 0] * 50)
    book = np.full((n, 3), 0.05)
    for i in range(n):
        book[i, y[i]] = 0.90
    model = np.full((n, 3), 1 / 3)
    mean_diff, low, high = main.bootstrap_loss_difference(model, book, y, n_resamples=200)
    assert low <= mean_diff <= high
    assert low > 0                 # model is worse and the interval says so


def test_bootstrap_is_repeatable_with_same_seed():
    rng = np.random.default_rng(1)
    model = rng.dirichlet([2, 2, 2], size=50)
    book = rng.dirichlet([2, 2, 2], size=50)
    y = rng.integers(0, 3, size=50)
    first = main.bootstrap_loss_difference(model, book, y, n_resamples=100, seed=7)
    second = main.bootstrap_loss_difference(model, book, y, n_resamples=100, seed=7)
    assert first == second


def test_blend_probabilities_sum_to_one_and_ends():
    model = np.array([[0.6, 0.3, 0.1]])
    book = np.array([[0.2, 0.3, 0.5]])
    blended = main.blend_probabilities(model, book, 0.3)
    assert abs(blended.sum() - 1.0) < 1e-9
    assert np.allclose(main.blend_probabilities(model, book, 0.0), book)
    assert np.allclose(main.blend_probabilities(model, book, 1.0), model)


def test_choose_blend_weight_ignores_useless_model():
    # bookmaker is right, the model is a constant guess -> weight 0
    y = np.array([0, 1, 2, 0, 1, 2])
    book = np.full((6, 3), 0.05)
    for i in range(6):
        book[i, y[i]] = 0.90
    model = np.full((6, 3), 1 / 3)
    assert main.choose_blend_weight(model, book, y) == 0.0


def test_choose_blend_weight_uses_good_model():
    # the model is right, the "bookmaker" is a constant guess -> weight 1
    y = np.array([0, 1, 2, 0, 1, 2])
    model = np.full((6, 3), 0.05)
    for i in range(6):
        model[i, y[i]] = 0.90
    book = np.full((6, 3), 1 / 3)
    assert main.choose_blend_weight(model, book, y) == 1.0


def test_rps_perfect_forecast_is_zero():
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    y = np.array([0, 1, 2])
    assert np.allclose(main.rps_per_match(probs, y), 0.0)


def test_rps_known_values():
    # uniform forecast, home win: ((1/3 - 1)^2 + (2/3 - 1)^2) / 2 = 5/18
    uniform = np.array([[1 / 3, 1 / 3, 1 / 3]])
    assert abs(main.rps_per_match(uniform, np.array([0]))[0] - 5 / 18) < 1e-12
    # certain draw forecast when the home team wins: (1 + 0) / 2 = 0.5
    draw = np.array([[0.0, 1.0, 0.0]])
    assert abs(main.rps_per_match(draw, np.array([0]))[0] - 0.5) < 1e-12


def test_rps_respects_the_order_of_outcomes():
    # home win happened. Predicting a draw must be less wrong than predicting an away win.
    y = np.array([0])
    draw = main.rps_per_match(np.array([[0.0, 1.0, 0.0]]), y)[0]
    away = main.rps_per_match(np.array([[0.0, 0.0, 1.0]]), y)[0]
    assert draw < away
    assert abs(away - 1.0) < 1e-12


def test_diebold_mariano_identical_forecasts_have_no_difference():
    differences = np.zeros(100)
    statistic, p_value = main.diebold_mariano(differences)
    assert statistic == 0.0
    assert p_value == 1.0


def test_diebold_mariano_detects_a_clearly_worse_model():
    # the model is worse by about 0.05 in nearly every match
    rng = np.random.default_rng(3)
    differences = 0.05 + 0.01 * rng.standard_normal(200)
    statistic, p_value = main.diebold_mariano(differences)
    assert statistic > 0
    assert p_value < 0.001


def test_diebold_mariano_symmetric_differences_are_not_significant():
    differences = np.array([0.2, -0.2] * 50)       # mean exactly 0
    statistic, p_value = main.diebold_mariano(differences)
    assert abs(statistic) < 1e-12
    assert p_value > 0.99


def test_diebold_mariano_sign_follows_the_better_model():
    rng = np.random.default_rng(4)
    differences = -0.05 + 0.02 * rng.standard_normal(200)    # model is better
    statistic, p_value = main.diebold_mariano(differences)
    assert statistic < 0
    assert p_value < 0.001


# odds used by the margin tests: a balanced match, a strong favourite, and an away favourite
MARGIN_TEST_ODDS = pd.DataFrame({"B365H": [2.60, 1.20, 6.50],
                                 "B365D": [3.20, 7.00, 4.00],
                                 "B365A": [2.80, 15.0, 1.60]})


def test_all_margin_methods_return_valid_probabilities():
    for method in main.MARGIN_METHODS:
        probs = main.bookmaker_probabilities(MARGIN_TEST_ODDS, method)
        assert probs.shape == (3, 3)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-9)
        assert (probs > 0).all() and (probs < 1).all()


def test_margin_methods_keep_the_order_of_the_odds():
    # lower odds must always mean a higher probability
    for method in main.MARGIN_METHODS:
        probs = main.bookmaker_probabilities(MARGIN_TEST_ODDS, method)
        odds = MARGIN_TEST_ODDS.values
        for i in range(3):
            assert (np.argsort(odds[i]) == np.argsort(-probs[i])).all()


def test_proportional_method_by_hand():
    odds = pd.DataFrame({"B365H": [2.0], "B365D": [4.0], "B365A": [4.0]})
    # implied 0.5, 0.25, 0.25 add up to exactly 1.0 -> unchanged
    probs = main.bookmaker_probabilities(odds, "proportional")
    assert np.allclose(probs[0], [0.5, 0.25, 0.25])
    odds = pd.DataFrame({"B365H": [1.8], "B365D": [3.6], "B365A": [3.6]})
    # implied 0.5556, 0.2778, 0.2778 add up to 1.1111 -> divide by 1.1111
    probs = main.bookmaker_probabilities(odds, "proportional")
    assert np.allclose(probs[0], [0.5, 0.25, 0.25])


def test_methods_leave_odds_without_margin_unchanged():
    odds = pd.DataFrame({"B365H": [2.0], "B365D": [1 / 0.3], "B365A": [5.0]})   # 0.5 + 0.3 + 0.2
    for method in main.MARGIN_METHODS:
        probs = main.bookmaker_probabilities(odds, method)
        assert np.allclose(probs[0], [0.5, 0.3, 0.2], atol=1e-9)


def test_power_and_shin_shrink_longshots_more_than_favourites():
    # compare with the implied probabilities: the share that is removed
    # must be larger for the longshot (odds 15) than for the favourite (odds 1.2)
    odds = MARGIN_TEST_ODDS.iloc[[1]]
    implied = 1 / odds.values[0]
    for method in ["power", "shin"]:
        probs = main.bookmaker_probabilities(odds, method)[0]
        share_kept_favourite = probs[0] / implied[0]
        share_kept_longshot = probs[2] / implied[2]
        assert share_kept_longshot < share_kept_favourite


def test_power_method_solves_its_equation():
    implied = 1 / np.array([2.60, 3.20, 2.80])
    probs = main.power_row(np.array([2.60, 3.20, 2.80]))
    # probs = implied ^ k with k >= 1, and they add up to 1
    k = np.log(probs[0]) / np.log(implied[0])
    assert k >= 1.0
    assert np.allclose(implied ** k, probs)
    assert abs(probs.sum() - 1.0) < 1e-9


def test_shin_formula_without_insiders():
    # with z = 0 the formula gives implied / sqrt(sum); the sum is sqrt(booksum) > 1,
    # so a positive z is needed to bring it down to exactly 1
    implied = 1 / np.array([2.60, 3.20, 2.80])
    assert abs(main.shin_probabilities_for(0.0, implied).sum() - np.sqrt(implied.sum())) < 1e-9
    assert main.shin_gap(0.0, implied) > 0
    assert main.shin_gap(0.999, implied) < 0


def test_unknown_margin_method_is_rejected():
    try:
        main.bookmaker_probabilities(MARGIN_TEST_ODDS, "magic")
        assert False
    except ValueError:
        pass


def test_favourite_longshot_summary_and_calibration_shapes():
    # tiny synthetic season: odds from MARGIN_TEST_ODDS repeated, results all home wins
    data = pd.concat([MARGIN_TEST_ODDS] * 40, ignore_index=True)
    data["FTR"] = "H"
    summary = main.favourite_longshot_summary(data, "proportional")
    assert summary["observations"].sum() == 3 * len(data)       # every probability is in one group
    table = main.bookmaker_calibration_table(data, "proportional")
    assert (table["matches"] >= main.CALIBRATION_MIN_MATCHES).all()
    assert ((table["actual_frequency"] >= 0) & (table["actual_frequency"] <= 1)).all()


def test_margin_methods_worked_example_from_the_readme():
    odds = pd.DataFrame({"B365H": [1.20], "B365D": [7.00], "B365A": [15.0]})
    proportional = main.bookmaker_probabilities(odds, "proportional")[0]
    power = main.bookmaker_probabilities(odds, "power")[0]
    shin = main.bookmaker_probabilities(odds, "shin")[0]
    assert np.allclose(proportional, [0.7991, 0.1370, 0.0639], atol=5e-5)
    assert np.allclose(power, [0.8220, 0.1235, 0.0544], atol=5e-5)
    assert np.allclose(shin, [0.8139, 0.1305, 0.0556], atol=5e-5)
    implied = 1 / odds.values[0]
    k = np.log(power[0]) / np.log(implied[0])
    assert abs(k - 1.0748) < 5e-5
    z = brentq(main.shin_gap, 0.0, 0.999, args=(implied,))
    assert abs(z - 0.0224) < 5e-5


# ---------------------------------------------------------------
# Dixon-Coles tests
# ---------------------------------------------------------------
def make_dc_league(seed, n_rounds, league="L1", first_day="2020-01-01"):
    # synthetic league: 8 teams with KNOWN strengths, double round robin repeated n_rounds
    # times, one match per day. Goals are drawn from the Dixon-Coles scoreline matrix.
    rng = np.random.default_rng(seed)
    teams = ["T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7"]
    attack = np.array([0.4, 0.25, 0.1, 0.0, -0.05, -0.15, -0.25, -0.3])
    defence = np.array([-0.2, -0.1, 0.0, 0.05, 0.05, 0.1, 0.1, 0.0]) + 0.25
    home_adv = 0.3
    rho = -0.1
    rows = []
    day = pd.Timestamp(first_day)
    for round_number in range(n_rounds):
        for h in range(8):
            for a in range(8):
                if h == a:
                    continue
                lam = np.exp(attack[h] + defence[a] + home_adv)
                mu = np.exp(attack[a] + defence[h])
                matrix = main.dc_scoreline_matrix(lam, mu, rho)
                cell = rng.choice(matrix.size, p=matrix.reshape(-1))
                rows.append({"League": league, "Date": day, "HomeTeam": teams[h],
                             "AwayTeam": teams[a], "FTHG": cell // matrix.shape[1],
                             "FTAG": cell % matrix.shape[1]})
                day = day + pd.Timedelta(days=1)
    truth = {"attack": attack - attack.mean(), "defence": defence, "home_adv": home_adv, "rho": rho}
    return pd.DataFrame(rows), truth


def test_dc_scoreline_matrix_sums_to_one():
    for lam, mu, rho in [(1.5, 1.1, -0.1), (0.4, 0.3, 0.05), (3.0, 0.6, -0.2), (2.2, 2.0, 0.0)]:
        matrix = main.dc_scoreline_matrix(lam, mu, rho)
        assert matrix.shape == (main.MAX_GOALS + 1, main.MAX_GOALS + 1)
        assert abs(matrix.sum() - 1.0) < 1e-12
        assert (matrix >= 0).all()


def test_dc_with_rho_zero_is_the_plain_poisson_model():
    lambda_home = np.array([1.5, 0.8, 2.4])
    lambda_away = np.array([1.1, 2.0, 0.7])
    outcome_dc, p00, p11 = main.dc_scoreline_probabilities(lambda_home, lambda_away, 0.0)
    outcome_poisson, over_probs = main.poisson_probabilities(lambda_home, lambda_away)
    assert np.allclose(outcome_dc, outcome_poisson, atol=1e-12)
    # and the scoreline probabilities are the independent product
    assert abs(p00[0] - np.exp(-1.5) * np.exp(-1.1)) < 1e-6
    assert abs(p11[0] - (1.5 * np.exp(-1.5)) * (1.1 * np.exp(-1.1))) < 1e-6


def test_dc_rho_changes_only_the_low_scores():
    plain = main.dc_scoreline_matrix(1.4, 1.2, 0.0)
    corrected = main.dc_scoreline_matrix(1.4, 1.2, -0.1)
    # negative rho: more 0-0 and 1-1, fewer 0-1 and 1-0
    assert corrected[0, 0] > plain[0, 0]
    assert corrected[1, 1] > plain[1, 1]
    assert corrected[0, 1] < plain[0, 1]
    assert corrected[1, 0] < plain[1, 0]
    # a score without correction only changes by the renormalisation (tiny)
    assert abs(corrected[3, 2] / plain[3, 2] - 1) < 1e-3


def test_dc_probabilities_are_valid_even_for_extreme_rho_and_rates():
    for lam in [0.2, 1.3, 3.5, 5.0]:
        for mu in [0.2, 1.0, 3.5, 5.0]:
            for rho in [-1.0, -0.25, 0.0, 0.06, 1.0]:      # also values far outside the valid range
                matrix = main.dc_scoreline_matrix(lam, mu, rho)
                assert (matrix >= 0).all()
                assert abs(matrix.sum() - 1.0) < 1e-12
                probs, p00, p11 = main.dc_scoreline_probabilities(np.array([lam]), np.array([mu]), rho)
                assert abs(probs.sum() - 1.0) < 1e-12
                assert (probs >= 0).all() and (probs <= 1).all()


def test_clip_rho_keeps_all_correction_factors_non_negative():
    for lam in [0.3, 1.0, 2.5, 4.0]:
        for mu in [0.3, 1.0, 2.5, 4.0]:
            for rho in [-2.0, -0.3, 0.0, 0.1, 2.0]:
                r = float(main.clip_rho(rho, lam, mu))
                assert 1 - lam * mu * r >= -1e-12
                assert 1 + lam * r >= -1e-12
                assert 1 + mu * r >= -1e-12
                assert 1 - r >= -1e-12
    # a rho that is already valid is not changed
    assert float(main.clip_rho(-0.05, 1.3, 1.1)) == -0.05


def test_dc_time_weights_fall_with_age():
    days = np.array([0, 10, 100, 1000])
    weights = main.dc_time_weights(days, 0.01)
    assert weights[0] == 1.0
    assert (np.diff(weights) < 0).all()                 # older match -> smaller weight
    assert np.allclose(weights, np.exp(-0.01 * days))
    assert np.allclose(main.dc_time_weights(days, 0.0), 1.0)    # xi = 0: no decay


def test_dc_gradient_matches_numerical_gradient():
    from scipy.optimize import check_grad
    data, truth = make_dc_league(1, 2)
    teams = sorted(set(data["HomeTeam"]) | set(data["AwayTeam"]))
    number = {}
    for k in range(len(teams)):
        number[teams[k]] = k
    home_idx = data["HomeTeam"].map(number).values
    away_idx = data["AwayTeam"].map(number).values
    home_goals = data["FTHG"].values.astype(float)
    away_goals = data["FTAG"].values.astype(float)
    cutoff = data["Date"].max() + pd.Timedelta(days=1)
    weights = main.dc_time_weights((cutoff - data["Date"]).dt.days.values, 0.01)
    n = len(teams)
    rng = np.random.default_rng(5)
    start = np.concatenate([rng.normal(0, 0.2, n), rng.normal(0.2, 0.2, n), [0.25, -0.05]])

    def value(p):
        return main.dc_negative_log_likelihood(p, n, home_idx, away_idx, home_goals, away_goals, weights)[0]

    def gradient(p):
        return main.dc_negative_log_likelihood(p, n, home_idx, away_idx, home_goals, away_goals, weights)[1]

    error = check_grad(value, gradient, start)
    assert error < 1e-3 * max(1.0, np.linalg.norm(gradient(start)))


def test_dc_fit_recovers_known_parameters():
    data, truth = make_dc_league(7, 25)                 # 25 * 56 = 1400 matches
    cutoff = data["Date"].max() + pd.Timedelta(days=1)
    model = main.fit_dixon_coles(data, cutoff, 0.0)
    assert model["converged"]
    teams = ["T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7"]
    for k in range(8):
        assert abs(model["attack"][teams[k]] - truth["attack"][k]) < 0.12
        assert abs(model["defence"][teams[k]] - truth["defence"][k]) < 0.12
    assert abs(model["home_adv"] - truth["home_adv"]) < 0.1
    assert abs(model["rho"] - truth["rho"]) < 0.1
    # the attack values are centred
    assert abs(sum(model["attack"].values())) < 1e-9


def test_dc_unseen_team_gets_average_strength_and_valid_probabilities():
    data, truth = make_dc_league(3, 6)
    cutoff = data["Date"].max() + pd.Timedelta(days=1)
    model = main.fit_dixon_coles(data, cutoff, 0.0)
    attack, defence = main.team_strength(model, "Newly Promoted FC")
    assert attack == 0.0
    assert abs(defence - np.mean(list(model["defence"].values()))) < 1e-12
    # a known team in a match against the new team: still valid probabilities
    attack_home, defence_home = main.team_strength(model, "T0")
    lam = np.exp(attack_home + defence + model["home_adv"])
    mu = np.exp(attack + defence_home)
    probs, p00, p11 = main.dc_scoreline_probabilities(np.array([lam]), np.array([mu]), model["rho"])
    assert abs(probs.sum() - 1.0) < 1e-12


def test_dc_prediction_does_not_use_results_on_or_after_the_prediction_date():
    # LEAKAGE TEST. Change the result of one match. Every match played on or before
    # that day (this includes the match itself) must get EXACTLY the same prediction.
    data, truth = make_dc_league(11, 6)                  # 336 matches
    targets = data.index[200:]                           # predict the last 136 matches
    changed = data.copy()
    k = 260                                              # a match in the middle of the targets
    changed.loc[k, "FTHG"] = 9
    changed.loc[k, "FTAG"] = 0

    before = main.dixon_coles_predict(data, targets, 0.003)
    after = main.dixon_coles_predict(changed, targets, 0.003)

    changed_date = data.loc[k, "Date"]
    unchanged_count = 0
    later_changed_count = 0
    for position in range(len(targets)):
        match_date = data.loc[targets[position], "Date"]
        same = np.array_equal(before["outcome_probs"][position], after["outcome_probs"][position])
        if match_date <= changed_date:
            assert same                                  # no leakage
            unchanged_count = unchanged_count + 1
        elif not same:
            later_changed_count = later_changed_count + 1
    assert unchanged_count > 0
    # sanity check of the test itself: later matches DO see the changed result
    assert later_changed_count > 0


def test_dc_leagues_are_fitted_separately():
    # two leagues with the same team names: results of league B must not change predictions in league A
    league_a, truth = make_dc_league(21, 5, league="A")
    league_b, truth_b = make_dc_league(22, 5, league="B")
    both = pd.concat([league_a, league_b], ignore_index=True).sort_values("Date", kind="stable")
    both = both.reset_index(drop=True)
    targets = both.index[both["Date"] >= pd.Timestamp("2020-03-01")]
    targets_a = [i for i in targets if both.loc[i, "League"] == "A"]

    changed = both.copy()
    for i in changed.index:
        if changed.loc[i, "League"] == "B":
            changed.loc[i, "FTHG"] = 5          # change every result of league B
    first = main.dixon_coles_predict(both, targets, 0.002)
    second = main.dixon_coles_predict(changed, targets, 0.002)
    position_of = {}
    for position in range(len(targets)):
        position_of[targets[position]] = position
    for i in targets_a:
        assert np.array_equal(first["outcome_probs"][position_of[i]],
                              second["outcome_probs"][position_of[i]])


def test_dc_predict_gives_valid_probabilities_and_counts_refits():
    data, truth = make_dc_league(5, 5)
    targets = data.index[150:]
    result = main.dixon_coles_predict(data, targets, 0.002)
    assert result["outcome_probs"].shape == (len(targets), 3)
    assert np.allclose(result["outcome_probs"].sum(axis=1), 1.0, atol=1e-9)
    assert (result["outcome_probs"] >= 0).all()
    assert result["refits"] >= 1
    assert result["not_converged"] == 0
    assert result["rho_clipped"] == 0


def test_dc_choose_xi_uses_only_the_validation_matches():
    data, truth = make_dc_league(9, 6)
    valid_index = data.index[150:230]
    y_valid = np.zeros(len(valid_index), dtype=int)
    for k in range(len(valid_index)):
        row = data.loc[valid_index[k]]
        if row["FTHG"] > row["FTAG"]:
            y_valid[k] = 0
        elif row["FTHG"] == row["FTAG"]:
            y_valid[k] = 1
        else:
            y_valid[k] = 2
    best_xi, result, losses = main.choose_xi(data, valid_index, y_valid)
    assert best_xi in main.XI_GRID
    assert set(losses.keys()) == set(main.XI_GRID)
    assert abs(losses[best_xi] - min(losses.values())) < 1e-9


# ---------------------------------------------------------------
# Tests: do the features add information beyond the odds?
# ---------------------------------------------------------------
def make_odds_data(seed, n, feature_effect):
    # synthetic matches. Bookmaker probabilities are random. The true outcome probabilities
    # follow a multinomial logistic model of the odds ratios; if feature_effect > 0 the first
    # feature also shifts the draw/away-win versus home-win log-odds. Other features are noise.
    rng = np.random.default_rng(seed)
    book = rng.dirichlet([6, 3, 4], size=n)
    ratios = main.odds_log_ratios(book)
    features = rng.standard_normal((n, 7))
    score_draw = 0.1 + 0.9 * ratios[:, 0] + feature_effect * features[:, 0]
    score_away = -0.1 + 0.9 * ratios[:, 1] - feature_effect * features[:, 0]
    scores = np.column_stack([np.zeros(n), score_draw, score_away])
    probs = np.exp(scores)
    probs = probs / probs.sum(axis=1).reshape(-1, 1)
    y = np.zeros(n, dtype=int)
    for i in range(n):
        y[i] = rng.choice(3, p=probs[i])
    return ratios, features, y


def test_odds_log_ratios_known_values():
    probs = np.array([[0.5, 0.25, 0.25]])
    ratios = main.odds_log_ratios(probs)
    assert abs(ratios[0, 0] - np.log(0.5)) < 1e-12
    assert abs(ratios[0, 1] - np.log(0.5)) < 1e-12


def test_odds_log_ratios_with_proportional_margin_removal_equal_the_raw_odds_ratio():
    # the margin cancels in a ratio: ln(pD/pH) = ln(oddsH/oddsD) for proportional probabilities
    odds = pd.DataFrame({"B365H": [2.6, 1.3], "B365D": [3.2, 5.5], "B365A": [2.8, 11.0]})
    ratios = main.odds_log_ratios(main.bookmaker_probabilities(odds, "proportional"))
    assert np.allclose(ratios[:, 0], np.log(odds["B365H"] / odds["B365D"]))
    assert np.allclose(ratios[:, 1], np.log(odds["B365H"] / odds["B365A"]))


def test_odds_models_give_valid_probabilities_and_nested_likelihoods():
    ratios, features, y = make_odds_data(1, 1500, 0.0)
    fit = main.fit_odds_models(ratios[:1000], features[:1000], y[:1000], ratios[1000:], features[1000:])
    assert fit["converged"]
    for key in ["probs_a", "probs_b"]:
        assert fit[key].shape == (500, 3)
        assert np.allclose(fit[key].sum(axis=1), 1.0, atol=1e-9)
        assert (fit[key] >= 0).all() and (fit[key] <= 1).all()
    # B contains A, so its log-likelihood cannot be lower, and the test statistic is not negative
    assert fit["llf_b"] >= fit["llf_a"] - 1e-6
    assert fit["lr_statistic"] >= -1e-6
    assert fit["lr_df"] == 14                      # 7 features x 2 equations


def test_features_without_information_are_not_significant():
    ratios, features, y = make_odds_data(2, 3000, 0.0)
    fit = main.fit_odds_models(ratios[:2500], features[:2500], y[:2500], ratios[2500:], features[2500:])
    assert fit["lr_p_value"] > 0.01


def test_features_with_information_are_detected():
    ratios, features, y = make_odds_data(3, 3000, 0.8)
    fit = main.fit_odds_models(ratios[:2500], features[:2500], y[:2500], ratios[2500:], features[2500:])
    assert fit["lr_p_value"] < 1e-6
    loss_a = main.match_log_losses(fit["probs_a"], y[2500:]).mean()
    loss_b = main.match_log_losses(fit["probs_b"], y[2500:]).mean()
    assert loss_b < loss_a                         # out of sample, the features help


def test_odds_models_cannot_see_test_outcomes():
    # LEAKAGE TEST. The fit function has no test-outcome input at all, and the fitted
    # models do not change when the test matches change.
    import inspect
    names = list(inspect.signature(main.fit_odds_models).parameters)
    assert names == ["odds_train", "features_train", "y_train", "odds_test", "features_test"]

    ratios, features, y = make_odds_data(4, 1200, 0.3)
    first = main.fit_odds_models(ratios[:1000], features[:1000], y[:1000], ratios[1000:], features[1000:])
    other_test_ratios = ratios[1000:] + 0.5
    other_test_features = features[1000:] * 3.0
    second = main.fit_odds_models(ratios[:1000], features[:1000], y[:1000],
                                  other_test_ratios, other_test_features)
    assert first["llf_a"] == second["llf_a"]
    assert first["llf_b"] == second["llf_b"]
    assert np.array_equal(first["result_b"].params, second["result_b"].params)


def test_features_vs_odds_coefficient_rows_have_all_terms():
    ratios, features, y = make_odds_data(5, 800, 0.2)
    fit = main.fit_odds_models(ratios[:700], features[:700], y[:700], ratios[700:], features[700:])
    names = ["constant", "ln(pDraw/pHome)", "ln(pAway/pHome)"] + main.FEATURE_NAMES
    rows = main.coefficient_rows(fit["result_b"], "9999", "B", names)
    assert len(rows) == 2 * len(names)
    assert {row["equation"] for row in rows} == {"draw vs home win", "away win vs home win"}


def test_dc_likelihood_stays_finite_for_extreme_parameters():
    # While searching, the optimiser may try absurd values. The likelihood must then stay
    # finite and must not raise an overflow (np.errstate turns every floating point warning
    # into an error here).
    data, truth = make_dc_league(2, 2)
    teams = sorted(set(data["HomeTeam"]) | set(data["AwayTeam"]))
    number = {}
    for k in range(len(teams)):
        number[teams[k]] = k
    home_idx = data["HomeTeam"].map(number).values
    away_idx = data["AwayTeam"].map(number).values
    home_goals = data["FTHG"].values.astype(float)
    away_goals = data["FTAG"].values.astype(float)
    weights = np.ones(len(data))
    n = len(teams)
    extreme = np.concatenate([np.full(n, 800.0), np.full(n, 800.0), [0.2, -0.05]])
    with np.errstate(all="raise"):
        value, gradient = main.dc_negative_log_likelihood(
            extreme, n, home_idx, away_idx, home_goals, away_goals, weights)
    assert np.isfinite(value)
    assert np.isfinite(gradient).all()
