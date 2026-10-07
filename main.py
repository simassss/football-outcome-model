"""
Football match outcome prediction

Idea:
1. Load historical results and bookmaker odds (football-data.co.uk)
   for one or more leagues. Files are cached in the data/ folder.
2. Build features for every match from the PAST only:
   - form of each team (last 5 matches)
   - our own Elo rating of each team (K and home advantage are tuned)
   - (optional) professional Elo rating from clubelo.com
3. Walk-forward evaluation: for every test season, train on ALL earlier
   seasons and test on that season (no shuffling, no peeking).
4. Models: logistic regression, gradient boosting, Poisson goals model.
5. Baselines: class frequencies, always-home-win, bookmaker probabilities
   (Bet365 odds, margin removed).
6. Metrics: log loss, accuracy, calibration plot, value-bet backtest.

Run:
    python main.py                              (Premier League only)
    python main.py --leagues E0 D1 SP1 I1 F1    (five big leagues)
    python main.py --leagues E0 --clubelo       (add ClubElo comparison)
    python main.py --first-test-season 2021     (test seasons 2020/21 ... 2024/25)

7. Bootstrap confidence intervals for the log loss difference to the bookmaker.
8. Blend of model and bookmaker probabilities (weight chosen on validation).

9. Ranked Probability Score and a Diebold-Mariano test against the bookmaker.
10. Three ways to remove the bookmaker margin (proportional, power, Shin) and a
    favourite-longshot bias check of the bookmaker probabilities.
11. Dixon-Coles goals model (attack/defence per team, home advantage, low-score
    correction rho, time decay xi chosen on the validation season), compared with
    the Poisson model and the bookmaker, plus a check of the share of draws.
12. Multinomial logistic regression (statsmodels): bookmaker log-odds ratios only
    (model A) versus log-odds ratios plus form and Elo (model B), with a likelihood-ratio
    test on the training seasons and the out-of-sample log loss of both.

Output: results/metrics.csv, results/walk_forward.csv, results/bootstrap.csv,
        results/blend.csv, results/rps.csv, results/dm.csv, results/margin_methods.csv,
        results/bookmaker_calibration.csv, results/favourite_longshot.csv,
        results/bookmaker_calibration.png, results/dixon_coles.csv,
        results/dixon_coles_tests.csv, results/dixon_coles_draw_share.csv,
        results/dixon_coles_parameters.csv, results/features_vs_odds.csv,
        results/features_vs_odds_coefficients.csv, results/calibration.png
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import poisson, norm, chi2
from scipy.optimize import brentq, minimize
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

# ---------------------------------------------------------------
# Settings
# ---------------------------------------------------------------
LEAGUE_NAMES = {
    "E0": "England Premier League",
    "D1": "Germany Bundesliga",
    "SP1": "Spain La Liga",
    "I1": "Italy Serie A",
    "F1": "France Ligue 1",
}

# 2010/11 ... 2024/25. All of these have the Bet365 odds columns (B365H/B365D/B365A)
# without missing values in the Premier League files (checked when this list was made).
SEASONS = ["1011", "1112", "1213", "1314", "1415", "1516", "1617", "1718",
           "1819", "1920", "2021", "2122", "2223", "2324", "2425"]
FIRST_TEST_SEASON = "1920"                   # default: test seasons 2019/20 ... 2024/25
N_LAST = 5                                   # form = last 5 matches
MIN_MATCHES = 3                              # need at least 3 past matches
EDGE = 1.05                                  # bet only if p * odds > 1.05

ELO_START = 1500                             # starting rating of every team
ELO_K = 20                                   # default: how fast ratings change
ELO_HOME = 60                                # default: home advantage in Elo points
ELO_K_GRID = [10, 20, 30, 40]                # values tried when tuning Elo
ELO_HOME_GRID = [40, 60, 80, 100]

MAX_GOALS = 10                               # Poisson model: scorelines up to 10-10

# Dixon-Coles model
XI_GRID = [0.0, 0.001, 0.002, 0.003, 0.005]  # time decay values tried (per day); 0 = no decay
DC_BLOCK_DAYS = 7                            # refit the model once per block of 7 days
DC_RIDGE = 1.0                               # small penalty that keeps rarely seen teams near average
# Bounds for rho. With expected goals up to 4 per team these bounds keep all four
# correction factors >= 0 (1 + 4 * (-0.25) = 0 and 1 - 16 * 0.06 > 0).
DC_RHO_MIN = -0.25
DC_RHO_MAX = 0.06

MARGIN_METHODS = ["proportional", "power", "shin"]  # ways to remove the bookmaker margin
CALIBRATION_BIN_WIDTH = 0.05                 # bookmaker calibration: bins of width 0.05
CALIBRATION_MIN_MATCHES = 30                 # ignore bins with fewer observations

N_BOOTSTRAP = 1000                          # bootstrap resamples of the test matches
BOOTSTRAP_SEED = 42                          # fixed seed so results are repeatable

# columns we use from the football-data.co.uk files
FILE_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "B365H", "B365D", "B365A"]

DATA_DIR = "data"
CLUBELO_DIR = os.path.join("data", "clubelo")
RESULTS_DIR = "results"

OUTCOME_CODE = {"H": 0, "D": 1, "A": 2}      # 0 = home win, 1 = draw, 2 = away win

# ClubElo writes team names without spaces (e.g. "ManCity").
# Names that differ from that simple rule go here. If a team is
# reported as "not found", check its name on clubelo.com and add it.
CLUBELO_NAMES = {
    "Man City": "ManCity",
    "Man United": "ManUnited",
    "Nott'm Forest": "Forest",
}


# ---------------------------------------------------------------
# Step 1. Load data
# ---------------------------------------------------------------
def load_season(league, season):
    path = os.path.join(DATA_DIR, league + "_" + season + ".csv")
    if os.path.exists(path):
        df = pd.read_csv(path, encoding="latin-1")
    else:
        url = "https://www.football-data.co.uk/mmz4281/" + season + "/" + league + ".csv"
        df = pd.read_csv(url, encoding="latin-1")
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(path, index=False)

    # The files have about 120 columns. Keep only the ones we need BEFORE adding
    # new columns (adding columns to a very wide table makes pandas print a warning).
    df = df[FILE_COLUMNS].copy()
    df["Season"] = season
    df["League"] = league
    return df


def load_data(leagues):
    frames = []
    for league in leagues:
        for season in SEASONS:
            frames.append(load_season(league, season))

    data = pd.concat(frames, ignore_index=True)

    # keep only the columns we need
    columns = ["League", "Season"] + FILE_COLUMNS
    data = data[columns]
    data = data.dropna()

    data["Date"] = pd.to_datetime(data["Date"], dayfirst=True, format="mixed")
    data = data.sort_values("Date")
    data = data.reset_index(drop=True)
    return data


# ---------------------------------------------------------------
# Step 2. Features
# ---------------------------------------------------------------
def get_points(goals_for, goals_against):
    # 3 points for a win, 1 for a draw, 0 for a loss
    if goals_for > goals_against:
        return 3
    elif goals_for == goals_against:
        return 1
    else:
        return 0


def average_last(history, n):
    # history is a list of (points, goals_for, goals_against)
    # take the last n matches and return the averages
    last = history[-n:]
    total_points = 0
    total_for = 0
    total_against = 0
    for match in last:
        total_points = total_points + match[0]
        total_for = total_for + match[1]
        total_against = total_against + match[2]
    count = len(last)
    return total_points / count, total_for / count, total_against / count


def elo_expected(home_elo, away_elo, home_adv=ELO_HOME):
    # probability that the home team "wins" according to Elo
    # (draws are counted as half a win in the Elo formula)
    diff = away_elo - home_elo - home_adv
    return 1 / (1 + 10 ** (diff / 400))


def elo_update(home_elo, away_elo, home_goals, away_goals, k=ELO_K, home_adv=ELO_HOME):
    # returns the new ratings after the match
    expected = elo_expected(home_elo, away_elo, home_adv)
    if home_goals > away_goals:
        actual = 1.0
    elif home_goals == away_goals:
        actual = 0.5
    else:
        actual = 0.0
    change = k * (actual - expected)
    return home_elo + change, away_elo - change


def elo_differences(data, k, home_adv):
    # Elo difference (home - away) BEFORE every match, for all rows of data
    home_teams = data["HomeTeam"].values
    away_teams = data["AwayTeam"].values
    home_goals = data["FTHG"].values
    away_goals = data["FTAG"].values

    elo = {}
    diffs = np.zeros(len(data))
    for i in range(len(data)):
        home = home_teams[i]
        away = away_teams[i]
        if home not in elo:
            elo[home] = ELO_START
        if away not in elo:
            elo[away] = ELO_START

        diffs[i] = elo[home] - elo[away]

        new_home, new_away = elo_update(elo[home], elo[away],
                                        home_goals[i], away_goals[i], k, home_adv)
        elo[home] = new_home
        elo[away] = new_away
    return diffs


def tune_elo(data, test_season):
    # Choose Elo K and home advantage using ONLY seasons before test_season:
    #   - seasons before the validation season: fit a 1-feature model
    #   - validation season (the one right before the test season): score it
    # The test season is never used.
    seasons_before = SEASONS[:SEASONS.index(test_season)]
    if len(seasons_before) < 2:
        raise ValueError("Need at least two seasons before " + test_season + " to tune Elo")
    valid_season = seasons_before[-1]
    train_seasons = seasons_before[:-1]

    y = data["FTR"].map(OUTCOME_CODE).values
    is_train = data["Season"].isin(train_seasons).values
    is_valid = (data["Season"] == valid_season).values

    best_k = ELO_K
    best_home = ELO_HOME
    best_loss = None
    for k in ELO_K_GRID:
        for home_adv in ELO_HOME_GRID:
            diffs = elo_differences(data, k, home_adv).reshape(-1, 1) / 100
            model = LogisticRegression(max_iter=1000)
            model.fit(diffs[is_train], y[is_train])
            probs = model.predict_proba(diffs[is_valid])
            loss = log_loss(y[is_valid], probs, labels=[0, 1, 2])
            if best_loss is None or loss < best_loss:
                best_loss = loss
                best_k = k
                best_home = home_adv
    return best_k, best_home, best_loss


def build_features(data, elo_k=ELO_K, elo_home=ELO_HOME):
    # history[team] = list of past matches of that team
    # elo[team]     = current rating of that team
    history = {}
    elo = {}
    rows = []

    for i in range(len(data)):
        home = data.loc[i, "HomeTeam"]
        away = data.loc[i, "AwayTeam"]
        home_goals = data.loc[i, "FTHG"]
        away_goals = data.loc[i, "FTAG"]

        if home not in history:
            history[home] = []
            elo[home] = ELO_START
        if away not in history:
            history[away] = []
            elo[away] = ELO_START

        # features use ONLY matches played before this one (no data leakage)
        if len(history[home]) >= MIN_MATCHES and len(history[away]) >= MIN_MATCHES:
            h_pts, h_gf, h_ga = average_last(history[home], N_LAST)
            a_pts, a_gf, a_ga = average_last(history[away], N_LAST)
            row = {
                "index": i,
                "home_pts": h_pts, "home_gf": h_gf, "home_ga": h_ga,
                "away_pts": a_pts, "away_gf": a_gf, "away_ga": a_ga,
                "elo_diff": elo[home] - elo[away],
            }
            rows.append(row)

        # only now update the history and ratings with this match
        history[home].append((get_points(home_goals, away_goals), home_goals, away_goals))
        history[away].append((get_points(away_goals, home_goals), away_goals, home_goals))
        new_home, new_away = elo_update(elo[home], elo[away], home_goals, away_goals,
                                        elo_k, elo_home)
        elo[home] = new_home
        elo[away] = new_away

    features = pd.DataFrame(rows)
    features = features.set_index("index")
    return features


# ---------------------------------------------------------------
# Step 3. ClubElo (optional external rating)
# ---------------------------------------------------------------
def clubelo_name(team):
    if team in CLUBELO_NAMES:
        return CLUBELO_NAMES[team]
    return team.replace(" ", "")


def fetch_clubelo(team):
    # returns the rating history table of one team, or None if not found
    name = clubelo_name(team)
    path = os.path.join(CLUBELO_DIR, name + ".csv")
    if os.path.exists(path):
        table = pd.read_csv(path)
    else:
        try:
            table = pd.read_csv("http://api.clubelo.com/" + name)
        except Exception:
            return None
        if len(table) == 0 or "Elo" not in table.columns:
            return None
        os.makedirs(CLUBELO_DIR, exist_ok=True)
        table.to_csv(path, index=False)
    table["From"] = pd.to_datetime(table["From"])
    table["To"] = pd.to_datetime(table["To"])
    return table


def clubelo_rating(table, date):
    # rating that was valid on the given date (NaN if there is none)
    valid = table[(table["From"] <= date) & (table["To"] >= date)]
    if len(valid) == 0:
        return np.nan
    return valid["Elo"].iloc[0]


def clubelo_difference(data, features):
    # Elo difference (home - away) from clubelo.com for every feature row.
    # We use the rating of the day BEFORE the match to avoid leakage.
    # Returns None if some team or date could not be matched.
    teams = set()
    for i in features.index:
        teams.add(data.loc[i, "HomeTeam"])
        teams.add(data.loc[i, "AwayTeam"])

    tables = {}
    missing = []
    for team in sorted(teams):
        table = fetch_clubelo(team)
        if table is None:
            missing.append(team)
        else:
            tables[team] = table

    if len(missing) > 0:
        print("ClubElo: teams not found:", ", ".join(missing))
        print("Add them to CLUBELO_NAMES in main.py (check the name on clubelo.com).")
        return None

    diffs = []
    for i in features.index:
        date = data.loc[i, "Date"] - pd.Timedelta(days=1)
        home_rating = clubelo_rating(tables[data.loc[i, "HomeTeam"]], date)
        away_rating = clubelo_rating(tables[data.loc[i, "AwayTeam"]], date)
        diffs.append(home_rating - away_rating)

    result = pd.Series(diffs, index=features.index)
    if result.isna().any():
        print("ClubElo: no rating for", int(result.isna().sum()), "matches, skipping ClubElo.")
        return None
    return result


# ---------------------------------------------------------------
# Step 4. Models and baselines
# ---------------------------------------------------------------
def proportional_probabilities(odds):
    # odds has 3 columns (home, draw, away). Implied probability = 1 / odds.
    # The three implied probabilities add up to MORE than 1 (the bookmaker margin).
    # Method (a): divide each by the sum, so every probability is scaled by the same factor.
    implied = 1 / odds
    total = implied.sum(axis=1).reshape(-1, 1)
    return implied / total


def power_gap(k, implied):
    # how far the sum of (implied probability ^ k) is from 1
    return (implied ** k).sum() - 1


def power_row(odds_row):
    # Method (b), power method, for ONE match: find the exponent k >= 1 such that
    #   (1/odds_home)^k + (1/odds_draw)^k + (1/odds_away)^k = 1
    # and use (1/odds)^k as the probabilities. Raising a number below 1 to a power
    # above 1 makes it smaller, and small numbers shrink by a larger share.
    implied = 1 / odds_row
    if implied.sum() <= 1:
        return implied / implied.sum()          # no margin to remove
    k = brentq(power_gap, 1.0, 100.0, args=(implied,))
    return implied ** k


def shin_probabilities_for(z, implied):
    # Shin's formula: probabilities for a given insider share z (0 <= z < 1)
    total = implied.sum()
    root = np.sqrt(z ** 2 + 4 * (1 - z) * implied ** 2 / total)
    return (root - z) / (2 * (1 - z))


def shin_gap(z, implied):
    # how far the sum of Shin's probabilities is from 1
    return shin_probabilities_for(z, implied).sum() - 1


def shin_row(odds_row):
    # Method (c), Shin's method, for ONE match: find the insider share z for which
    # Shin's probabilities add up to exactly 1 (see README for the idea).
    implied = 1 / odds_row
    if implied.sum() <= 1:
        return implied / implied.sum()          # no margin to remove
    z = brentq(shin_gap, 0.0, 0.999, args=(implied,))
    probs = shin_probabilities_for(z, implied)
    return probs / probs.sum()                  # remove tiny numerical error


def bookmaker_probabilities(df, method="proportional"):
    # margin-free bookmaker probabilities, one row per match: [home, draw, away]
    odds = df[["B365H", "B365D", "B365A"]].values
    if method == "proportional":
        return proportional_probabilities(odds)
    elif method == "power":
        result = np.zeros((len(odds), 3))
        for i in range(len(odds)):
            result[i] = power_row(odds[i])
        return result
    elif method == "shin":
        result = np.zeros((len(odds), 3))
        for i in range(len(odds)):
            result[i] = shin_row(odds[i])
        return result
    else:
        raise ValueError("unknown margin method: " + method)


def frequency_probabilities(y_train, n_rows):
    # always predict the average rates seen in the training data
    rates = []
    for outcome in range(3):
        rates.append((y_train == outcome).mean())
    return np.tile(rates, (n_rows, 1))


def logistic_probabilities(X_train, y_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    return model.predict_proba(X_test_scaled)


def boosting_probabilities(X_train, y_train, X_test):
    model = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                           max_iter=100, random_state=42)
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)


def poisson_goal_rates(X_train, home_goals_train, away_goals_train, X_test):
    # Poisson regression predicts the expected number of goals (lambda)
    # of the home team and of the away team
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    home_model = PoissonRegressor(alpha=0.01, max_iter=1000)
    home_model.fit(X_train_scaled, home_goals_train)
    away_model = PoissonRegressor(alpha=0.01, max_iter=1000)
    away_model.fit(X_train_scaled, away_goals_train)

    lambda_home = home_model.predict(X_test_scaled)
    lambda_away = away_model.predict(X_test_scaled)
    return lambda_home, lambda_away


def poisson_probabilities(lambda_home, lambda_away):
    # For every match: assume home and away goals are independent Poisson
    # variables. Then the probability of the score h:a is
    #   P(home scores h) * P(away scores a)
    # and we add up the scores to get home win / draw / away win
    # and the probability of more than 2.5 goals in total.
    goals = np.arange(MAX_GOALS + 1)
    outcome_probs = np.zeros((len(lambda_home), 3))
    over_probs = np.zeros(len(lambda_home))

    for i in range(len(lambda_home)):
        home_pmf = poisson.pmf(goals, lambda_home[i])
        away_pmf = poisson.pmf(goals, lambda_away[i])
        matrix = np.outer(home_pmf, away_pmf)     # matrix[h, a] = P(score h:a)
        matrix = matrix / matrix.sum()            # cut-off at MAX_GOALS: renormalise

        p_home = np.tril(matrix, -1).sum()        # cells where h > a
        p_draw = np.trace(matrix)                 # cells where h == a
        p_away = np.triu(matrix, 1).sum()         # cells where h < a
        outcome_probs[i] = [p_home, p_draw, p_away]

        # total goals <= 2: scores 0:0, 0:1, 0:2, 1:0, 1:1, 2:0
        p_under = matrix[0, 0] + matrix[0, 1] + matrix[0, 2] + matrix[1, 0] + matrix[1, 1] + matrix[2, 0]
        over_probs[i] = 1 - p_under

    return outcome_probs, over_probs


def three_model_probabilities(X_train, y_train, home_goals_train, away_goals_train, X_test):
    # fit the three models on the training rows and predict the test rows
    # returns a dictionary {name: probabilities} and the Poisson "over 2.5" probabilities
    probs = {}
    probs["Logistic regression"] = logistic_probabilities(X_train, y_train, X_test)
    probs["Gradient boosting"] = boosting_probabilities(X_train, y_train, X_test)

    # Poisson goals model: predict goals, then turn them into probabilities
    lambda_home, lambda_away = poisson_goal_rates(
        X_train, home_goals_train, away_goals_train, X_test)
    poisson_probs, over_probs = poisson_probabilities(lambda_home, lambda_away)
    probs["Poisson goals model"] = poisson_probs
    return probs, over_probs


# ---------------------------------------------------------------
# Step 4b. Bootstrap confidence interval for the log loss difference
# ---------------------------------------------------------------
def match_log_losses(probs, y):
    # log loss of every single match: -log(probability given to the real outcome)
    # probabilities are clipped so that log(0) can never happen
    losses = np.zeros(len(y))
    for i in range(len(y)):
        p = probs[i, y[i]]
        p = min(max(p, 1e-15), 1.0)
        losses[i] = -np.log(p)
    return losses


def bootstrap_loss_difference(model_probs, book_probs, y, n_resamples=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    # difference = model log loss - bookmaker log loss
    #   negative -> model is better, positive -> bookmaker is better
    # We resample the test matches WITH replacement n_resamples times and
    # compute the mean difference of each resample. The 2.5% and 97.5%
    # percentiles of those means form the 95% confidence interval.
    difference = match_log_losses(model_probs, y) - match_log_losses(book_probs, y)
    n = len(difference)

    rng = np.random.default_rng(seed)
    means = []
    for r in range(n_resamples):
        chosen = rng.integers(0, n, size=n)      # n random match numbers
        means.append(difference[chosen].mean())

    low = np.percentile(means, 2.5)
    high = np.percentile(means, 97.5)
    return difference.mean(), low, high


# ---------------------------------------------------------------
# Step 4b-2. Ranked Probability Score and Diebold-Mariano test
# ---------------------------------------------------------------
def rps_per_match(probs, y):
    # Ranked Probability Score of every match (lower is better).
    # The outcomes are ORDERED: home win (0) < draw (1) < away win (2).
    # RPS compares the cumulative predicted probabilities with the
    # cumulative real outcome, so predicting a draw when the home team wins
    # is "less wrong" than predicting an away win.
    #   RPS = ((cum_pred_1 - cum_obs_1)^2 + (cum_pred_2 - cum_obs_2)^2) / 2
    # cum_1 = P(home win), cum_2 = P(home win or draw)
    scores = np.zeros(len(y))
    for i in range(len(y)):
        cum_pred_1 = probs[i, 0]
        cum_pred_2 = probs[i, 0] + probs[i, 1]
        if y[i] == 0:                 # home win: cumulative outcome is (1, 1)
            cum_obs_1 = 1.0
            cum_obs_2 = 1.0
        elif y[i] == 1:               # draw: cumulative outcome is (0, 1)
            cum_obs_1 = 0.0
            cum_obs_2 = 1.0
        else:                         # away win: cumulative outcome is (0, 0)
            cum_obs_1 = 0.0
            cum_obs_2 = 0.0
        scores[i] = ((cum_pred_1 - cum_obs_1) ** 2 + (cum_pred_2 - cum_obs_2) ** 2) / 2
    return scores


def diebold_mariano(loss_differences):
    # Diebold-Mariano test: is the average difference of two forecasts' losses
    # different from 0?  loss_differences[i] = loss of model - loss of bookmaker
    # for match i (positive = model is worse). Returns (statistic, p-value).
    #
    # Assumptions (please read):
    #  - We forecast one match at a time (horizon 1), so we use the plain
    #    sample variance of the differences. We do not correct for
    #    autocorrelation. This treats the matches of a season as uncorrelated.
    #  - The statistic is compared with a normal distribution. That is only an
    #    approximation; with about 377 matches per season it is reasonable,
    #    with few matches it would not be.
    #  - The p-value is two-sided and is NOT corrected for the many tests we run.
    n = len(loss_differences)
    mean = loss_differences.mean()
    variance = loss_differences.var(ddof=1)

    if variance == 0:
        # all differences are equal: identical forecasts (mean 0) or a constant gap
        if mean == 0:
            return 0.0, 1.0
        return float(np.sign(mean) * np.inf), 0.0

    statistic = mean / np.sqrt(variance / n)
    p_value = 2 * (1 - norm.cdf(abs(statistic)))
    return statistic, p_value


# ---------------------------------------------------------------
# Step 4c. Blend of model and bookmaker probabilities
# ---------------------------------------------------------------
def blend_probabilities(model_probs, book_probs, weight):
    # weight = share of the model: 0 = only bookmaker, 1 = only model
    return weight * model_probs + (1 - weight) * book_probs


def choose_blend_weight(model_probs, book_probs, y):
    # try the weights 0.0, 0.1, ... 1.0 and keep the one with the lowest log loss.
    # IMPORTANT: the caller passes VALIDATION data here, never the test season.
    # On a tie the smaller weight (more trust in the bookmaker) wins.
    best_weight = 0.0
    best_loss = None
    for step in range(11):
        weight = step / 10
        blended = blend_probabilities(model_probs, book_probs, weight)
        loss = log_loss(y, blended, labels=[0, 1, 2])
        if best_loss is None or loss < best_loss - 1e-12:
            best_loss = loss
            best_weight = weight
    return best_weight


# ---------------------------------------------------------------
# Step 4d. Dixon-Coles goals model
# ---------------------------------------------------------------
# Every team has an attack value and a defence value, and there is one home
# advantage. Expected goals of a match:
#   home team: lambda = exp(attack_home + defence_away + home_advantage)
#   away team: mu     = exp(attack_away + defence_home)
# A higher attack value = scores more. A higher defence value = concedes MORE
# goals (it is a "weakness" value). The Poisson scoreline probabilities are
# corrected for the four low scores 0-0, 1-0, 0-1, 1-1 with the parameter rho.
# Recent matches count more than old ones (time decay with parameter xi).
# Teams are fitted only on matches BEFORE the day that is predicted.

def dc_time_weights(days_ago, xi):
    # weight of a training match that was played days_ago days before the
    # prediction day: exp(-xi * days_ago). xi = 0 gives every match weight 1.
    return np.exp(-xi * days_ago)


def clip_rho(rho, lam, mu):
    # The four correction factors must not become negative (a negative factor
    # would give a negative probability):
    #   1 - lam * mu * rho >= 0   means   rho <= 1 / (lam * mu)
    #   1 + lam * rho      >= 0   means   rho >= -1 / lam
    #   1 + mu * rho       >= 0   means   rho >= -1 / mu
    #   1 - rho            >= 0   means   rho <= 1
    # This function moves rho into the allowed range of the match (if needed).
    # It works for single numbers and for numpy arrays.
    lowest = np.maximum(-1 / lam, -1 / mu)
    highest = np.minimum(1 / (lam * mu), 1.0)
    return np.minimum(np.maximum(rho, lowest), highest)


def dc_scoreline_matrix(lam, mu, rho):
    # matrix[h, a] = probability that the score is h:a (home:away), for h and a
    # from 0 to MAX_GOALS. With rho = 0 this is the plain independent Poisson matrix.
    goals = np.arange(MAX_GOALS + 1)
    matrix = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))

    rho = float(clip_rho(rho, lam, mu))
    matrix[0, 0] = matrix[0, 0] * (1 - lam * mu * rho)
    matrix[0, 1] = matrix[0, 1] * (1 + lam * rho)
    matrix[1, 0] = matrix[1, 0] * (1 + mu * rho)
    matrix[1, 1] = matrix[1, 1] * (1 - rho)

    # The four corrections cancel each other, so the total stays 1 except for
    # the cut-off at MAX_GOALS. Renormalise as in the Poisson model.
    return matrix / matrix.sum()


def dc_scoreline_probabilities(lambda_home, lambda_away, rho):
    # For many matches: home win / draw / away win probabilities, and the
    # probabilities of the scores 0-0 and 1-1 (used to check the draw share).
    n = len(lambda_home)
    outcome_probs = np.zeros((n, 3))
    prob_00 = np.zeros(n)
    prob_11 = np.zeros(n)
    for i in range(n):
        matrix = dc_scoreline_matrix(lambda_home[i], lambda_away[i], rho)
        p_home = np.tril(matrix, -1).sum()        # cells where h > a
        p_draw = np.trace(matrix)                 # cells where h == a
        p_away = np.triu(matrix, 1).sum()         # cells where h < a
        outcome_probs[i] = [p_home, p_draw, p_away]
        prob_00[i] = matrix[0, 0]
        prob_11[i] = matrix[1, 1]
    return outcome_probs, prob_00, prob_11


def dc_negative_log_likelihood(params, n_teams, home_idx, away_idx,
                               home_goals, away_goals, weights):
    # The number that the optimiser makes small, and its gradient.
    # params = [attack of every team, defence of every team, home advantage, rho]
    attack_raw = params[0:n_teams]
    defence = params[n_teams:2 * n_teams]
    home_adv = params[2 * n_teams]
    rho = params[2 * n_teams + 1]

    # Only differences between teams matter: adding a constant to all attack
    # values and subtracting it from all defence values changes nothing. To get
    # one single solution, the attack values are centred (their mean is 0).
    attack = attack_raw - attack_raw.mean()

    log_lambda = attack[home_idx] + defence[away_idx] + home_adv
    log_mu = attack[away_idx] + defence[home_idx]
    # np.clip only matters if the optimiser tries an absurd step while it searches: it keeps
    # exp() finite (no overflow). For realistic values (|log goals| < 30) it changes nothing.
    lam = np.exp(np.clip(log_lambda, -30, 30))
    mu = np.exp(np.clip(log_mu, -30, 30))

    # Poisson part of the log-likelihood of every match
    # (the constant log(goals!) is left out, it does not depend on the parameters)
    log_lik = home_goals * log_lambda - lam + away_goals * log_mu - mu

    # Dixon-Coles correction factor tau of every match. Only the scores
    # 0-0, 0-1, 1-0 and 1-1 are corrected; every other score has tau = 1.
    # We also collect how tau changes with log(lambda), log(mu) and rho (for the gradient).
    tau = np.ones(len(home_goals))
    dtau_dloglam = np.zeros(len(home_goals))
    dtau_dlogmu = np.zeros(len(home_goals))
    dtau_drho = np.zeros(len(home_goals))

    is_00 = (home_goals == 0) & (away_goals == 0)
    tau[is_00] = 1 - lam[is_00] * mu[is_00] * rho
    dtau_dloglam[is_00] = -lam[is_00] * mu[is_00] * rho
    dtau_dlogmu[is_00] = -lam[is_00] * mu[is_00] * rho
    dtau_drho[is_00] = -lam[is_00] * mu[is_00]

    is_01 = (home_goals == 0) & (away_goals == 1)
    tau[is_01] = 1 + lam[is_01] * rho
    dtau_dloglam[is_01] = lam[is_01] * rho
    dtau_drho[is_01] = lam[is_01]

    is_10 = (home_goals == 1) & (away_goals == 0)
    tau[is_10] = 1 + mu[is_10] * rho
    dtau_dlogmu[is_10] = mu[is_10] * rho
    dtau_drho[is_10] = mu[is_10]

    is_11 = (home_goals == 1) & (away_goals == 1)
    tau[is_11] = 1 - rho
    dtau_drho[is_11] = -1.0

    # rho is bounded by the optimiser so that tau stays positive for realistic
    # goal rates. The floor only protects log() while the optimiser is searching.
    tau = np.maximum(tau, 1e-10)
    log_lik = log_lik + np.log(tau)

    # Ridge penalty: attack and defence values (defence relative to its mean)
    # are pulled gently towards 0. Without it, a team with only a few matches
    # (for example 0 goals scored) would get an infinitely low value.
    defence_centred = defence - defence.mean()
    penalty = DC_RIDGE * ((attack ** 2).sum() + (defence_centred ** 2).sum())

    negative_log_lik = -(weights * log_lik).sum() + penalty

    # Gradient. For every match: how does its log-likelihood change when
    # log(lambda) or log(mu) change?
    d_loglam = home_goals - lam + dtau_dloglam / tau
    d_logmu = away_goals - mu + dtau_dlogmu / tau
    weighted_d_loglam = weights * d_loglam
    weighted_d_logmu = weights * d_logmu

    # np.bincount adds up the values of all matches of the same team
    # (bincount(team_numbers, values)[t] = sum of values of the matches with team number t)
    grad_attack = (np.bincount(home_idx, weights=weighted_d_loglam, minlength=n_teams)
                   + np.bincount(away_idx, weights=weighted_d_logmu, minlength=n_teams))
    grad_defence = (np.bincount(away_idx, weights=weighted_d_loglam, minlength=n_teams)
                    + np.bincount(home_idx, weights=weighted_d_logmu, minlength=n_teams))
    grad_home_adv = weighted_d_loglam.sum()
    grad_rho = (weights * dtau_drho / tau).sum()

    # gradient of the NEGATIVE log-likelihood including the penalty
    grad_attack = -grad_attack + 2 * DC_RIDGE * attack
    grad_defence = -grad_defence + 2 * DC_RIDGE * defence_centred
    # the attack values are centred, so the gradient for the raw values
    # is the gradient minus its mean
    grad_attack = grad_attack - grad_attack.mean()

    gradient = np.concatenate([grad_attack, grad_defence, [-grad_home_adv, -grad_rho]])
    return negative_log_lik, gradient


def fit_dixon_coles(train, cutoff_date, xi):
    # Fit the model on the matches in `train` (all played before cutoff_date).
    # Method: weighted maximum likelihood with scipy.optimize.minimize and
    # L-BFGS-B. Why: the function is smooth, we know its exact gradient (so it is
    # fast and accurate even with about 40 parameters), and L-BFGS-B allows bounds,
    # which we need to keep rho in a safe range.
    if len(train) == 0:
        raise ValueError("no matches before " + str(cutoff_date))

    teams = sorted(set(train["HomeTeam"]) | set(train["AwayTeam"]))
    team_number = {}
    for k in range(len(teams)):
        team_number[teams[k]] = k
    n_teams = len(teams)

    home_idx = train["HomeTeam"].map(team_number).values
    away_idx = train["AwayTeam"].map(team_number).values
    home_goals = train["FTHG"].values.astype(float)
    away_goals = train["FTAG"].values.astype(float)

    # time decay: weight of every match from its age on the cutoff day
    days_ago = (cutoff_date - train["Date"]).dt.days.values
    weights = dc_time_weights(days_ago, xi)

    # starting values: all teams equal, defence value so that the average goal rate fits
    average_goals = (home_goals.mean() + away_goals.mean()) / 2
    start = np.zeros(2 * n_teams + 2)
    start[n_teams:2 * n_teams] = np.log(average_goals)
    start[2 * n_teams] = 0.2                      # home advantage
    start[2 * n_teams + 1] = 0.0                  # rho

    bounds = [(None, None)] * (2 * n_teams + 1) + [(DC_RHO_MIN, DC_RHO_MAX)]
    result = minimize(dc_negative_log_likelihood, start, method="L-BFGS-B", jac=True,
                      bounds=bounds,
                      args=(n_teams, home_idx, away_idx, home_goals, away_goals, weights))

    attack_raw = result.x[0:n_teams]
    attack = attack_raw - attack_raw.mean()
    defence = result.x[n_teams:2 * n_teams]

    model = {}
    model["attack"] = {}
    model["defence"] = {}
    for k in range(n_teams):
        model["attack"][teams[k]] = attack[k]
        model["defence"][teams[k]] = defence[k]
    model["home_adv"] = result.x[2 * n_teams]
    model["rho"] = result.x[2 * n_teams + 1]
    model["average_defence"] = defence.mean()
    model["converged"] = bool(result.success)
    return model


def team_strength(model, team):
    # (attack, defence) of a team. A team that is not in the training matches
    # (for example a newly promoted team) gets the average team: attack 0 (the
    # mean of the centred attack values) and the average defence value.
    if team in model["attack"]:
        return model["attack"][team], model["defence"][team]
    return 0.0, model["average_defence"]


def dixon_coles_predict(data, target_index, xi):
    # Predict the matches data.loc[target_index] (sorted by date).
    # The model is refitted once per block of DC_BLOCK_DAYS days. Every fit uses
    # ONLY matches played before the first day of the block, so no match is
    # ever predicted with its own result or with a later result. The time decay
    # is measured from the first day of the block (at most DC_BLOCK_DAYS - 1
    # days before the match itself). Every league is fitted separately.
    targets = data.loc[target_index]
    n = len(targets)
    outcome_probs = np.zeros((n, 3))
    prob_00 = np.zeros(n)
    prob_11 = np.zeros(n)

    dates = targets["Date"].tolist()
    leagues = targets["League"].values
    home_teams = targets["HomeTeam"].values
    away_teams = targets["AwayTeam"].values

    info = {"refits": 0, "not_converged": 0, "unseen_matches": 0, "rho_clipped": 0,
            "rho_values": [], "home_adv_values": []}

    start = 0
    while start < n:
        # the block: all matches within DC_BLOCK_DAYS days of the first match of the block
        block_start_date = dates[start]
        block_end_date = block_start_date + pd.Timedelta(days=DC_BLOCK_DAYS)
        end = start
        while end < n and dates[end] < block_end_date:
            end = end + 1

        for league in sorted(set(leagues[start:end])):
            positions = []
            for i in range(start, end):
                if leagues[i] == league:
                    positions.append(i)

            # training matches: same league, strictly before the block
            train = data[(data["League"] == league) & (data["Date"] < block_start_date)]
            model = fit_dixon_coles(train, block_start_date, xi)
            info["refits"] = info["refits"] + 1
            if not model["converged"]:
                info["not_converged"] = info["not_converged"] + 1
            info["rho_values"].append(model["rho"])
            info["home_adv_values"].append(model["home_adv"])

            lambda_home = np.zeros(len(positions))
            lambda_away = np.zeros(len(positions))
            for k in range(len(positions)):
                i = positions[k]
                if home_teams[i] not in model["attack"] or away_teams[i] not in model["attack"]:
                    info["unseen_matches"] = info["unseen_matches"] + 1
                attack_home, defence_home = team_strength(model, home_teams[i])
                attack_away, defence_away = team_strength(model, away_teams[i])
                lambda_home[k] = np.exp(attack_home + defence_away + model["home_adv"])
                lambda_away[k] = np.exp(attack_away + defence_home)

            clipped = clip_rho(model["rho"], lambda_home, lambda_away) != model["rho"]
            info["rho_clipped"] = info["rho_clipped"] + int(clipped.sum())

            probs, p00, p11 = dc_scoreline_probabilities(lambda_home, lambda_away, model["rho"])
            for k in range(len(positions)):
                outcome_probs[positions[k]] = probs[k]
                prob_00[positions[k]] = p00[k]
                prob_11[positions[k]] = p11[k]

        start = end

    result = {"outcome_probs": outcome_probs, "prob_00": prob_00, "prob_11": prob_11}
    result.update(info)
    return result


def choose_xi(data, valid_index, y_valid):
    # Choose the time decay xi from XI_GRID by log loss on the VALIDATION season
    # (the season right before the test season). Only matches before every
    # validation match are used to predict it, and the test season is never used.
    best_xi = XI_GRID[0]
    best_loss = None
    best_result = None
    valid_losses = {}
    for xi in XI_GRID:
        result = dixon_coles_predict(data, valid_index, xi)
        loss = log_loss(y_valid, result["outcome_probs"], labels=[0, 1, 2])
        valid_losses[xi] = loss
        if best_loss is None or loss < best_loss - 1e-12:
            best_loss = loss
            best_xi = xi
            best_result = result
    return best_xi, best_result, valid_losses


# ---------------------------------------------------------------
# Step 4e. Do the features add information beyond the bookmaker odds?
# ---------------------------------------------------------------
# Two multinomial logistic regressions (statsmodels MNLogit), outcome 0 = home win is the
# reference category:
#   model A: outcome ~ ln(pDraw / pHome) and ln(pAway / pHome) of the bookmaker
#   model B: model A plus the form and Elo features
# If B is not better than A, the features carry no information that the odds do not have.

FEATURE_NAMES = ["home_pts", "home_gf", "home_ga", "away_pts", "away_gf", "away_ga", "elo_diff"]


def odds_log_ratios(probs):
    # probs = margin-free bookmaker probabilities [home, draw, away], one row per match.
    # Returns two columns: ln(pDraw / pHome) and ln(pAway / pHome).
    ratios = np.zeros((len(probs), 2))
    ratios[:, 0] = np.log(probs[:, 1] / probs[:, 0])
    ratios[:, 1] = np.log(probs[:, 2] / probs[:, 0])
    return ratios


def coefficient_rows(result, season, model_name, term_names):
    # one row per coefficient: the two equations are "draw" and "away win" (each against home win)
    rows = []
    equations = ["draw vs home win", "away win vs home win"]
    for e in range(2):
        for t in range(len(term_names)):
            rows.append({"season": season, "model": model_name, "equation": equations[e],
                         "term": term_names[t],
                         "coefficient": round(float(result.params[t, e]), 4),
                         "std_error": round(float(result.bse[t, e]), 4),
                         "p_value": round(float(result.pvalues[t, e]), 4)})
    return rows


def fit_odds_models(odds_train, features_train, y_train, odds_test, features_test):
    # Fit A and B on the TRAINING matches only. The outcomes of the test matches are not
    # an input of this function, so they cannot be used. The features are standardised
    # with the mean and spread of the training matches.
    scaler = StandardScaler()
    features_train_scaled = scaler.fit_transform(features_train)
    features_test_scaled = scaler.transform(features_test)

    design_a_train = sm.add_constant(odds_train, has_constant="add")
    design_a_test = sm.add_constant(odds_test, has_constant="add")
    design_b_train = sm.add_constant(np.column_stack([odds_train, features_train_scaled]),
                                     has_constant="add")
    design_b_test = sm.add_constant(np.column_stack([odds_test, features_test_scaled]),
                                    has_constant="add")

    result_a = sm.MNLogit(y_train, design_a_train).fit(disp=0, maxiter=300)
    result_b = sm.MNLogit(y_train, design_b_train).fit(disp=0, maxiter=300)

    # Likelihood-ratio test of B against A: twice the gain in log-likelihood follows a
    # chi-square distribution if the extra features have no effect. The degrees of freedom
    # are the number of extra coefficients (7 features x 2 equations = 14).
    lr_statistic = 2 * (result_b.llf - result_a.llf)
    lr_df = int(result_b.df_model - result_a.df_model)
    lr_p_value = chi2.sf(lr_statistic, lr_df)

    fit = {}
    fit["probs_a"] = np.asarray(result_a.predict(design_a_test))
    fit["probs_b"] = np.asarray(result_b.predict(design_b_test))
    fit["llf_a"] = result_a.llf
    fit["llf_b"] = result_b.llf
    fit["lr_statistic"] = lr_statistic
    fit["lr_df"] = lr_df
    fit["lr_p_value"] = lr_p_value
    fit["converged"] = bool(result_a.mle_retvals["converged"] and result_b.mle_retvals["converged"])
    fit["result_a"] = result_a
    fit["result_b"] = result_b
    return fit


# ---------------------------------------------------------------
# Step 5. Backtest
# ---------------------------------------------------------------
def backtest(model_probs, odds, results):
    # bet 1 unit on an outcome when model_prob * odds > EDGE
    bets = 0
    profit = 0.0
    for i in range(len(results)):
        for outcome in range(3):
            p = model_probs[i, outcome]
            o = odds[i, outcome]
            if p * o > EDGE:
                bets = bets + 1
                if results[i] == outcome:
                    profit = profit + (o - 1)
                else:
                    profit = profit - 1
    return bets, profit


# ---------------------------------------------------------------
# Step 6. Calibration plot
# ---------------------------------------------------------------
def calibration_points(probs_home, actual_home):
    # split predictions into bins of width 0.1 and compare
    # the average predicted probability with the real frequency
    xs = []
    ys = []
    for k in range(10):
        low = k / 10
        high = (k + 1) / 10
        in_bin = (probs_home >= low) & (probs_home < high)
        if in_bin.sum() >= 10:
            xs.append(probs_home[in_bin].mean())
            ys.append(actual_home[in_bin].mean())
    return xs, ys


def save_calibration_plot(all_probs, y_test, title, path):
    actual_home = (y_test == 0).astype(float)
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="perfect calibration")
    for name in all_probs:
        xs, ys = calibration_points(all_probs[name][:, 0], actual_home)
        plt.plot(xs, ys, marker="o", label=name)
    plt.xlabel("Predicted home-win probability")
    plt.ylabel("Actual home-win frequency")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ---------------------------------------------------------------
# Step 7. One walk-forward fold
# ---------------------------------------------------------------
def evaluate_fold(data, test_season, use_clubelo):
    # Train on all seasons BEFORE test_season, test on test_season.
    train_seasons = SEASONS[:SEASONS.index(test_season)]

    elo_k, elo_home, valid_loss = tune_elo(data, test_season)
    print("Season %s: tuned Elo K=%d, home advantage=%d (validation log loss %.4f)"
          % (test_season, elo_k, elo_home, valid_loss))

    features = build_features(data, elo_k, elo_home)
    rows = data.loc[features.index]          # keep only rows that have features

    y = rows["FTR"].map(OUTCOME_CODE).values
    X = features.values
    home_goals = rows["FTHG"].values
    away_goals = rows["FTAG"].values

    is_train = rows["Season"].isin(train_seasons).values
    is_test = (rows["Season"] == test_season).values
    X_train = X[is_train]
    y_train = y[is_train]
    X_test = X[is_test]
    y_test = y[is_test]
    test_rows = rows[is_test]
    print("  train matches:", len(y_train), " test matches:", len(y_test))

    # collect probabilities of every approach in one dictionary
    all_probs = {}
    all_probs["Class frequencies"] = frequency_probabilities(y_train, len(y_test))
    all_probs["Bookmaker (Bet365)"] = bookmaker_probabilities(test_rows)
    model_probs, over_probs = three_model_probabilities(
        X_train, y_train, home_goals[is_train], away_goals[is_train], X_test)
    for name in model_probs:
        all_probs[name] = model_probs[name]

    # Blend of model and bookmaker. The weight is chosen on the VALIDATION
    # season (the season right before the test season). For that, the models are
    # trained only on the seasons before the validation season. The test
    # season is not used to choose the weight.
    valid_season = SEASONS[SEASONS.index(test_season) - 1]
    before_valid = SEASONS[:SEASONS.index(test_season) - 1]
    is_before_valid = rows["Season"].isin(before_valid).values
    is_valid = (rows["Season"] == valid_season).values
    valid_probs, unused_over = three_model_probabilities(
        X[is_before_valid], y[is_before_valid],
        home_goals[is_before_valid], away_goals[is_before_valid], X[is_valid])
    book_valid = bookmaker_probabilities(rows[is_valid])

    blend_probs = {}
    blend_weights = {}
    for name in model_probs:
        weight = choose_blend_weight(valid_probs[name], book_valid, y[is_valid])
        blend_weights[name] = weight
        blend_probs[name] = blend_probabilities(model_probs[name],
                                                all_probs["Bookmaker (Bet365)"], weight)
    print("  blend weights chosen on validation season %s (share of model):" % valid_season,
          ", ".join("%s %.1f" % (name, blend_weights[name]) for name in blend_weights))

    # bonus: the same model also predicts "more than 2.5 goals" (totals market)
    total_test = home_goals[is_test] + away_goals[is_test]
    total_train = home_goals[is_train] + away_goals[is_train]
    actual_over = (total_test > 2.5).astype(int)
    base_rate = (total_train > 2.5).mean()
    loss_over_poisson = log_loss(actual_over, over_probs, labels=[0, 1])
    loss_over_base = log_loss(actual_over, np.full(len(actual_over), base_rate), labels=[0, 1])
    print("  over 2.5 goals log loss - Poisson model: %.4f, base rate: %.4f"
          % (loss_over_poisson, loss_over_base))

    # reference: how often does the home team win?
    home_accuracy = (y_test == 0).mean()
    print("  always predicting a home win: accuracy %.3f" % home_accuracy)

    if use_clubelo:
        clubelo_diff = clubelo_difference(rows, features)
        if clubelo_diff is not None:
            correlation = np.corrcoef(features["elo_diff"].values, clubelo_diff.values)[0, 1]
            print("  correlation between our Elo and ClubElo: %.3f" % correlation)
            X_club = np.column_stack([X, clubelo_diff.values])
            all_probs["Logistic regression + ClubElo"] = logistic_probabilities(
                X_club[is_train], y_train, X_club[is_test])

    # Dixon-Coles model. It is kept in fold["dc"], NOT in all_probs, so that all
    # tables of the other approaches stay exactly as they are.
    # xi is chosen on the validation season, the blend weight too (as for the other models).
    started = time.perf_counter()
    valid_index = rows.index[is_valid]
    test_index = rows.index[is_test]
    best_xi, valid_result, valid_losses = choose_xi(data, valid_index, y[is_valid])
    test_result = dixon_coles_predict(data, test_index, best_xi)
    dc_weight = choose_blend_weight(valid_result["outcome_probs"], book_valid, y[is_valid])
    dc_blend_probs = blend_probabilities(test_result["outcome_probs"],
                                         all_probs["Bookmaker (Bet365)"], dc_weight)

    # for the draw-share check: scoreline probabilities of the Poisson model
    # (the same as dc_scoreline_probabilities with rho = 0)
    lambda_home_poisson, lambda_away_poisson = poisson_goal_rates(
        X_train, home_goals[is_train], away_goals[is_train], X_test)
    poisson_check, poisson_00, poisson_11 = dc_scoreline_probabilities(
        lambda_home_poisson, lambda_away_poisson, 0.0)
    if not np.allclose(poisson_check, all_probs["Poisson goals model"], atol=1e-9):
        raise ValueError("scoreline probabilities do not match the Poisson goals model")

    dc = {}
    dc["xi"] = best_xi
    dc["valid_losses"] = valid_losses
    dc["probs"] = test_result["outcome_probs"]
    dc["prob_00"] = test_result["prob_00"]
    dc["prob_11"] = test_result["prob_11"]
    dc["blend_weight"] = dc_weight
    dc["blend_probs"] = dc_blend_probs
    dc["info"] = test_result
    dc["poisson_prob_00"] = poisson_00
    dc["poisson_prob_11"] = poisson_11
    print("  Dixon-Coles: xi=%.3f chosen on validation season %s (validation log loss per xi: %s), "
          "blend weight %.1f" % (best_xi, valid_season,
                                 ", ".join("%.3f: %.4f" % (xi, valid_losses[xi]) for xi in XI_GRID),
                                 dc_weight))
    print("  Dixon-Coles: %d refits, %d not converged, %d matches with an unseen team, "
          "rho clipped %d times, %.0f seconds"
          % (test_result["refits"], test_result["not_converged"], test_result["unseen_matches"],
             test_result["rho_clipped"], time.perf_counter() - started))

    # Do the form and Elo features add information beyond the bookmaker odds?
    # (kept in fold["odds_models"], the other tables are not touched)
    odds_ratios_train = odds_log_ratios(bookmaker_probabilities(rows[is_train]))
    odds_ratios_test = odds_log_ratios(bookmaker_probabilities(rows[is_test]))
    odds_fit = fit_odds_models(odds_ratios_train, X_train, y_train, odds_ratios_test, X_test)
    odds_fit["train_matches"] = len(y_train)
    print("  odds model A vs A + features B: likelihood-ratio test on the training seasons "
          "chi2=%.2f (df=%d), p=%.4f; converged: %s"
          % (odds_fit["lr_statistic"], odds_fit["lr_df"], odds_fit["lr_p_value"],
             odds_fit["converged"]))

    fold = {}
    fold["season"] = test_season
    fold["odds_models"] = odds_fit
    fold["dc"] = dc
    fold["all_probs"] = all_probs
    fold["blend_probs"] = blend_probs
    fold["blend_weights"] = blend_weights
    fold["y_test"] = y_test
    fold["test_rows"] = test_rows
    fold["home_accuracy"] = home_accuracy
    return fold


def score_table(fold):
    # log loss, accuracy and backtest of every approach for one fold
    y_test = fold["y_test"]
    odds = fold["test_rows"][["B365H", "B365D", "B365A"]].values
    table = []
    for name in fold["all_probs"]:
        probs = fold["all_probs"][name]
        loss = log_loss(y_test, probs, labels=[0, 1, 2])
        accuracy = (probs.argmax(axis=1) == y_test).mean()
        bets, profit = backtest(probs, odds, y_test)
        if bets > 0:
            roi = 100 * profit / bets
        else:
            roi = 0.0
        table.append({"approach": name, "log_loss": round(loss, 4),
                      "accuracy": round(accuracy, 3), "bets": bets,
                      "profit_units": round(profit, 2), "roi_percent": round(roi, 1)})
    return pd.DataFrame(table)


def candidate_probabilities(fold):
    # every approach that is compared with the bookmaker: all approaches except
    # the bookmaker itself, plus the blends
    candidates = {}
    for name in fold["all_probs"]:
        if name != "Bookmaker (Bet365)":
            candidates[name] = fold["all_probs"][name]
    for name in fold["blend_probs"]:
        candidates["Blend: " + name] = fold["blend_probs"][name]
    return candidates


def bootstrap_table(fold):
    # bootstrap CI of (approach log loss - bookmaker log loss) for one fold.
    y_test = fold["y_test"]
    book_probs = fold["all_probs"]["Bookmaker (Bet365)"]
    candidates = candidate_probabilities(fold)

    table = []
    for name in candidates:
        mean_diff, low, high = bootstrap_loss_difference(candidates[name], book_probs, y_test)
        table.append({"season": fold["season"], "approach": name,
                      "loss_difference": round(mean_diff, 4),
                      "ci_low": round(low, 4), "ci_high": round(high, 4)})
    return pd.DataFrame(table)


def rps_table(folds):
    # mean RPS of every approach (and blend, and the bookmaker) in every test
    # season, as a table like walk_forward.csv (one row per approach)
    scores = {}
    for fold in folds:
        everything = candidate_probabilities(fold)
        everything["Bookmaker (Bet365)"] = fold["all_probs"]["Bookmaker (Bet365)"]
        for name in everything:
            if name not in scores:
                scores[name] = {}
            scores[name][fold["season"]] = rps_per_match(everything[name], fold["y_test"]).mean()
    table = pd.DataFrame(scores).T
    table["mean"] = table.mean(axis=1)
    table = table.round(4)
    table.index.name = "approach"
    return table


def dm_row(season, name, log_loss_diff, rps_diff):
    # one row of the Diebold-Mariano table from two arrays of per-match differences
    ll_stat, ll_p = diebold_mariano(log_loss_diff)
    rps_stat, rps_p = diebold_mariano(rps_diff)
    return {"season": season, "approach": name,
            "log_loss_diff": round(log_loss_diff.mean(), 4),
            "dm_stat_log_loss": round(ll_stat, 3), "dm_p_log_loss": round(ll_p, 6),
            "rps_diff": round(rps_diff.mean(), 5),
            "dm_stat_rps": round(rps_stat, 3), "dm_p_rps": round(rps_p, 6)}


def dm_table(folds):
    # Diebold-Mariano test of every approach against the bookmaker, for
    # log loss and for RPS, in every test season. Two summary "seasons":
    #   "mean"   = plain average of the per-season numbers (descriptive only,
    #              an average of p-values is not itself a p-value)
    #   "pooled" = ONE test on the matches of all test seasons together
    rows = []
    pooled_ll = {}
    pooled_rps = {}
    for fold in folds:
        y_test = fold["y_test"]
        book_probs = fold["all_probs"]["Bookmaker (Bet365)"]
        book_ll = match_log_losses(book_probs, y_test)
        book_rps = rps_per_match(book_probs, y_test)
        candidates = candidate_probabilities(fold)
        for name in candidates:
            diff_ll = match_log_losses(candidates[name], y_test) - book_ll
            diff_rps = rps_per_match(candidates[name], y_test) - book_rps
            rows.append(dm_row(fold["season"], name, diff_ll, diff_rps))
            if name not in pooled_ll:
                pooled_ll[name] = []
                pooled_rps[name] = []
            pooled_ll[name].append(diff_ll)
            pooled_rps[name].append(diff_rps)

    per_season = pd.DataFrame(rows)
    number_columns = ["log_loss_diff", "dm_stat_log_loss", "dm_p_log_loss",
                      "rps_diff", "dm_stat_rps", "dm_p_rps"]
    summary = []
    for name in pooled_ll:
        mine = per_season[per_season["approach"] == name]
        mean_row = {"season": "mean", "approach": name}
        for column in number_columns:
            mean_row[column] = round(mine[column].mean(), 6)
        summary.append(mean_row)
        summary.append(dm_row("pooled", name,
                              np.concatenate(pooled_ll[name]), np.concatenate(pooled_rps[name])))
    return pd.concat([per_season, pd.DataFrame(summary)], ignore_index=True)


def blend_table(fold):
    # chosen weight and test log loss of every blend, next to the bookmaker alone
    y_test = fold["y_test"]
    book_loss = log_loss(y_test, fold["all_probs"]["Bookmaker (Bet365)"], labels=[0, 1, 2])
    table = []
    for name in fold["blend_probs"]:
        model_loss = log_loss(y_test, fold["all_probs"][name], labels=[0, 1, 2])
        blend_loss = log_loss(y_test, fold["blend_probs"][name], labels=[0, 1, 2])
        table.append({"season": fold["season"], "model": name,
                      "model_weight": fold["blend_weights"][name],
                      "model_log_loss": round(model_loss, 4),
                      "bookmaker_log_loss": round(book_loss, 4),
                      "blend_log_loss": round(blend_loss, 4)})
    return pd.DataFrame(table)


# ---------------------------------------------------------------
# Step 8. Margin removal methods and favourite-longshot bias
# (the bookmaker probabilities use no training data, so there is no leakage here)
# ---------------------------------------------------------------
def margin_method_table(folds):
    # log loss and RPS of the bookmaker probabilities for every margin method,
    # on the same test matches as all other tables. The "mean" rows are plain
    # averages over the test seasons.
    rows = []
    for method in MARGIN_METHODS:
        losses = []
        scores = []
        for fold in folds:
            probs = bookmaker_probabilities(fold["test_rows"], method)
            loss = log_loss(fold["y_test"], probs, labels=[0, 1, 2])
            score = rps_per_match(probs, fold["y_test"]).mean()
            losses.append(loss)
            scores.append(score)
            rows.append({"method": method, "season": fold["season"],
                         "log_loss": round(loss, 4), "rps": round(score, 4)})
        rows.append({"method": method, "season": "mean",
                     "log_loss": round(np.mean(losses), 4), "rps": round(np.mean(scores), 4)})
    return pd.DataFrame(rows)


def bookmaker_calibration_table(data, method):
    # For each outcome (home win, draw, away win) and each bin of the bookmaker's
    # probability: how often did that outcome really happen?
    probs = bookmaker_probabilities(data, method)
    y = data["FTR"].map(OUTCOME_CODE).values
    outcome_names = ["Home win", "Draw", "Away win"]
    n_bins = int(round(1 / CALIBRATION_BIN_WIDTH))

    rows = []
    for outcome in range(3):
        predicted = probs[:, outcome]
        happened = (y == outcome).astype(float)
        for b in range(n_bins):
            low = b * CALIBRATION_BIN_WIDTH
            high = (b + 1) * CALIBRATION_BIN_WIDTH
            in_bin = (predicted >= low) & (predicted < high)
            if in_bin.sum() >= CALIBRATION_MIN_MATCHES:
                rows.append({"method": method, "outcome": outcome_names[outcome],
                             "bin_low": round(low, 2), "bin_high": round(high, 2),
                             "matches": int(in_bin.sum()),
                             "mean_predicted": round(predicted[in_bin].mean(), 4),
                             "actual_frequency": round(happened[in_bin].mean(), 4)})
    return pd.DataFrame(rows)


def favourite_longshot_summary(data, method):
    # Favourite-longshot bias: longshots (low probability) win LESS often than the
    # bookmaker probability says, favourites win MORE often.
    # All three outcomes of every match are put into three groups by probability.
    probs = bookmaker_probabilities(data, method)
    y = data["FTR"].map(OUTCOME_CODE).values
    happened = np.zeros((len(y), 3))
    for outcome in range(3):
        happened[:, outcome] = (y == outcome)
    predicted = probs.reshape(-1)               # all probabilities in one long list
    happened = happened.reshape(-1)             # same order

    groups = [("longshots (probability below 0.20)", 0.0, 0.20),
              ("middle (0.20 to 0.50)", 0.20, 0.50),
              ("favourites (0.50 or more)", 0.50, 1.01)]
    rows = []
    for name, low, high in groups:
        in_group = (predicted >= low) & (predicted < high)
        n = int(in_group.sum())
        mean_predicted = predicted[in_group].mean()
        actual = happened[in_group].mean()
        # binomial standard error of the actual frequency (treats observations as independent)
        std_error = np.sqrt(actual * (1 - actual) / n)
        rows.append({"method": method, "group": name, "observations": n,
                     "mean_predicted": round(mean_predicted, 4),
                     "actual_frequency": round(actual, 4),
                     "actual_minus_predicted": round(actual - mean_predicted, 4),
                     "std_error": round(std_error, 4)})
    return pd.DataFrame(rows)


def save_bookmaker_calibration_plot(table, path):
    # one panel per outcome, one line per margin method
    outcome_names = ["Home win", "Draw", "Away win"]
    plt.figure(figsize=(15, 5))
    for k in range(3):
        plt.subplot(1, 3, k + 1)
        plt.plot([0, 1], [0, 1], "k--", label="perfect calibration")
        for method in MARGIN_METHODS:
            part = table[(table["outcome"] == outcome_names[k]) & (table["method"] == method)]
            plt.plot(part["mean_predicted"], part["actual_frequency"], marker="o", label=method)
        plt.xlabel("Bookmaker probability (margin removed)")
        plt.ylabel("Actual frequency")
        plt.title(outcome_names[k])
        plt.legend()
        plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ---------------------------------------------------------------
# Step 9. Dixon-Coles results
# ---------------------------------------------------------------
def dc_comparison_table(folds):
    # log loss and RPS per season (and the mean) of: Dixon-Coles, the Poisson
    # model, the bookmaker (proportional margin removal), the class frequencies
    # and the Dixon-Coles / bookmaker blend. All on the same test matches.
    rows = []
    approaches = ["Class frequencies", "Bookmaker (Bet365)", "Poisson goals model",
                  "Dixon-Coles", "Blend: Dixon-Coles"]
    for approach in approaches:
        losses = []
        scores = []
        for fold in folds:
            if approach == "Dixon-Coles":
                probs = fold["dc"]["probs"]
            elif approach == "Blend: Dixon-Coles":
                probs = fold["dc"]["blend_probs"]
            else:
                probs = fold["all_probs"][approach]
            loss = log_loss(fold["y_test"], probs, labels=[0, 1, 2])
            score = rps_per_match(probs, fold["y_test"]).mean()
            losses.append(loss)
            scores.append(score)
            rows.append({"approach": approach, "season": fold["season"],
                         "log_loss": round(loss, 4), "rps": round(score, 4)})
        rows.append({"approach": approach, "season": "mean",
                     "log_loss": round(np.mean(losses), 4), "rps": round(np.mean(scores), 4)})
    return pd.DataFrame(rows)


def dc_tests_table(folds):
    # Bootstrap interval and Diebold-Mariano test of Dixon-Coles against the
    # bookmaker and against the Poisson model (difference = first - second,
    # positive = Dixon-Coles is worse), per season, as mean of the seasons, and
    # as one pooled test over all test matches.
    names = ["Dixon-Coles vs Bookmaker", "Dixon-Coles vs Poisson goals model",
             "Blend: Dixon-Coles vs Bookmaker"]
    rows = []
    pooled = {}
    for name in names:
        pooled[name] = {"model": [], "reference": [], "y": [], "diff_ll": [], "diff_rps": []}

    for fold in folds:
        y_test = fold["y_test"]
        book_probs = fold["all_probs"]["Bookmaker (Bet365)"]
        pairs = {names[0]: (fold["dc"]["probs"], book_probs),
                 names[1]: (fold["dc"]["probs"], fold["all_probs"]["Poisson goals model"]),
                 names[2]: (fold["dc"]["blend_probs"], book_probs)}
        for name in names:
            model_probs = pairs[name][0]
            reference_probs = pairs[name][1]
            diff_ll = match_log_losses(model_probs, y_test) - match_log_losses(reference_probs, y_test)
            diff_rps = rps_per_match(model_probs, y_test) - rps_per_match(reference_probs, y_test)
            unused_mean, low, high = bootstrap_loss_difference(model_probs, reference_probs, y_test)
            row = dm_row(fold["season"], name, diff_ll, diff_rps)
            row["ci_low"] = round(low, 4)
            row["ci_high"] = round(high, 4)
            rows.append(row)
            pooled[name]["model"].append(model_probs)
            pooled[name]["reference"].append(reference_probs)
            pooled[name]["y"].append(y_test)
            pooled[name]["diff_ll"].append(diff_ll)
            pooled[name]["diff_rps"].append(diff_rps)

    per_season = pd.DataFrame(rows)
    number_columns = ["log_loss_diff", "dm_stat_log_loss", "dm_p_log_loss",
                      "rps_diff", "dm_stat_rps", "dm_p_rps", "ci_low", "ci_high"]
    summary = []
    for name in names:
        mine = per_season[per_season["approach"] == name]
        mean_row = {"season": "mean", "approach": name}
        for column in number_columns:
            mean_row[column] = round(mine[column].mean(), 6)
        summary.append(mean_row)

        all_y = np.concatenate(pooled[name]["y"])
        all_model = np.vstack(pooled[name]["model"])
        all_reference = np.vstack(pooled[name]["reference"])
        unused_mean, low, high = bootstrap_loss_difference(all_model, all_reference, all_y)
        pooled_row = dm_row("pooled", name, np.concatenate(pooled[name]["diff_ll"]),
                            np.concatenate(pooled[name]["diff_rps"]))
        pooled_row["ci_low"] = round(low, 4)
        pooled_row["ci_high"] = round(high, 4)
        summary.append(pooled_row)

    table = pd.concat([per_season, pd.DataFrame(summary)], ignore_index=True)
    return table.rename(columns={"approach": "comparison"})


def dc_draw_share_table(folds):
    # Does the model predict the share of draws, of 0-0 and of 1-1 correctly?
    # Average predicted probability versus the real share, per season and pooled.
    # The bookmaker only gives a draw probability, not score probabilities.
    rows = []
    pooled = {}
    for model_name in ["Dixon-Coles", "Poisson goals model", "Bookmaker (Bet365)"]:
        pooled[model_name] = {"draw": [], "p00": [], "p11": []}
    pooled_actual = {"draw": [], "s00": [], "s11": []}

    for fold in folds:
        test_rows = fold["test_rows"]
        actual_draw = (test_rows["FTHG"].values == test_rows["FTAG"].values).astype(float)
        actual_00 = ((test_rows["FTHG"].values == 0) & (test_rows["FTAG"].values == 0)).astype(float)
        actual_11 = ((test_rows["FTHG"].values == 1) & (test_rows["FTAG"].values == 1)).astype(float)
        pooled_actual["draw"].append(actual_draw)
        pooled_actual["s00"].append(actual_00)
        pooled_actual["s11"].append(actual_11)

        predictions = {
            "Dixon-Coles": (fold["dc"]["probs"][:, 1], fold["dc"]["prob_00"], fold["dc"]["prob_11"]),
            "Poisson goals model": (fold["all_probs"]["Poisson goals model"][:, 1],
                                    fold["dc"]["poisson_prob_00"], fold["dc"]["poisson_prob_11"]),
            "Bookmaker (Bet365)": (fold["all_probs"]["Bookmaker (Bet365)"][:, 1], None, None)}
        for model_name in predictions:
            draw, p00, p11 = predictions[model_name]
            pooled[model_name]["draw"].append(draw)
            if p00 is not None:
                pooled[model_name]["p00"].append(p00)
                pooled[model_name]["p11"].append(p11)
            row = {"season": fold["season"], "model": model_name,
                   "predicted_draw": round(draw.mean(), 4), "actual_draw": round(actual_draw.mean(), 4)}
            if p00 is not None:
                row["predicted_0_0"] = round(p00.mean(), 4)
                row["actual_0_0"] = round(actual_00.mean(), 4)
                row["predicted_1_1"] = round(p11.mean(), 4)
                row["actual_1_1"] = round(actual_11.mean(), 4)
            rows.append(row)

    all_draw = np.concatenate(pooled_actual["draw"])
    all_00 = np.concatenate(pooled_actual["s00"])
    all_11 = np.concatenate(pooled_actual["s11"])
    for model_name in pooled:
        row = {"season": "pooled", "model": model_name,
               "predicted_draw": round(np.concatenate(pooled[model_name]["draw"]).mean(), 4),
               "actual_draw": round(all_draw.mean(), 4)}
        if len(pooled[model_name]["p00"]) > 0:
            row["predicted_0_0"] = round(np.concatenate(pooled[model_name]["p00"]).mean(), 4)
            row["actual_0_0"] = round(all_00.mean(), 4)
            row["predicted_1_1"] = round(np.concatenate(pooled[model_name]["p11"]).mean(), 4)
            row["actual_1_1"] = round(all_11.mean(), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def features_vs_odds_table(folds):
    # per test season: likelihood-ratio test of B against A (on the training seasons) and
    # the out-of-sample log loss of the bookmaker, A and B on the test season.
    # b_minus_a < 0 would mean that the features help. The "mean" row averages the seasons.
    # The "pooled" row is one comparison over all test matches (no likelihood-ratio test
    # there, because every season has a different training set).
    rows = []
    pooled_a = []
    pooled_b = []
    pooled_book = []
    pooled_y = []
    for fold in folds:
        fit = fold["odds_models"]
        y_test = fold["y_test"]
        book_probs = fold["all_probs"]["Bookmaker (Bet365)"]
        loss_a = match_log_losses(fit["probs_a"], y_test)
        loss_b = match_log_losses(fit["probs_b"], y_test)
        loss_book = match_log_losses(book_probs, y_test)
        unused_mean, low, high = bootstrap_loss_difference(fit["probs_b"], fit["probs_a"], y_test)
        unused_stat, dm_p = diebold_mariano(loss_b - loss_a)
        rows.append({"season": fold["season"], "train_matches": fit["train_matches"],
                     "test_matches": len(y_test),
                     "lr_statistic": round(fit["lr_statistic"], 3), "lr_df": fit["lr_df"],
                     "lr_p_value": round(fit["lr_p_value"], 4),
                     "log_loss_bookmaker": round(loss_book.mean(), 4),
                     "log_loss_a": round(loss_a.mean(), 4),
                     "log_loss_b": round(loss_b.mean(), 4),
                     "b_minus_a": round((loss_b - loss_a).mean(), 4),
                     "ci_low": round(low, 4), "ci_high": round(high, 4),
                     "dm_p_b_vs_a": round(dm_p, 4),
                     "a_minus_bookmaker": round((loss_a - loss_book).mean(), 4)})
        pooled_a.append(fit["probs_a"])
        pooled_b.append(fit["probs_b"])
        pooled_book.append(book_probs)
        pooled_y.append(y_test)
    table = pd.DataFrame(rows)

    mean_row = {"season": "mean"}
    for column in ["lr_statistic", "lr_p_value", "log_loss_bookmaker", "log_loss_a",
                   "log_loss_b", "b_minus_a", "ci_low", "ci_high", "dm_p_b_vs_a",
                   "a_minus_bookmaker"]:
        mean_row[column] = round(table[column].mean(), 4)

    all_a = np.vstack(pooled_a)
    all_b = np.vstack(pooled_b)
    all_book = np.vstack(pooled_book)
    all_y = np.concatenate(pooled_y)
    loss_a = match_log_losses(all_a, all_y)
    loss_b = match_log_losses(all_b, all_y)
    loss_book = match_log_losses(all_book, all_y)
    unused_mean, low, high = bootstrap_loss_difference(all_b, all_a, all_y)
    unused_stat, dm_p = diebold_mariano(loss_b - loss_a)
    pooled_row = {"season": "pooled", "test_matches": len(all_y),
                  "log_loss_bookmaker": round(loss_book.mean(), 4),
                  "log_loss_a": round(loss_a.mean(), 4), "log_loss_b": round(loss_b.mean(), 4),
                  "b_minus_a": round((loss_b - loss_a).mean(), 4),
                  "ci_low": round(low, 4), "ci_high": round(high, 4),
                  "dm_p_b_vs_a": round(dm_p, 4),
                  "a_minus_bookmaker": round((loss_a - loss_book).mean(), 4)}
    table = pd.concat([table, pd.DataFrame([mean_row, pooled_row])], ignore_index=True)

    # the mean and pooled rows have no value in some columns; keep the counts as whole numbers
    for column in ["train_matches", "test_matches", "lr_df"]:
        table[column] = table[column].astype("Int64")
    return table


def features_vs_odds_coefficients(folds):
    # coefficients of model B in every test season (fitted on the training seasons)
    term_names = ["constant", "ln(pDraw/pHome)", "ln(pAway/pHome)"] + FEATURE_NAMES
    rows = []
    for fold in folds:
        rows = rows + coefficient_rows(fold["odds_models"]["result_b"], fold["season"],
                                       "B", term_names)
    return pd.DataFrame(rows)


def dc_parameter_table(folds):
    # what was chosen and fitted for Dixon-Coles in every test season
    rows = []
    for fold in folds:
        dc = fold["dc"]
        info = dc["info"]
        row = {"season": fold["season"], "xi": dc["xi"],
               "mean_rho": round(np.mean(info["rho_values"]), 4),
               "mean_home_advantage": round(np.mean(info["home_adv_values"]), 4),
               "blend_weight": dc["blend_weight"],
               "refits": info["refits"], "not_converged": info["not_converged"],
               "matches_with_unseen_team": info["unseen_matches"],
               "rho_clipped": info["rho_clipped"]}
        for xi in XI_GRID:
            row["valid_log_loss_xi_" + str(xi)] = round(dc["valid_losses"][xi], 4)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def get_test_seasons(first_test_season):
    # walk-forward test seasons: first_test_season and every later season.
    # Elo tuning needs at least two seasons before a test season.
    if first_test_season not in SEASONS:
        raise ValueError("unknown season: " + first_test_season)
    position = SEASONS.index(first_test_season)
    if position < 2:
        raise ValueError("first test season needs two earlier seasons, "
                         "the earliest allowed is " + SEASONS[2])
    return SEASONS[position:]


def run(leagues, use_clubelo, first_test_season=FIRST_TEST_SEASON):
    fold_seasons = get_test_seasons(first_test_season)
    data = load_data(leagues)
    print("Leagues:", ", ".join(leagues))
    print("Matches loaded:", len(data))
    print("Test seasons:", ", ".join(fold_seasons))
    print()

    folds = []
    for season in fold_seasons:
        folds.append(evaluate_fold(data, season, use_clubelo))
        print()

    # log loss of every approach in every fold
    walk = {}
    for fold in folds:
        table = score_table(fold)
        for i in range(len(table)):
            name = table.loc[i, "approach"]
            if name not in walk:
                walk[name] = {}
            walk[name][fold["season"]] = table.loc[i, "log_loss"]
    walk_forward = pd.DataFrame(walk).T
    walk_forward["mean"] = walk_forward.mean(axis=1).round(4)
    walk_forward.index.name = "approach"

    print("Walk-forward log loss (lower is better):")
    print(walk_forward.to_string())

    # the last fold is the main result (most training data)
    last = folds[-1]
    results = score_table(last)
    print()
    print("Test season %s in detail:" % last["season"])
    print(results.to_string(index=False))

    # bootstrap confidence intervals and blends, for every test season
    bootstrap_parts = []
    blend_parts = []
    for fold in folds:
        bootstrap_parts.append(bootstrap_table(fold))
        blend_parts.append(blend_table(fold))
    bootstrap_results = pd.concat(bootstrap_parts, ignore_index=True)
    blend_results = pd.concat(blend_parts, ignore_index=True)

    print()
    print("Log loss minus bookmaker log loss, 95%% bootstrap CI (%d resamples)." % N_BOOTSTRAP)
    print("Negative = better than the bookmaker; the CI must stay below 0 to support that.")
    print(bootstrap_results.to_string(index=False))
    print()
    print("Blend of model and bookmaker (weight chosen on the validation season):")
    print(blend_results.to_string(index=False))

    # RPS and Diebold-Mariano test against the bookmaker
    rps_results = rps_table(folds)
    dm_results = dm_table(folds)
    print()
    print("Mean Ranked Probability Score (lower is better):")
    print(rps_results.to_string())
    print()
    print("Diebold-Mariano test against the bookmaker (difference = approach - bookmaker,")
    print("positive = worse than the bookmaker; p-values are two-sided, not corrected for")
    print("the many tests; with this many tests some 'significant' results are expected by chance):")
    print(dm_results.to_string(index=False))

    # bookmaker margin removal methods and favourite-longshot bias
    margin_results = margin_method_table(folds)
    calibration_parts = []
    summary_parts = []
    for method in MARGIN_METHODS:
        calibration_parts.append(bookmaker_calibration_table(data, method))
        summary_parts.append(favourite_longshot_summary(data, method))
    book_calibration = pd.concat(calibration_parts, ignore_index=True)
    longshot_results = pd.concat(summary_parts, ignore_index=True)
    print()
    print("Bookmaker probabilities with three margin removal methods (same test matches):")
    print(margin_results.to_string(index=False))
    print()
    print("Favourite-longshot check, all %d matches (actual - predicted < 0 for longshots"
          " would mean the bias exists):" % len(data))
    print(longshot_results.to_string(index=False))

    # Dixon-Coles model compared with Poisson, bookmaker and class frequencies
    dc_results = dc_comparison_table(folds)
    dc_tests = dc_tests_table(folds)
    dc_draws = dc_draw_share_table(folds)
    dc_parameters = dc_parameter_table(folds)
    print()
    print("Dixon-Coles versus the other approaches (same test matches):")
    print(dc_results.to_string(index=False))
    print()
    print("Dixon-Coles tests (difference = first - second, positive = Dixon-Coles is worse;")
    print("ci = 95% bootstrap interval of the log loss difference; p-values are not corrected for many tests):")
    print(dc_tests.to_string(index=False))
    print()
    print("Share of draws, 0-0 and 1-1: average predicted probability versus real share:")
    print(dc_draws.to_string(index=False))
    print()
    print("Dixon-Coles settings and fit diagnostics:")
    print(dc_parameters.to_string(index=False))

    # do the form and Elo features add information beyond the odds?
    odds_table = features_vs_odds_table(folds)
    odds_coefficients = features_vs_odds_coefficients(folds)
    print()
    print("Odds only (A) versus odds + form + Elo (B). lr_* = likelihood-ratio test on the training")
    print("seasons; the log losses are out-of-sample on the test season; b_minus_a < 0 would mean")
    print("that the features help (ci = 95% bootstrap interval, dm_p = Diebold-Mariano p-value):")
    print(odds_table.to_string(index=False))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    odds_table.to_csv(os.path.join(RESULTS_DIR, "features_vs_odds.csv"), index=False)
    odds_coefficients.to_csv(os.path.join(RESULTS_DIR, "features_vs_odds_coefficients.csv"), index=False)
    dc_results.to_csv(os.path.join(RESULTS_DIR, "dixon_coles.csv"), index=False)
    dc_tests.to_csv(os.path.join(RESULTS_DIR, "dixon_coles_tests.csv"), index=False)
    dc_draws.to_csv(os.path.join(RESULTS_DIR, "dixon_coles_draw_share.csv"), index=False)
    dc_parameters.to_csv(os.path.join(RESULTS_DIR, "dixon_coles_parameters.csv"), index=False)
    margin_results.to_csv(os.path.join(RESULTS_DIR, "margin_methods.csv"), index=False)
    book_calibration.to_csv(os.path.join(RESULTS_DIR, "bookmaker_calibration.csv"), index=False)
    longshot_results.to_csv(os.path.join(RESULTS_DIR, "favourite_longshot.csv"), index=False)
    save_bookmaker_calibration_plot(book_calibration,
                                    os.path.join(RESULTS_DIR, "bookmaker_calibration.png"))
    rps_results.to_csv(os.path.join(RESULTS_DIR, "rps.csv"))
    dm_results.to_csv(os.path.join(RESULTS_DIR, "dm.csv"), index=False)
    bootstrap_results.to_csv(os.path.join(RESULTS_DIR, "bootstrap.csv"), index=False)
    blend_results.to_csv(os.path.join(RESULTS_DIR, "blend.csv"), index=False)
    results.to_csv(os.path.join(RESULTS_DIR, "metrics.csv"), index=False)
    walk_forward.to_csv(os.path.join(RESULTS_DIR, "walk_forward.csv"))
    save_calibration_plot(last["all_probs"], last["y_test"],
                          "Calibration (test season " + last["season"] + ")",
                          os.path.join(RESULTS_DIR, "calibration.png"))
    print()
    print("Saved results/metrics.csv, results/walk_forward.csv, results/bootstrap.csv,")
    print("results/blend.csv, results/rps.csv, results/dm.csv, results/margin_methods.csv,")
    print("results/bookmaker_calibration.csv, results/favourite_longshot.csv,")
    print("results/bookmaker_calibration.png, results/dixon_coles.csv,")
    print("results/dixon_coles_tests.csv, results/dixon_coles_draw_share.csv,")
    print("results/dixon_coles_parameters.csv, results/features_vs_odds.csv,")
    print("results/features_vs_odds_coefficients.csv and results/calibration.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Football outcome prediction")
    parser.add_argument("--leagues", nargs="+", default=["E0"],
                        help="football-data.co.uk league codes: " + " ".join(LEAGUE_NAMES))
    parser.add_argument("--clubelo", action="store_true",
                        help="add ClubElo ratings as an extra comparison")
    parser.add_argument("--first-test-season", default=FIRST_TEST_SEASON,
                        help="first walk-forward test season, e.g. 1920 for 2019/20 "
                             "(allowed: " + " ".join(SEASONS[2:]) + ")")
    args = parser.parse_args()

    for code in args.leagues:
        if code not in LEAGUE_NAMES:
            parser.error("unknown league code: " + code)
    if args.first_test_season not in SEASONS[2:]:
        parser.error("--first-test-season must be one of: " + " ".join(SEASONS[2:]))

    run(args.leagues, args.clubelo, args.first_test_season)
