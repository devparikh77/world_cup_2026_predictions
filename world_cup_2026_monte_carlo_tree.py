from __future__ import annotations

import importlib.util
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
OUT = ROOT / "outputs"
PREDICTOR_PATH = WORK / "world_cup_2026_first_round_predictor.py"

N_SIMS = 1000
SEED = 20260611
SIM_ELO_K = 18.0

STAGE_DEPTH = {
    "Group": 0,
    "R32": 1,
    "R16": 2,
    "QF": 3,
    "SF": 4,
    "Runner-up": 5,
    "Champion": 6,
}


GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}
TEAM_TO_GROUP = {team: group for group, teams in GROUPS.items() for team in teams}
TEAMS = list(TEAM_TO_GROUP)

R32_SLOTS = [
    ("M73", "2A", "2B"),
    ("M74", "1E", "3A/B/C/D/F"),
    ("M75", "1F", "2C"),
    ("M76", "1C", "2F"),
    ("M77", "1I", "3C/D/F/G/H"),
    ("M78", "2E", "2I"),
    ("M79", "1A", "3C/E/F/H/I"),
    ("M80", "1L", "3E/H/I/J/K"),
    ("M81", "1D", "3B/E/F/I/J"),
    ("M82", "1G", "3A/E/H/I/J"),
    ("M83", "2K", "2L"),
    ("M84", "1H", "2J"),
    ("M85", "1B", "3E/F/G/I/J"),
    ("M86", "1J", "2H"),
    ("M87", "1K", "3D/E/I/J/L"),
    ("M88", "2D", "2G"),
]

KNOCKOUT_TREE = {
    "M89": ("M73", "M75"),
    "M90": ("M74", "M77"),
    "M91": ("M76", "M78"),
    "M92": ("M79", "M80"),
    "M93": ("M83", "M84"),
    "M94": ("M81", "M82"),
    "M95": ("M86", "M88"),
    "M96": ("M85", "M87"),
    "M97": ("M89", "M90"),
    "M98": ("M93", "M94"),
    "M99": ("M91", "M92"),
    "M100": ("M95", "M96"),
    "M101": ("M97", "M98"),
    "M102": ("M99", "M100"),
    "M104": ("M101", "M102"),
}


def load_predictor():
    spec = importlib.util.spec_from_file_location("wc2026_predictor", PREDICTOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def elo_expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(rating_a - rating_b) / 400.0))


def result_score(goals_a: int, goals_b: int) -> float:
    if goals_a > goals_b:
        return 1.0
    if goals_a < goals_b:
        return 0.0
    return 0.5


def margin_multiplier(goal_diff: int, rating_diff: float) -> float:
    if goal_diff <= 1:
        return 1.0
    return math.log(goal_diff + 1.0) * (2.2 / ((abs(rating_diff) * 0.001) + 2.2))


def update_ratings(
    ratings: dict[str, float],
    team_a: str,
    team_b: str,
    goals_a: int,
    goals_b: int,
    weight: float = 1.0,
) -> None:
    rating_a = ratings[team_a]
    rating_b = ratings[team_b]
    expected_a = elo_expected(rating_a, rating_b)
    actual_a = result_score(goals_a, goals_b)
    margin = abs(goals_a - goals_b)
    change = SIM_ELO_K * weight * margin_multiplier(margin, rating_a - rating_b) * (actual_a - expected_a)
    ratings[team_a] = rating_a + change
    ratings[team_b] = rating_b - change


def scoreline_matrix(
    predictor,
    team_a: str,
    team_b: str,
    ratings: dict[str, float],
    goal_strengths: pd.DataFrame,
    draw_rate: float,
) -> np.ndarray:
    p_a, p_draw, p_b, xg_a, xg_b, details = predictor.scoreline_probabilities(
        team_a, team_b, ratings[team_a], ratings[team_b], goal_strengths, draw_rate
    )
    # Reconstruct the same low-score-adjusted Poisson matrix from the xG values.
    matrix = np.outer(predictor.poisson_pmf(xg_a), predictor.poisson_pmf(xg_b))
    matrix[0, 0] *= 1.10
    matrix[1, 1] *= 1.08
    matrix[1, 0] *= 1.03
    matrix[0, 1] *= 1.03
    return matrix / matrix.sum()


def draw_score(rng: np.random.Generator, matrix: np.ndarray) -> tuple[int, int]:
    idx = rng.choice(matrix.size, p=matrix.ravel())
    return tuple(map(int, np.unravel_index(idx, matrix.shape)))


def init_standings() -> dict[str, dict[str, dict[str, float]]]:
    return {
        group: {
            team: {"pts": 0, "gf": 0, "ga": 0, "gd": 0}
            for team in teams
        }
        for group, teams in GROUPS.items()
    }


def apply_group_result(table: dict[str, dict[str, float]], home: str, away: str, hg: int, ag: int) -> None:
    table[home]["gf"] += hg
    table[home]["ga"] += ag
    table[away]["gf"] += ag
    table[away]["ga"] += hg
    table[home]["gd"] = table[home]["gf"] - table[home]["ga"]
    table[away]["gd"] = table[away]["gf"] - table[away]["ga"]
    if hg > ag:
        table[home]["pts"] += 3
    elif ag > hg:
        table[away]["pts"] += 3
    else:
        table[home]["pts"] += 1
        table[away]["pts"] += 1


def rank_group(table: dict[str, dict[str, float]], ratings: dict[str, float]) -> list[str]:
    return sorted(
        table.keys(),
        key=lambda team: (
            table[team]["pts"],
            table[team]["gd"],
            table[team]["gf"],
            ratings[team],
        ),
        reverse=True,
    )


def assign_third_place_slots(ranks: dict[str, list[str]], standings, ratings: dict[str, float]) -> dict[str, str]:
    third_rows = []
    for group, ranking in ranks.items():
        team = ranking[2]
        row = standings[group][team]
        third_rows.append((group, team, row["pts"], row["gd"], row["gf"], ratings[team]))
    thirds = sorted(third_rows, key=lambda x: (x[2], x[3], x[4], x[5]), reverse=True)
    qualifiers = {group for group, *_ in thirds[:8]}
    third_team = {group: team for group, team, *_ in thirds}
    assignments = {}
    used = set()
    for match_id, _, slot in R32_SLOTS:
        if not slot.startswith("3"):
            continue
        candidates = [group for group in slot[1:].split("/") if group in qualifiers and group not in used]
        if not candidates:
            candidates = [group for group in qualifiers if group not in used]
        chosen = sorted(
            candidates,
            key=lambda group: next(row for row in thirds if row[0] == group)[2:],
            reverse=True,
        )[0]
        assignments[match_id] = third_team[chosen]
        used.add(chosen)
    return assignments


def resolve_slot(slot: str, ranks: dict[str, list[str]], third_assignments: dict[str, str], match_id: str) -> str:
    if slot.startswith("1") or slot.startswith("2"):
        rank = int(slot[0]) - 1
        group = slot[1]
        return ranks[group][rank]
    return third_assignments[match_id]


def knockout_winner(
    rng: np.random.Generator,
    predictor,
    team_a: str,
    team_b: str,
    ratings: dict[str, float],
    goal_strengths: pd.DataFrame,
    draw_rate: float,
) -> tuple[str, str, int, int, str]:
    matrix = scoreline_matrix(predictor, team_a, team_b, ratings, goal_strengths, draw_rate)
    ga, gb = draw_score(rng, matrix)
    if ga > gb:
        winner = team_a
        method = "90"
    elif gb > ga:
        winner = team_b
        method = "90"
    else:
        # Extra time: low-scoring extension based on roughly one-third of normal-match time.
        xg_a = max(0.05, matrix.sum(axis=1).dot(np.arange(matrix.shape[0])) * 0.30)
        xg_b = max(0.05, matrix.sum(axis=0).dot(np.arange(matrix.shape[1])) * 0.30)
        et_a = int(rng.poisson(xg_a))
        et_b = int(rng.poisson(xg_b))
        if et_a > et_b:
            winner = team_a
            method = "ET"
        elif et_b > et_a:
            winner = team_b
            method = "ET"
        else:
            p_a = elo_expected(ratings[team_a], ratings[team_b])
            winner = team_a if rng.random() < p_a else team_b
            method = "PEN"
    update_ratings(ratings, team_a, team_b, ga, gb, weight=1.20)
    loser = team_b if winner == team_a else team_a
    return winner, loser, ga, gb, method


def simulate_once(
    rng: np.random.Generator,
    predictor,
    fixtures: pd.DataFrame,
    initial_ratings: dict[str, float],
    goal_strengths: pd.DataFrame,
    draw_rate: float,
) -> tuple[dict[str, str], dict[str, list[str]], list[dict[str, object]]]:
    ratings = dict(initial_ratings)
    standings = init_standings()
    match_log = []

    group_fixtures = fixtures.iloc[:72]
    for match in group_fixtures.itertuples(index=False):
        matrix = scoreline_matrix(predictor, match.home_team, match.away_team, ratings, goal_strengths, draw_rate)
        hg, ag = draw_score(rng, matrix)
        group = TEAM_TO_GROUP[match.home_team]
        apply_group_result(standings[group], match.home_team, match.away_team, hg, ag)
        update_ratings(ratings, match.home_team, match.away_team, hg, ag, weight=1.0)
        match_log.append(
            {
                "stage": "Group",
                "home_team": match.home_team,
                "away_team": match.away_team,
                "home_goals": hg,
                "away_goals": ag,
                "winner": match.home_team if hg > ag else match.away_team if ag > hg else "Draw",
            }
        )

    ranks = {group: rank_group(table, ratings) for group, table in standings.items()}
    third_assignments = assign_third_place_slots(ranks, standings, ratings)
    stage_results = {team: "Group" for team in TEAMS}
    for ranking in ranks.values():
        for team in ranking[:2]:
            stage_results[team] = "R32"
    for team in third_assignments.values():
        stage_results[team] = "R32"
    winners = {}
    losers = {}

    for match_id, slot_a, slot_b in R32_SLOTS:
        team_a = resolve_slot(slot_a, ranks, third_assignments, match_id)
        team_b = resolve_slot(slot_b, ranks, third_assignments, match_id)
        winner, loser, ga, gb, method = knockout_winner(rng, predictor, team_a, team_b, ratings, goal_strengths, draw_rate)
        winners[match_id] = winner
        losers[match_id] = loser
        stage_results[winner] = "R16"
        match_log.append({"stage": "Round of 32", "home_team": team_a, "away_team": team_b, "home_goals": ga, "away_goals": gb, "winner": winner, "method": method})

    for match_id in [f"M{i}" for i in range(89, 97)]:
        source_a, source_b = KNOCKOUT_TREE[match_id]
        team_a, team_b = winners[source_a], winners[source_b]
        winner, loser, ga, gb, method = knockout_winner(rng, predictor, team_a, team_b, ratings, goal_strengths, draw_rate)
        winners[match_id] = winner
        losers[match_id] = loser
        stage_results[winner] = "QF"
        match_log.append({"stage": "Round of 16", "home_team": team_a, "away_team": team_b, "home_goals": ga, "away_goals": gb, "winner": winner, "method": method})

    for match_id in [f"M{i}" for i in range(97, 101)]:
        source_a, source_b = KNOCKOUT_TREE[match_id]
        team_a, team_b = winners[source_a], winners[source_b]
        winner, loser, ga, gb, method = knockout_winner(rng, predictor, team_a, team_b, ratings, goal_strengths, draw_rate)
        winners[match_id] = winner
        losers[match_id] = loser
        stage_results[winner] = "SF"
        match_log.append({"stage": "Quarterfinal", "home_team": team_a, "away_team": team_b, "home_goals": ga, "away_goals": gb, "winner": winner, "method": method})

    for match_id in ["M101", "M102"]:
        source_a, source_b = KNOCKOUT_TREE[match_id]
        team_a, team_b = winners[source_a], winners[source_b]
        winner, loser, ga, gb, method = knockout_winner(rng, predictor, team_a, team_b, ratings, goal_strengths, draw_rate)
        winners[match_id] = winner
        losers[match_id] = loser
        stage_results[winner] = "Final"
        match_log.append({"stage": "Semifinal", "home_team": team_a, "away_team": team_b, "home_goals": ga, "away_goals": gb, "winner": winner, "method": method})

    finalist_a, finalist_b = winners["M101"], winners["M102"]
    champion, runner_up, ga, gb, method = knockout_winner(rng, predictor, finalist_a, finalist_b, ratings, goal_strengths, draw_rate)
    stage_results[champion] = "Champion"
    stage_results[runner_up] = "Runner-up"
    match_log.append({"stage": "Final", "home_team": finalist_a, "away_team": finalist_b, "home_goals": ga, "away_goals": gb, "winner": champion, "method": method})

    return stage_results, ranks, match_log


def build_initial_ratings(predictor, ratings, market_values, odds_priors) -> dict[str, float]:
    enhanced = {}
    for team in TEAMS:
        parts = predictor.enhanced_rating(team, ratings, market_values, odds_priors, is_host_home=False)
        enhanced[team] = float(parts["final_rating"])
    return enhanced


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    predictor = load_predictor()
    results = pd.read_csv(predictor.RESULTS_CSV, parse_dates=["date"])
    base_ratings, _, draw_rate = predictor.train_elo(results)
    goal_strengths = predictor.train_goal_strengths(results, base_ratings)
    market_values = predictor.parse_transfermarkt_values()
    odds_priors = predictor.betting_priors()
    initial_ratings = build_initial_ratings(predictor, base_ratings, market_values, odds_priors)
    fixtures = results[
        (results["tournament"] == "FIFA World Cup")
        & (results["date"] >= predictor.TOURNAMENT_START)
        & results["home_score"].isna()
    ].sort_values(["date", "city", "home_team"]).reset_index(drop=True)

    rng = np.random.default_rng(SEED)
    stage_counts = {team: Counter() for team in TEAMS}
    group_rank_counts = {team: Counter() for team in TEAMS}
    champion_counts = Counter()
    final_pair_counts = Counter()
    sample_logs = []

    for sim in range(N_SIMS):
        stage_results, ranks, match_log = simulate_once(rng, predictor, fixtures, initial_ratings, goal_strengths, draw_rate)
        for group, ranking in ranks.items():
            for idx, team in enumerate(ranking, start=1):
                group_rank_counts[team][idx] += 1
                if idx <= 2:
                    stage_counts[team]["group_top2"] += 1
                if idx <= 3:
                    stage_counts[team]["group_top3"] += 1
        champion = next(team for team, stage in stage_results.items() if stage == "Champion")
        runner_up = next(team for team, stage in stage_results.items() if stage == "Runner-up")
        champion_counts[champion] += 1
        final_pair_counts[tuple(sorted([champion, runner_up]))] += 1
        for team, stage in stage_results.items():
            stage_counts[team][stage] += 1
            depth = STAGE_DEPTH[stage]
            if depth >= STAGE_DEPTH["R32"]:
                stage_counts[team]["reached_R32"] += 1
            if depth >= STAGE_DEPTH["R16"]:
                stage_counts[team]["reached_R16"] += 1
            if depth >= STAGE_DEPTH["QF"]:
                stage_counts[team]["reached_QF"] += 1
            if depth >= STAGE_DEPTH["SF"]:
                stage_counts[team]["reached_SF"] += 1
            if depth >= STAGE_DEPTH["Runner-up"]:
                stage_counts[team]["reached_Final"] += 1
        if sim < 5:
            for row in match_log:
                sample_logs.append({"simulation": sim + 1, **row})

    rows = []
    for team in TEAMS:
        row = {
            "team": team,
            "group": TEAM_TO_GROUP[team],
            "avg_group_rank": sum(rank * count for rank, count in group_rank_counts[team].items()) / N_SIMS,
            "top2_probability": stage_counts[team]["group_top2"] / N_SIMS,
            "top3_probability": stage_counts[team]["group_top3"] / N_SIMS,
            "round_of_32_probability": stage_counts[team]["reached_R32"] / N_SIMS,
            "round_of_16_probability": stage_counts[team]["reached_R16"] / N_SIMS,
            "quarterfinal_probability": stage_counts[team]["reached_QF"] / N_SIMS,
            "semifinal_probability": stage_counts[team]["reached_SF"] / N_SIMS,
            "final_probability": stage_counts[team]["reached_Final"] / N_SIMS,
            "runner_up_probability": stage_counts[team]["Runner-up"] / N_SIMS,
            "champion_probability": stage_counts[team]["Champion"] / N_SIMS,
        }
        rows.append(row)
    probabilities = pd.DataFrame(rows).sort_values("champion_probability", ascending=False)
    probabilities.to_csv(OUT / "world_cup_2026_mc_tree_probabilities.csv", index=False)

    final_pairs = pd.DataFrame(
        [
            {"final_pair": " vs ".join(pair), "count": count, "probability": count / N_SIMS}
            for pair, count in final_pair_counts.most_common(25)
        ]
    )
    final_pairs.to_csv(OUT / "world_cup_2026_mc_tree_common_finals.csv", index=False)
    pd.DataFrame(sample_logs).to_csv(OUT / "world_cup_2026_mc_tree_sample_paths.csv", index=False)

    metadata = pd.DataFrame(
        [
            {
                "n_simulations": N_SIMS,
                "seed": SEED,
                "sim_elo_k": SIM_ELO_K,
                "draw_rate": draw_rate,
                "description": "Sequential Monte Carlo tournament tree with rating updates after each simulated match.",
            }
        ]
    )
    metadata.to_csv(OUT / "world_cup_2026_mc_tree_metadata.csv", index=False)

    print(probabilities.head(15).to_string(index=False))
    print("\nMost common finals")
    print(final_pairs.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
