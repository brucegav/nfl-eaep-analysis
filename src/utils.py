"""
utils.py

Miscellaneous functions which allow for computing franchise ELO ratings for future NFL seasons.
"""

import numpy as np
import pandas as pd 


def expected_result(elo_a, elo_b, home_field=True):
    """
    Function that returns the expected score of an NFL game based on the ELO win probability
    formula (FiveThirtyEight, Arpad Elo).  400 is a constant used as a fixed scale factor.

    Args:
        elo_a (integer): current pre-game ELO rating for Team A
        elo_b (integer): current pre-game ELO rating for Team B
        home_field=True (bool): if Team A is playing at home, apply home_advantage

    Returns:
        expected_result (float): returns a probability score between 0 and 1 representing Team A's
                                   chance of winning.  A an equal chance returns 0.5. This score
                                   gets compared against the real-life result of the game to define
                                   the ELO rating.
    """
    
    # provides the standard home field advantage score for ELO, else 0
    home_advantage = 65 if home_field else 0
    
    expected_result = 1 / (1 + 10 ** ((elo_b - elo_a - home_advantage) / 400))

    return expected_result



def margin_multiplier(score_diff):
    """
    Function that defines how much a specific game result should move the resulting ELO
    rating.  It scales up for larger margins of victory and down for small margins of victory.

    Args:
        score_diff (integer): the score differential of Team A and Team B in an NFL game.

    Returns:
        margin_multiplier (float): the amount an ELO rating should be scaled (up or down)
                                    based on the margin of victory in an NFL game
    
    """

    margin_multiplier = np.log(abs(score_diff) + 1) * (2.2 / (abs(score_diff) * 0.001 + 2.2))

    return margin_multiplier



def compute_elo_ratings(games_df, initial_elos, K=20):
    """
    Computes season-ending ELO ratings for NFL teams over a specified timeframe
    by applying the ELO update formula game-by-game.

    Args:
        games_df (pd.DataFrame): Game-by-game results with home/away teams and scores.
        initial_elos (dict): Starting ELO values for each team as {team: elo}.
        K (int): Update factor controlling rating volatility. Defaults to 20.

    Returns:
        season_ending_elos (dict): Season-ending ELO ratings as {season: {team: elo}}.
    """
    current_elos = initial_elos.copy()
    season_ending_elos = {}
    prev_season = None

    for _, game in games_df.iterrows():
        home = game['home_team']
        away = game['away_team']

        if home not in current_elos or away not in current_elos:
            continue

        # mean reversion at start of each new season
        if prev_season is not None and game['season'] != prev_season:
            for team in current_elos:
                current_elos[team] = current_elos[team] * 0.75 + 1505 * 0.25
            season_ending_elos[prev_season] = current_elos.copy()

        prev_season = game['season']

        if pd.isna(game['home_score']) or pd.isna(game['away_score']):
            continue

        home_score = game['home_score']
        away_score = game['away_score']
        neutral = game['location'] == 'Neutral'

        if home_score > away_score:
            actual = 1
        elif home_score < away_score:
            actual = 0
        else:
            actual = 0.5

        exp = expected_result(current_elos[home], current_elos[away], home_field=not neutral)

        score_diff = abs(home_score - away_score)
        mult = margin_multiplier(score_diff)

        change = K * mult * (actual - exp)
        current_elos[home] += change
        current_elos[away] -= change

    # save final season
    season_ending_elos[prev_season] = current_elos.copy()

    return season_ending_elos