"""
features.py

Feature  and target variable creation and insertion for modeling and analysis.  Uses existing datasets 
to aggregate specific features for quarterbacks, head coaches, and franchises, which can then be used 
in machine learning models (logistic regression and random forest classifier specifically).  

Features created include:

- coach_games_at_season: how many games a franchise has had this head coach up until the current season
- eaep: a team's EAEP for the current season
- trailing_3yr_eaep: a team's EAEP for the previous 3 seasons
- total_dr_av: a team's total draft approximate value for a specific season
- trailing_3yr_draft_av: a team's total draft approximate value for the 3 previous seasons 
- qb_rate_plus: a team's primary starter QB's Rate + for the current season
- has_franchise_qb: whether or not a team has a franchise QB in a specific season
"""

import pandas as pd

# ===========================================================
# ROW-LEVEL FUNCTIONS
# ===========================================================


def get_season_length(season):
    """
    Defines season length for each era of NFL history.  Used to standardize the definition of 'season'
    since regular season lengths have varied across NFL history.  

    Args:
        season (int): An NFL season

    Returns:
        integer value: deoending on the timeframe, returns 14 (1977 and prior), 16 (2020 and prior),
        or 17 (2021 and beyond)
    """

    if season <= 1977:
        return 14
    elif season <= 2020:
        return 16
    else:
        return 17
    


def get_coach_features(team, season, coaches_df):
    """
    Creates the coach_games_at_season feature for modeling.  The feature is an integer value holding
    the number of games an NFL head coach has coached a specific NFL up to the current season.
    It is used to measure franchise coaching stability - frequent and consecutive low values represents
    low stability, the inverse represents high stability


    Args:
        team (string): An NFL team
        season (integer): An NFL regular season
        coaches_df (pd.DataFrame): A dataframe consisting of historical head coach data,
        organized by NFL team and season.

    Returns:
        total_games (int): The number of regular season NFL games a head coach has coached for 
        a specific franchise up until the current season.
    """

    coach_row = coaches_df[
        (coaches_df['team'] == team) &
        (coaches_df['start_year'] <= season) &
        (coaches_df['end_year'] >= season)
    ]
    if len(coach_row) == 0:
        return None
    
    start = coach_row.iloc[0]['start_year']
    total_games = 0
    
    for yr in range(start, season):
        total_games += get_season_length(yr)
    
    return total_games



def get_qb_features(team, season, qb_df):
    """
    Creates the qb_rate_plus and has_franchise_qb features for modeling.  Uses qb_df dataframe to obtain 
    rate_plus for the team's QB in that season.  If no QB row exists for a season, marks 0 to denote the 
    franchise not having a franchise QB in that season. Franchise quarterback is 
    defined as having served as the primary starter (most games started) for a minimum of 3 consecutive seasons,
    with the same franchise.  Rate + is a QB's passer rating normalized to the league average for that season with 
    100 representing an average QB.

    Args:
        team (string): An NFL franchise
        season (integer): An NFL regular season
        qb_df (pd.DataFrame): Dataframe consisting of historical quarterback data, organized by team and NFL
        season
    Returns:
        qb_rate_plus (float): A franchise quarterback Rate + value for a specific season
        has_franchise_qb (integer): A binary variable that is either 1 for yes or 0 for no.
    """

    qb_row = qb_df[
        (qb_df['team'] == team) &
        (qb_df['start_year'] <= season) &
        (qb_df['end_year'] >= season)
    ]

    if len(qb_row) == 0:
        return 90, 0

    return qb_row.iloc[0]['rate_plus'], 1



def is_sustained_decline(group):
    """
    Creates target variable, 'is_sustained_decline' for modeling.  Sustained decline is defined as 
    3 consecutive years below league-average EAEP (50)

    Args:
        group (pd.DataFrame): A dataframe containing the EAEP values for a team in every season of its history

    Returns:
        group (pd.DataFrame): A dataframe containing the season, EAEP value and sustained_decline variable (0 = no or 1 = yes)
    """
    
    group = group.sort_values('season')
    below_50 = (group['eaep'] < 50).astype(int)
    group['sustained_decline'] = (
        below_50.shift(-1).fillna(0).astype(int) &
        below_50.shift(-2).fillna(0).astype(int) &
        below_50.shift(-3).fillna(0).astype(int)
    ).astype(int)

    return group


# ============================================
# DATAFRAME-LEVEL BUILDER FUNCTIONS
# ============================================


def build_qb_features(feature_df, qb_df):
    """
    Applies get_qb_features across all rows of feature_df and assigns results to standardize
    column names.

    Args:
        feature_df (pd.DataFrame): A dataframe consisting of features data
        qb_df (pd.DataFrame):  A dataframe consisting of historical NFL quarterback data

    Returns: 
        feature_df (pd.DataFrame): A an updated dataframe consisting of features data, now including
        qb_rate_plus and has_franchise_qb
    """

    results = feature_df.apply(
        lambda row: get_qb_features(row['team'], row['season'], qb_df),
        axis=1
    )

    feature_df['qb_rate_plus'] = results.apply(lambda x: x[0])
    feature_df['has_franchise_qb'] = results.apply(lambda x: x[1])

    return feature_df



def build_coach_features(feature_df, coaches_df):
    """
    Applies get_coach_features across all rows of the feature_df and assigns result to 
    a standardized column name.

    Args:
        feature_df (pd.DataFrame): Feature dataframe with season and team columns.
        coaches_df (pd.DataFrame): Dataframe with historical NFL head coach data

    Returns:
        feature_df (pd.DataFrame): Updated feature dataframe with coach_games_at_season column.
    """

    feature_df['coach_games_at_season'] = feature_df.apply(
        lambda row: get_coach_features(row['team'], row['season'], coaches_df),
        axis=1
    )

    return feature_df


def build_eaep_features(rank_percentile):
    """
    Constructs the base feature dataframe with eaep and trailing_3yr_eaep
    features from the rank_percentile matrix

    Args:
        rank_percentile (pd.DataFrame): per-season ELO percentile ranks, seasons
        as index, teams as columns.

    Returns:
        feature_df (pd.DataFrame): Long-format dataframe with columns season,
        team, eaep, trailing_3yr_eaep.
    """

    eaep_long = rank_percentile.stack().reset_index()
    eaep_long.columns = ['season', 'team', 'eaep']
    eaep_long['eaep'] = (eaep_long['eaep'] * 100).round(2)
    eaep_long = eaep_long[eaep_long['season'] >= 1960]

    eaep_long['trailing_3yr_eaep'] = (
        eaep_long.groupby('team')['eaep']
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )

    return eaep_long



def build_decline_target(feature_df):
    """
    Applies is_sustained_decline across all franchises and inserts the sustained_decline
    target variable into the feature dataframe

    Args:
        feature_df (pd.DataFrame): Feature dataframe with season and team columns.

    Returns:
        feature_df (pd.DataFrame): Updated feature dataframe with sustained_decline
        column.
    """

    return feature_df.groupby('team', group_keys=False).apply(is_sustained_decline)


def build_draft_features(feature_df, draft_1960_2025):
    """
    Builds total_dr_av and trailing_3yr_draft_av features and merges them into feature_df.

    Args:
        draft_1960_2025 (pd.DataFrame): Combined draft AV data from 1960-2025.
        feature_df (pd.DataFrame): Feature dataframe with season and team columns.

    Returns:
        feature_df (pd.DataFrame): Updated feature dataframe with total_dr_av and 
        trailing_3yr_draft_av columns.
    """

    draft_totals = draft_1960_2025.groupby(['season', 'team'])['dr_av'].sum().reset_index()
    draft_totals.rename(columns={'dr_av': 'total_dr_av'}, inplace=True)

    feature_df = feature_df.merge(draft_totals, on=['season', 'team'], how='left')

    feature_df['trailing_3yr_draft_av'] = (
        feature_df.groupby('team')['total_dr_av']
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )

    return feature_df