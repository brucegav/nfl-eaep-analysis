"""
data_prep.py

Data loading and preprocessing utilities for the NFL ELO analysis project.
Handles raw  data ingestion, seasonal ELO extraction, and master dataframe
construction from FiveThirtyEight's NFL ELO dataset.
"""

import pandas as pd
import os
import nflreadpy as nfl
import glob

# constants 
TEAM_FILE_MAP = {
    'ari': 'ARI',
    'atl': 'ATL',
    'bal': 'BAL',
    'buf': 'BUF',
    'car': 'CAR',
    'chi': 'CHI',
    'cin': 'CIN',
    'cle': 'CLE',
    'dal': 'DAL',
    'den': 'DEN',
    'det': 'DET',
    'gb': 'GB',
    'hou': 'HOU',
    'ind': 'IND',
    'jax': 'JAX',
    'kan': 'KC',
    'lac': 'LAC',
    'lar': 'LAR',
    'lvr': 'OAK',
    'mia': 'MIA',
    'min': 'MIN',
    'nep': 'NE',
    'nor': 'NO',
    'nyg': 'NYG',
    'nyj': 'NYJ',
    'phi': 'PHI',
    'pit': 'PIT',
    'sea': 'SEA',
    'sf': 'SF',
    'tbb': 'TB',
    'ten': 'TEN',
    'was': 'WSH'
}


PFR_TEAM_MAP = {
    'GNB': 'GB',
    'NWE': 'NE',
    'NOR': 'NO',
    'SFO': 'SF',
    'TAM': 'TB',
    'KAN': 'KC',
    'WAS': 'WSH',
    'RAM': 'LAR',
    'LAR': 'LAR',
    'SDG': 'LAC',
    'LAC': 'LAC',
    'OAK': 'OAK',
    'RAI': 'OAK',
    'LVR': 'OAK',
    'PHO': 'ARI',
    'ARI': 'ARI',
    'HOU': 'TEN',
    'TEN': 'TEN',
    'BAL': 'IND',
    'IND': 'IND',
    'BOS': 'NE',
    'ATL': 'ATL',
    'BUF': 'BUF',
    'CAR': 'CAR',
    'CHI': 'CHI',
    'CIN': 'CIN',
    'CLE': 'CLE',
    'DAL': 'DAL',
    'DEN': 'DEN',
    'DET': 'DET',
    'JAX': 'JAX',
    'MIA': 'MIA',
    'MIN': 'MIN',
    'NYG': 'NYG',
    'NYJ': 'NYJ',
    'PHI': 'PHI',
    'PIT': 'PIT',
    'SEA': 'SEA'
}

CURRENT_NFL_TEAMS = [
    'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN', 
    'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC', 'LAC', 'LAR', 'OAK', 'MIA', 
    'MIN', 'NE', 'NO', 'NYG', 'NYJ', 'PHI', 'PIT', 'SEA', 'SF', 'TB', 'TEN', 'WSH'
]

def load_elo_data(file_path):
    """
    Loads and cleans ELO dataset from FiveThirtyEight dataset.
    """
    
    elo_df = pd.read_csv(file_path)

    #create dict of all unique team codes to map or loop through all franchises
    team_codes = sorted(elo_df['team1'].unique())

    seasonal_elos = []
    
    for team in team_codes:
        # capture all games for this team
        team_games = elo_df[(elo_df['team1'] == team) | (elo_df['team2'] == team)].copy()
        #sort by date to find end of regular season
        team_games = team_games.sort_values('date')
        # identify ELO after each game
        team_games['final_elo'] = team_games.apply(
            lambda x: x['elo1_post'] if x['team1'] == team else x['elo2_post'],
            axis=1
        )
        # capture only last game of each regular season
        final_season_elo = team_games.groupby('season')['final_elo'].last().reset_index()
        final_season_elo['team'] = team
        seasonal_elos.append(final_season_elo)
    
    # create master dataframe for all teams and every season's final ELO
    master_df = pd.concat(seasonal_elos).pivot(index='season', columns='team', values='final_elo')


    # created filtered dataframe
    available_current_teams = [team for team in CURRENT_NFL_TEAMS if team in master_df.columns]
    master_df = master_df[available_current_teams].copy()

    return master_df

def load_coaching_data(file_path):
    """
    Loads and cleans coaching tenure data for coaching analysis from manually curated
    Pro Football Reference data
    """
    
    dfs = []

    # loop through team coaching_tenure files for each team to combine into master coaches_df
    for prefix, team_code in TEAM_FILE_MAP.items():
        try:
            coaches_df = pd.read_csv(os.path.join(file_path, f'{prefix}_coaching_tenures.csv'))
            coaches_df['team'] = team_code
            dfs.append(coaches_df)
            print(f'Ok: {prefix}')
        except FileNotFoundError:
            print(f'Missing: {prefix}')
        except Exception as e:
            print(f'Error: {prefix} - {e}')

    coaches_df = pd.concat(dfs, ignore_index=True)

    # clean rank columns
    coaches_df['AvRk'] = coaches_df['AvRk'].fillna(coaches_df['AvRnk'])
    coaches_df['BstRk'] = coaches_df['BstRk'].fillna(coaches_df['BstRnk'])
    coaches_df = coaches_df.drop(columns=['AvRnk', 'BstRnk'], errors='ignore')

    return coaches_df

def load_draft_data(manual_data_dir):
    """
    Loads and cleans NFL franchise historical draft pick data for franchise draft efficiency
    analysis from manually curated Pro Football Reference data and nflreadpy
    """
    
    # load 1980-2025 draft pick data from nflreadpy
    draft_picks = nfl.load_draft_picks()
    draft_df = draft_picks.to_pandas()

    # load 1960-1979 draft pick data from manually sourced PFR data csvs
    dfs = []
    for file in sorted(glob.glob(os.path.join(manual_data_dir, 'draft_*.csv'))):
        year = int(os.path.basename(file).split('_')[1].split('.')[0])
        df = pd.read_csv(file)
        df['season'] = year
        dfs.append(df)

    draft_pre_1980 = pd.concat(dfs, ignore_index=True)

    # handle issues with Cardinals and Rams abbreviations before applying team mapping
    draft_pre_1980.loc[draft_pre_1980['Tm'] == 'STL', 'Tm'] = 'ARI'
    # change 'STL' pre 1988 to 'ARI' and 1995-2015 to 'LAR'
    draft_df.loc[(draft_df['team'] == 'STL') & (draft_df['season'] <= 1987), 'team'] = 'ARI'
    draft_df.loc[(draft_df['team'] == 'STL') & (draft_df['season'] >= 1995), 'team'] = 'LAR'

    # apply PFR_TEAM_MAP to both dfs to align team abbreviations with ELO df, standardize 'team' column
    draft_pre_1980['team'] = draft_pre_1980['Tm'].map(PFR_TEAM_MAP)
    draft_df['team'] = draft_df['team'].map(PFR_TEAM_MAP)

    # combine and clean draft_df and draft_pre_1980
    draft_pre_1980_clean = draft_pre_1980[['season', 'team', 'DrAV']].rename(columns={'DrAV': 'dr_av'})
    draft_df_clean = draft_df[['season', 'team', 'dr_av']]
    draft_1960_2025 = pd.concat([draft_pre_1980_clean, draft_df_clean], ignore_index=True)

    return draft_1960_2025
    
    


def load_qb_data(file_path):
    """
    Loads franchise quarterback history data for franchise QB and Rate + analysis from manually
    curated Pro Football Reference data, used for modeling
    """
    
    qb_df = pd.read_csv(file_path)
    qb_df['team'] = qb_df['team'].str.strip().str.upper()

    return qb_df