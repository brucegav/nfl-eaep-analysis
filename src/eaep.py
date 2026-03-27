"""
eaep.py

Core metric computation for the NFL ELO analysis project.
Implements the Era-Adjusted ELO Percentile (EAEP), an original metric that ranks
each franchise by season ELO percentile relative to contemporary franchises, averaged across
their full franchise history to produce an era-neutral measure of sustained competitive
performance.
"""

import pandas as pd



def elo_percentile(df):
    """
    Computes the Era-Adjusted ELO Percentile (EAEP) for each NFL franchise.

    Ranks each team by season ELO Percentile relative to all other teams in that specific
    NFL regular season, then averages across all other seasons the team has been in existence
    to produce  a single franchise-length performance metric that is era-neutral and cumulative-bias
    free.

    Args:
        df (pd.DataFrame): Pivot table of season-ending ELO values with seasons as index and team codes
        as columns.

    Returns:
        eaep (pd.Series): Mean ELO percentile rank per franchise (0-100).  
        rank_percentile (pd.DataFrame): Per-season percentile ranks, same shape as input df.
    """

    rank_percentile = (df.rank(pct=True, axis=1))
    eaep = (rank_percentile.mean() * 100).round(2)

    return eaep, rank_percentile



def net_score(series):
    """
    Computes a ELO differential for an NFL franchise.  Given a series of ELO values for all seasons of a 
    franchise's history, an integer value is returned, showing how many seasons above or below 
    league-average ELO (1500) the franchise has been, in total.
    Args:
        series (pd.Series): A Pandas Series of a franchises ELO values

    Returns:
        elo_diff (integer): The amount of seasons a franchise has been above or below league-average ELO.  
        Positive value means x seasons above league-average in total, negative means x seasons below 
        league-average in total.
    """

    above = (series > 1500).sum()
    below = (series.count() - above)
    elo_diff = above - below
    
    return elo_diff 



def get_sustained_peaks(trajectory, threshold=55, min_seasons=3):
    """
    Identifies seasons where an NFL franchise achieved sustained competitive success.
    Sustained success is defined as an EAEP value at or above the threshold for at
    least min_seasons consecutive seasons.

    Args:
        trajectory (pd.Series): A franchise's EAEP values indexed by season.
        threshold (int): Minimum EAEP value to qualify as a peak season. Default
                         of 55 was chosen because sustaining this value across an
                         entire franchise history places a team in the top 10 all-time
                         EAEP rankings.
        min_seasons (int): Minimum consecutive seasons required to qualify as a
                           sustained peak. Default of 3 filters out anomalous 1-2
                           season spikes that do not represent genuine eras of success.

    Returns:
        pd.Series: EAEP values for all seasons falling within a sustained peak window,
                   indexed by season. Returns empty Series if no sustained peaks exist.
    """
    
    full = trajectory.reindex(range(trajectory.index.min(),
                                    trajectory.index.max() + 1))
    above = full >= threshold
    runs = (above != above.shift()).cumsum()
    sustained = above.groupby(runs).transform('sum') >= min_seasons

    return full[above & sustained].dropna()