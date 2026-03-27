"""
visualizations.py

Module that provides frequently used plots throughout the analysis.  These can be used on any
NFL franchise in the dataset and are limited by specific team vs specific team analysis cases.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



def plot_eaep_trajectory(rank_percentile, team, color):
    """
    Plots a specific NFL franchises EAEP trajectory over the life of their franchise in a line chart.
    Provides a trend line to map the overall EAEP trend and a league average baseline to compare against.

    Args:
        rank_percentile (pd.DataFrame): a dataframe of ELO percentile rankings for an NFL franchise over every season
                                of their existence
        team (str): the name of an NFL franchise
        color (str): chosen color of the teams trajectory line in the plot

    Returns:
        fig (matplotlib.figure.Figure): The trajectory line chart figure object
    """
    
    trajectory = rank_percentile[team].dropna() * 100
    trajectory_smooth = trajectory.rolling(window=5, center=True, min_periods=1).mean()

    x = np.arange(len(trajectory_smooth))
    slope, intercept = np.polyfit(x, trajectory_smooth.values, 1)
    trend_line = slope * x + intercept

    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(trajectory_smooth.index, trajectory_smooth.values,
            linewidth=2.5, color=color, label='ELO Percentile')

    ax.plot(trajectory_smooth.index, trend_line,
            color='lightblue', linewidth=2, linestyle='--', alpha=0.7, label='Overall Trend')

    ax.axhline(50, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label='League Average')

    ax.set_xlabel('Season', fontsize=12)
    ax.set_ylabel('ELO Percentile Rank', fontsize=12)
    ax.set_title(f'{team} ELO Percentile Trajectory', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.4)

    plt.tight_layout()

    return fig



def plot_eaep_barh(eaep):
    """
    Plots a horizontal bar chart of Era-Adjusted ELO Percentile (EAEP) for all NFL franchises,
    sorted from highest to lowest. Bars above 50 are highlighted in red, below in gray.

    Args:
        eaep (pd.Series): Mean ELO percentile rank per franchise (0-100), indexed by team code.

    Returns:
        fig (matplotlib.figure.Figure): The EAEP horizontal bar chart figure object.
    """
    
    plot_eaep = eaep.sort_values(ascending=False)
    colors = ['red' if val > 50 else 'gray' for val in plot_eaep]

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.barh(plot_eaep.index, plot_eaep.values, color=colors)
    ax.set_xlabel('EAEP', fontsize=12)
    ax.set_ylabel('Teams', fontsize=12)
    ax.set_xlim(plot_eaep.values.min() - 3, plot_eaep.values.max() + 3)
    ax.set_title('NFL Era-Adjusted ELO Percentiles (1920-2021)', fontsize=16, fontweight='bold')
    ax.invert_yaxis()

    plt.tight_layout()

    return fig



def plot_coaching_scatter(coaches_df, team):
    """
    Plots a scatter plot comparison of Average EAEP during tenure and length of tenure for every head coach
    an NFL franchise has had (or that is available in the dataset).  Does not include interim coaches.  Interim 
    head coaches are defined as those that coached less than 20 consecutive games for the franchise.

    Args:
        coaches_df (pd.DataFrame):  a dataframe of every head coach an NFL franchise has had, their tenure length,
                                    and average EAEP during tenure.  Does not include interim head coaches.

        team (str):  An NFL franchise.

    Returns:
        fig (matplotlib.figure.Figure):  The Coaching Tenure scatter plot
    """

    fig, ax = plt.subplots(figsize=(14, 9))

    ax.scatter(
        coaches_df['tenure_seasons'],
        coaches_df['avg_eaep'],
        s=120,
        alpha=0.8,
        edgecolors='white',
        linewidths=0.5,
        zorder=5
    )

    for _, row in coaches_df.iterrows():
        ax.annotate(
            f"{row['coach'].title()}",
            (row['tenure_seasons'], row['avg_eaep']),
            textcoords='offset points',
            xytext=(8, 4),
            fontsize=8,
            alpha=0.5
        )

    ax.axhline(50, color='black', linestyle='--', linewidth=1, alpha=0.4, label='League Average EAEP')
    ax.axvline(coaches_df['tenure_seasons'].median(), color='gray', 
               linestyle='--', linewidth=1, alpha=0.4, label='Median Tenure')
    ax.set_xlabel('Tenure (Seasons)', fontsize=12)
    ax.set_ylabel('Average EAEP During Tenure', fontsize=12)
    ax.set_title(f'{team} Head Coaching Tenures: Performance vs Longevity',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()

    return fig
               
        



def plot_franchise_heatmap(decades_df):
    """
    Plots a heatmap of average franchise ELO by decade for all NFL franchises.
    Color scale is centered at league average ELO (1500).

    Args:
        decades_df (pd.DataFrame): DataFrame of average ELO values grouped by decade,
                                   with decades as index and team codes as columns.

    Returns:
        fig (matplotlib.figure.Figure): The franchise decade heatmap figure object.
    """
    
    fig, ax = plt.subplots(figsize=(18, 8))

    sns.heatmap(
        decades_df.T,
        cmap='RdYlGn_r',
        center=1500,
        annot=True,
        fmt='.0f',
        linewidths=0.5,
        cbar_kws={'label': 'Average ELO'},
        ax=ax
    )

    ax.set_title('NFL Franchise ELO by Decade', fontsize=20, pad=20)
    ax.set_xlabel('Decade', fontsize=14)
    ax.set_ylabel('Franchise', fontsize=14)

    plt.tight_layout()

    return fig


def plot_gantt_runs(top_3_runs, team_colors):
    """
    Plots a Gantt-style chart showing sustained top 5 ELO percentile runs of 3 or more
    consecutive seasons for NFL franchises across the history of the NFL.

    Args:
        top_3_runs (pd.DataFrame): DataFrame of sustained top 5 ELO percentile runs with
                                   columns team, start, end, and duration.
        team_colors (dict): Dictionary mapping team codes to hex color strings.

    Returns:
        fig (matplotlib.figure.Figure): The Gantt chart figure object.
    """
    
    fig, ax = plt.subplots(figsize=(20, 14))

    teams = top_3_runs['team'].unique()
    team_positions = {team: i for i, team in enumerate(sorted(teams))}

    for _, row in top_3_runs.iterrows():
        y = team_positions[row['team']]
        ax.barh(y, row['end'] - row['start'],
                left=row['start'],
                height=0.6,
                color=team_colors.get(row['team'], 'gray'),
                alpha=0.8)

    ax.set_yticks(range(len(team_positions)))
    ax.set_yticklabels(sorted(teams))
    ax.set_xlabel('Season', fontsize=12)
    ax.set_title('NFL Franchises: Sustained Top 5 ELO Percentile Runs (3+ Seasons)',
                 fontsize=16, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_xlim(1920, 2022)
    ax.set_xticks(range(1920, 2023, 5))
    ax.tick_params(axis='x', rotation=45)

    milestones = {
        1950: 'NFL/AAFC\nMerger',
        1970: 'AFL/NFL\nMerger',
        1993: 'Free Agency &\nSalary Cap'
    }

    for year, label in milestones.items():
        ax.axvline(year, color='black', linewidth=1, linestyle='--', alpha=0.4)
        ax.text(year + 0.3, len(team_positions) - 0.5, label,
                fontsize=7, alpha=0.7, va='top')

    plt.tight_layout()

    return fig