"""
__init__.py
"""

from src.data_prep import load_elo_data, load_coaching_data, load_draft_data, load_qb_data
from src.eaep import elo_percentile, net_score, get_sustained_peaks
from src.features import build_eaep_features, build_coach_features, build_qb_features, build_draft_features, build_decline_target
from src.modeling import train_test_split_temporal, train_logistic_regression, train_random_forest, evaluate_model
from src.utils import expected_result, margin_multiplier, compute_elo_ratings
from src.visualizations import plot_eaep_trajectory, plot_eaep_barh, plot_coaching_scatter, plot_franchise_heatmap, plot_gantt_runs

