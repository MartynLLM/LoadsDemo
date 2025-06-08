"""
Statistical Calculations Module

Contains functions for calculating statistical properties of time series.
"""

import numpy as np
from typing import Dict, Optional
from .data_io import read_flow_data # type: ignore



def calculate_series_statistics(data: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive statistics for a time series.
    
    Parameters:
    data (np.ndarray): Time series data
    
    Returns:
    dict: Dictionary containing calculated statistics
    """
    if len(data) == 0:
        return {}
    
    stats = {
        'mean': np.mean(data),
        'variance': np.var(data, ddof=1),
        'standard_deviation': np.std(data, ddof=1),
        'min_value': np.min(data),
        'max_value': np.max(data),
        'sample_size': len(data),
        'num_negative_values': np.sum(data < 0)
    }
    
    # Calculate autocorrelation
    stats['first_order_autocorrelation'] = calculate_autocorrelation(data, lag=1)
    
    return stats


def calculate_autocorrelation(data: np.ndarray, lag: int = 1) -> float:
    """
    Calculate autocorrelation for given lag.
    
    Parameters:
    data (np.ndarray): Time series data
    lag (int): Lag for autocorrelation calculation
    
    Returns:
    float: Autocorrelation coefficient
    """
    data = np.array(data)
    n = len(data)
    
    if n <= lag:
        return float('nan')
    
    if n == 1:
        return float('nan')
    
    # Get original series and lagged series
    y = data[lag:]      # y(t+lag): from index lag to end
    y_lag = data[:-lag] # y(t): from index 0 to n-lag-1
    
    # Calculate means
    y_mean = np.mean(y)
    y_lag_mean = np.mean(y_lag)
    
    # Calculate covariance and variances
    covariance = np.mean((y - y_mean) * (y_lag - y_lag_mean))
    var_y = np.var(y, ddof=0)       # Population variance for correlation
    var_y_lag = np.var(y_lag, ddof=0)
    
    # Calculate correlation coefficient
    if var_y > 0 and var_y_lag > 0:
        return covariance / np.sqrt(var_y * var_y_lag)
    else:
        return 0.0


def calculate_correlation(series1: np.ndarray, series2: np.ndarray) -> float:
    """
    Calculate Pearson correlation coefficient between two series.
    
    Parameters:
    series1 (np.ndarray): First time series
    series2 (np.ndarray): Second time series
    
    Returns:
    float: Correlation coefficient
    """
    if len(series1) != len(series2):
        raise ValueError("Series must have the same length")
    
    if len(series1) < 2:
        return float('nan')
    
    return np.corrcoef(series1, series2)[0, 1]


def calculate_error_percentages(achieved_stats: Dict[str, float], 
                              target_stats: Dict[str, float],
                              target_correlation: float) -> Dict[str, float]:
    """
    Calculate percentage errors between achieved and target statistics.
    
    Parameters:
    achieved_stats (dict): Statistics from generated series
    target_stats (dict): Target statistics from original series
    target_correlation (float): Target correlation with original series
    
    Returns:
    dict: Dictionary containing error percentages
    """
    errors = {}
    
    # Mean error
    target_mean = target_stats['mean']
    achieved_mean = achieved_stats['mean']
    if target_mean != 0:
        errors['mean'] = abs(achieved_mean - target_mean) / abs(target_mean) * 100
    else:
        errors['mean'] = abs(achieved_mean - target_mean) * 100
    
    # Variance error
    target_var = target_stats['variance']
    achieved_var = achieved_stats['variance']
    if target_var != 0:
        errors['variance'] = abs(achieved_var - target_var) / abs(target_var) * 100
    else:
        errors['variance'] = abs(achieved_var - target_var) * 100
    
    # Autocorrelation error
    target_autocorr = target_stats['first_order_autocorrelation']
    achieved_autocorr = achieved_stats['first_order_autocorrelation']
    if target_autocorr != 0:
        errors['autocorrelation'] = abs(achieved_autocorr - target_autocorr) / abs(target_autocorr) * 100
    else:
        errors['autocorrelation'] = abs(achieved_autocorr - target_autocorr) * 100
    
    # Correlation error
    achieved_corr = achieved_stats.get('correlation_with_original', 0)
    if target_correlation != 0:
        errors['correlation'] = abs(achieved_corr - target_correlation) / abs(target_correlation) * 100
    else:
        errors['correlation'] = abs(achieved_corr - target_correlation) * 100
    
    return errors


def check_precision_requirements(error_percentages: Dict[str, float],
                               target_precision: float = 0.01,
                               required_metrics: Optional[list] = None) -> bool:
    """
    Check if error percentages meet precision requirements.
    
    Parameters:
    error_percentages (dict): Dictionary of error percentages
    target_precision (float): Target precision as decimal (0.01 = 1%)
    required_metrics (list): List of metrics that must meet precision (default: mean, variance, correlation)
    
    Returns:
    bool: True if all required metrics meet precision requirements
    """
    if required_metrics is None:
        required_metrics = ['mean', 'variance', 'correlation']
    
    target_precision_pct = target_precision * 100
    
    for metric in required_metrics:
        if metric in error_percentages:
            if error_percentages[metric] > target_precision_pct:
                return False
    
    return True


def analyze_flow_data(csv_file_path: str) -> Optional[Dict]:
    """
    Analyze flow data from CSV file and calculate statistics.
    
    Parameters:
    csv_file_path (str): Path to the CSV file
    
    Returns:
    dict: Dictionary containing data and statistics, or None if error
    """
    from .data_io import read_flow_data
    
    # Read the data
    data_dict = read_flow_data(csv_file_path)
    if data_dict is None:
        return None
    
    flow_data = data_dict['flow_data']
    
    if len(flow_data) == 0:
        print("Error: No valid flow data found")
        return None
    
    # Calculate statistics
    stats = calculate_series_statistics(flow_data)
    
    # Combine data and statistics
    result = {
        'data': flow_data,
        'dates': data_dict['dates'],
        'has_dates': data_dict['has_dates'],
        **stats
    }
    
    return result
