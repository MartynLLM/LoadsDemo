"""
Time Series Generation Algorithms Module

Contains algorithms for generating correlated time series with specific properties.
"""

import numpy as np
from typing import Optional, Tuple
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class TimeSeriesGenerator:
    """Base class for time series generators."""
    
    def __init__(self, random_seed: Optional[int] = None):
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def generate(self, *args, **kwargs) -> np.ndarray:
        """Generate time series. To be implemented by subclasses."""
        raise NotImplementedError


class CorrelatedSeriesGenerator(TimeSeriesGenerator):
    """Generator for series correlated with an original series."""
    
    def generate(self, original_series: np.ndarray, target_correlation: float,
                ensure_nonnegative: bool = True) -> np.ndarray:
        """
        Create a new series with specified correlation to the original series.
        
        Parameters:
        original_series (np.ndarray): The original time series
        target_correlation (float): Desired correlation with original series (-1 to 1)
        ensure_nonnegative (bool): If True, ensures generated series has no negative values
        
        Returns:
        np.ndarray: New correlated series
        """
        n = len(original_series)
        orig_mean = np.mean(original_series)
        orig_std = np.std(original_series, ddof=1)
        orig_min = np.min(original_series)
        
        # Check if original series has negative values
        has_negative = orig_min < 0
        
        if has_negative:
            # Shift original series to be non-negative for correlation calculation
            shifted_orig = original_series - orig_min + 0.001
            orig_mean_shifted = np.mean(shifted_orig)
            orig_std_shifted = np.std(shifted_orig, ddof=1)
            
            # Standardize shifted series
            if orig_std_shifted > 0:
                standardized_orig = (shifted_orig - orig_mean_shifted) / orig_std_shifted
            else:
                standardized_orig = np.zeros(n)
        else:
            # Use original series if already non-negative
            shifted_orig = original_series
            if orig_std > 0:
                standardized_orig = (original_series - orig_mean) / orig_std
            else:
                standardized_orig = np.zeros(n)
        
        # Generate correlated series using Gaussian copula approach
        independent_series = np.random.normal(0, 1, n)
        
        rho = target_correlation
        if abs(rho) <= 1:
            corr_factor = np.sqrt(1 - rho**2) if abs(rho) < 1 else 0
            standardized_new = rho * standardized_orig + corr_factor * independent_series
        else:
            print("Warning: Correlation must be between -1 and 1. Using sign of input.")
            rho = np.sign(rho)
            standardized_new = rho * standardized_orig
        
        # Transform to ensure non-negative values if required
        if ensure_nonnegative:
            new_series = self._transform_to_nonnegative(
                standardized_new, orig_mean, orig_std**2, has_negative, orig_min)
        else:
            # Simple linear transformation
            new_series = standardized_new * orig_std + orig_mean
        
        return new_series
    
    def _transform_to_nonnegative(self, standardized_series: np.ndarray,
                                 target_mean: float, target_variance: float,
                                 has_negative: bool, orig_min: float) -> np.ndarray:
        """Transform standardized series to ensure non-negative values."""
        # Convert standardized series to uniform [0,1] using normal CDF
        if SCIPY_AVAILABLE:
            uniform_values = stats.norm.cdf(standardized_series)
        else:
            # Fallback approximation for normal CDF
            uniform_values = 0.5 * (1 + np.tanh(standardized_series / np.sqrt(2)))
        
        # Transform to appropriate target mean
        if not has_negative:
            adjusted_mean = target_mean
        else:
            adjusted_mean = target_mean - orig_min + 0.001
        
        if adjusted_mean > 0 and target_variance > 0:
            # Use gamma distribution parameters
            gamma_shape = adjusted_mean**2 / target_variance
            gamma_scale = target_variance / adjusted_mean
            
            # Convert uniform to gamma using inverse CDF
            if SCIPY_AVAILABLE:
                new_series = stats.gamma.ppf(uniform_values, a=gamma_shape, scale=gamma_scale)
            else:
                # Fallback: exponential transformation
                new_series = -adjusted_mean * np.log(1 - uniform_values + 1e-10)
        else:
            # Simple exponential transformation
            new_series = np.exp(standardized_series * 0.5 + np.log(max(adjusted_mean, 1.0)))
        
        # Ensure no negative values
        new_series = np.maximum(new_series, 0)
        
        # If original had negative values, shift back
        if has_negative:
            new_series = new_series + orig_min - 0.001
            new_series = np.maximum(new_series, 0)
        
        return new_series


class AR1Generator(TimeSeriesGenerator):
    """Generator for AR(1) time series."""
    
    def generate(self, n: int, mean: float, variance: float, autocorr: float,
                ensure_nonnegative: bool = True, reference_series: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate an AR(1) time series with specified properties.
        
        Parameters:
        n (int): Length of the series
        mean (float): Target mean
        variance (float): Target variance
        autocorr (float): Target first-order autocorrelation
        ensure_nonnegative (bool): If True, ensures all values are >= 0
        reference_series (np.ndarray, optional): Reference series for initialization
        
        Returns:
        np.ndarray: Generated AR(1) time series
        """
        # Use reference series statistics for better initialization if available
        if reference_series is not None:
            init_value = reference_series[0]
            innovation_scale = np.std(np.diff(reference_series)) * 0.5
        else:
            init_value = mean
            innovation_scale = np.sqrt(variance) * 0.3
        
        # Generate AR(1) series
        phi = autocorr
        series = np.zeros(n)
        series[0] = init_value
        
        for t in range(1, n):
            innovation = np.random.normal(0, innovation_scale)
            series[t] = mean + phi * (series[t-1] - mean) + innovation
        
        if ensure_nonnegative:
            series = np.maximum(series, 0)
        
        # Adjust to exact mean and variance
        current_mean = np.mean(series)
        current_var = np.var(series, ddof=1)
        
        if current_var > 0:
            series = (series - current_mean) / np.sqrt(current_var) * np.sqrt(variance) + mean
        
        if ensure_nonnegative:
            series = np.maximum(series, 0)
        
        return series


class LogNormalAR1Generator(TimeSeriesGenerator):
    """Generator for log-normal AR(1) time series (always non-negative)."""
    
    def generate(self, n: int, mean: float, variance: float, autocorr: float) -> np.ndarray:
        """
        Generate a log-normal AR(1) time series.
        
        Parameters:
        n (int): Length of the series
        mean (float): Target mean
        variance (float): Target variance
        autocorr (float): Target first-order autocorrelation
        
        Returns:
        np.ndarray: Generated log-normal AR(1) time series
        """
        if mean <= 0:
            raise ValueError("Mean must be positive for log-normal distribution")
        
        # Convert to log-space parameters
        log_mean = np.log(mean)
        log_var = np.log(1 + variance / (mean**2))
        
        # AR(1) model in log space
        phi = autocorr
        
        if abs(phi) < 1:
            innovation_variance = log_var * (1 - phi**2)
        else:
            innovation_variance = log_var * 0.5
        
        innovation_std = np.sqrt(innovation_variance)
        
        # Generate log-space series
        log_series = np.zeros(n)
        log_series[0] = np.random.normal(log_mean, np.sqrt(log_var))
        
        for t in range(1, n):
            innovation = np.random.normal(0, innovation_std)
            log_series[t] = phi * (log_series[t-1] - log_mean) + log_mean + innovation
        
        # Transform back to original space
        series = np.exp(log_series)
        
        # Adjust to match target mean and variance
        current_mean = np.mean(series)
        current_var = np.var(series, ddof=1)
        
        if current_var > 0 and current_mean > 0:
            # Use moment matching for log-normal distribution
            series = series * (mean / current_mean)
            
            # Fine-tune variance while maintaining non-negativity
            current_var_adjusted = np.var(series, ddof=1)
            if current_var_adjusted > 0:
                variance_factor = np.sqrt(variance / current_var_adjusted)
                series = (series - mean) * variance_factor + mean
                series = np.maximum(series, 0)
        
        return series


class SeriesAdjuster:
    """Utility class for adjusting series properties while preserving structure."""
    
    @staticmethod
    def adjust_mean_variance_preserve_autocorr(series: np.ndarray, 
                                             target_mean: float, 
                                             target_variance: float,
                                             ensure_nonnegative: bool = True) -> np.ndarray:
        """
        Adjust mean and variance while minimally impacting autocorrelation.
        
        Parameters:
        series (np.ndarray): Input time series
        target_mean (float): Target mean
        target_variance (float): Target variance
        ensure_nonnegative (bool): If True, ensures all values are >= 0
        
        Returns:
        np.ndarray: Adjusted series
        """
        series = series.copy()
        current_mean = np.mean(series)
        current_var = np.var(series, ddof=1)
        
        # Gentle adjustment to preserve autocorrelation structure
        if abs(current_mean - target_mean) > 0.01 * abs(target_mean):
            # Simple shift for mean (preserves autocorrelation exactly)
            series = series + (target_mean - current_mean)
        
        # Gentle scaling for variance
        current_var = np.var(series, ddof=1)
        if current_var > 0 and abs(current_var - target_variance) > 0.01 * abs(target_variance):
            scale_factor = np.sqrt(target_variance / current_var)
            series_mean = np.mean(series)
            # Scale around mean (minimally affects autocorrelation)
            series = (series - series_mean) * scale_factor + series_mean
        
        if ensure_nonnegative:
            series = np.maximum(series, 0)
            # If clipping changed mean significantly, readjust
            new_mean = np.mean(series)
            if abs(new_mean - target_mean) > 0.05 * abs(target_mean):
                series = series + (target_mean - new_mean)
                series = np.maximum(series, 0)
        
        return series
    
    @staticmethod
    def blend_series(series1: np.ndarray, series2: np.ndarray, 
                    alpha: float, ensure_nonnegative: bool = True) -> np.ndarray:
        """
        Blend two time series with given weight.
        
        Parameters:
        series1 (np.ndarray): First series
        series2 (np.ndarray): Second series
        alpha (float): Blending weight (0 = all series1, 1 = all series2)
        ensure_nonnegative (bool): If True, ensures result is non-negative
        
        Returns:
        np.ndarray: Blended series
        """
        if len(series1) != len(series2):
            raise ValueError("Series must have the same length")
        
        blended = (1 - alpha) * series1 + alpha * series2
        
        if ensure_nonnegative:
            blended = np.maximum(blended, 0)
        
        return blended
    
    @staticmethod
    def fine_tune_correlation(base_series: np.ndarray, 
                            target_series: np.ndarray,
                            target_correlation: float,
                            adjustment_strength: float = 0.1,
                            ensure_nonnegative: bool = True) -> np.ndarray:
        """
        Fine-tune a series to better match target correlation.
        
        Parameters:
        base_series (np.ndarray): Series to adjust
        target_series (np.ndarray): Target series for correlation
        target_correlation (float): Desired correlation
        adjustment_strength (float): Strength of adjustment (0-1)
        ensure_nonnegative (bool): If True, ensures result is non-negative
        
        Returns:
        np.ndarray: Adjusted series
        """
        current_correlation = np.corrcoef(base_series, target_series)[0, 1]
        correlation_error = target_correlation - current_correlation
        
        if abs(correlation_error) < 0.001:  # Already close enough
            return base_series
        
        # Calculate adjustment direction
        target_std = np.std(target_series, ddof=1)
        base_std = np.std(base_series, ddof=1)
        
        if base_std > 0 and target_std > 0:
            # Normalize both series
            norm_base = (base_series - np.mean(base_series)) / base_std
            norm_target = (target_series - np.mean(target_series)) / target_std
            
            # Adjust towards better correlation
            adjustment = correlation_error * adjustment_strength
            norm_adjusted = norm_base + adjustment * norm_target
            
            # Transform back
            adjusted_series = norm_adjusted * base_std + np.mean(base_series)
            
            if ensure_nonnegative:
                adjusted_series = np.maximum(adjusted_series, 0)
            
            return adjusted_series
        
        return base_series


def create_generator(generator_type: str = 'correlated', 
                    random_seed: Optional[int] = None) -> TimeSeriesGenerator:
    """
    Factory function to create time series generators.
    
    Parameters:
    generator_type (str): Type of generator ('correlated', 'ar1', 'lognormal_ar1')
    random_seed (int, optional): Random seed
    
    Returns:
    TimeSeriesGenerator: Appropriate generator instance
    """
    generators = {
        'correlated': CorrelatedSeriesGenerator,
        'ar1': AR1Generator,
        'lognormal_ar1': LogNormalAR1Generator
    }
    
    if generator_type not in generators:
        raise ValueError(f"Unknown generator type: {generator_type}. "
                        f"Available types: {list(generators.keys())}")
    
    return generators[generator_type](random_seed=random_seed)