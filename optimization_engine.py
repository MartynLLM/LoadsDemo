"""
Optimization Engine Module

Contains the core optimization logic for generating time series with specific properties.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from .series_generators import (CorrelatedSeriesGenerator, AR1Generator, 
                               SeriesAdjuster, create_generator)
from .generation_criteria import GenerationCriteria, ValidationResult, validate_generation_result
from .statistics import calculate_series_statistics, calculate_correlation


class OptimizationEngine:
    """Core optimization engine for time series generation."""
    
    def __init__(self, criteria: GenerationCriteria):
        self.criteria = criteria
        self.adjuster = SeriesAdjuster()
        
    def generate_correlated_series(self, original_data: np.ndarray, 
                                 target_correlation: float,
                                 random_seed: Optional[int] = None) -> Dict:
        """
        Generate a correlated time series with optimization.
        
        Parameters:
        original_data (np.ndarray): Original time series data
        target_correlation (float): Target correlation with original series
        random_seed (int, optional): Random seed for reproducibility
        
        Returns:
        dict: Results containing generated series and statistics
        """
        print("="*60)
        print("CORRELATION-FOCUSED TIME SERIES GENERATOR")
        if self.criteria.ensure_nonnegative:
            print("(NON-NEGATIVE VALUES ONLY)")
        print("="*60)
        
        # Calculate target statistics
        original_stats = calculate_series_statistics(original_data)
        n = len(original_data)
        target_mean = original_stats['mean']
        target_variance = original_stats['variance']
        target_autocorr = original_stats['first_order_autocorrelation']
        
        self._print_original_properties(original_stats, target_correlation)
        
        # Generate base correlated series
        base_generator = create_generator('correlated', random_seed)
        base_series = base_generator.generate(
            original_data, target_correlation, self.criteria.ensure_nonnegative)
        
        # Optimize the series
        best_series = self._optimize_series(
            base_series, original_data, target_mean, target_variance, 
            target_autocorr, target_correlation, random_seed)
        
        # Validate and return results
        validation_result = validate_generation_result(
            original_data, best_series, target_correlation, self.criteria)
        
        return self._prepare_results(
            original_data, best_series, original_stats, 
            target_correlation, validation_result)
    
    def _print_original_properties(self, original_stats: Dict, target_correlation: float):
        """Print original series properties."""
        print(f"\nOriginal Series Properties:")
        print(f"Length: {original_stats['sample_size']}")
        print(f"Mean: {original_stats['mean']:.6f}")
        print(f"Variance: {original_stats['variance']:.6f}")
        print(f"Autocorrelation: {original_stats['first_order_autocorrelation']:.6f}")
        print(f"Min value: {original_stats['min_value']:.6f}")
        print(f"Max value: {original_stats['max_value']:.6f}")
        print(f"Target correlation with original: {target_correlation:.6f}")
        if self.criteria.ensure_nonnegative:
            print(f"Constraint: Generated series must be >= 0")
    
    def _optimize_series(self, base_series: np.ndarray, original_data: np.ndarray,
                        target_mean: float, target_variance: float, 
                        target_autocorr: float, target_correlation: float,
                        random_seed: Optional[int]) -> np.ndarray:
        """
        Optimize the base series to meet criteria.
        
        Parameters:
        base_series (np.ndarray): Base correlated series
        original_data (np.ndarray): Original time series
        target_mean (float): Target mean
        target_variance (float): Target variance
        target_autocorr (float): Target autocorrelation
        target_correlation (float): Target correlation
        random_seed (int, optional): Random seed
        
        Returns:
        np.ndarray: Optimized series
        """
        n = len(original_data)
        best_series = base_series.copy()
        best_overall_error = float('inf')
        
        print(f"\nOptimizing series to match required properties within {self.criteria.target_precision*100}%...")
        
        # Multi-stage optimization with different alpha ranges
        alpha_stages = self._get_alpha_stages()
        
        for stage in range(len(alpha_stages)):
            stage_max_iter = self.criteria.max_iterations // len(alpha_stages)
            alpha_range = alpha_stages[stage]
            
            print(f"  Stage {stage + 1}/{len(alpha_stages)}: Testing blending (alpha: {alpha_range})...")
            
            best_series, best_overall_error = self._optimize_stage(
                best_series, base_series, original_data, 
                target_mean, target_variance, target_autocorr, target_correlation,
                alpha_range, stage_max_iter, stage, random_seed)
            
            # Break if precision achieved
            if best_overall_error < self.criteria.target_precision:
                print(f"    ✓ Achieved {self.criteria.target_precision*100}% precision after stage {stage+1}")
                break
        
        # Final refinement if needed
        if best_overall_error >= self.criteria.target_precision:
            best_series = self._final_refinement(
                best_series, original_data, target_mean, target_variance, target_correlation)
        
        return best_series
    
    def _get_alpha_stages(self) -> List[List[float]]:
        """Get alpha ranges for different optimization stages."""
        if self.criteria.prioritize_autocorrelation:
            return [
                [0.3, 0.5, 0.7],  # Higher AR(1) influence
                [0.2, 0.4, 0.6],  # Moderate-high influence  
                [0.1, 0.3, 0.5]   # Balanced approach
            ]
        else:
            return [
                [0.02, 0.1, 0.2],   # Lower alpha for correlation preservation
                [0.01, 0.05, 0.15], # Even lower alpha
                [0.005, 0.02, 0.1]  # Minimal alpha
            ]
    
    def _optimize_stage(self, best_series: np.ndarray, base_series: np.ndarray,
                       original_data: np.ndarray, target_mean: float, 
                       target_variance: float, target_autocorr: float,
                       target_correlation: float, alpha_range: List[float],
                       stage_max_iter: int, stage: int, 
                       random_seed: Optional[int]) -> Tuple[np.ndarray, float]:
        """Optimize for a single stage."""
        best_overall_error = float('inf')
        current_best = best_series.copy()
        
        for alpha in alpha_range:
            for iteration in range(stage_max_iter):
                # Generate AR(1) component
                ar1_generator = create_generator('ar1', 
                    random_seed + iteration + stage * 1000 if random_seed else None)
                ar1_series = ar1_generator.generate(
                    len(original_data), target_mean, target_variance, target_autocorr,
                    self.criteria.ensure_nonnegative, original_data)
                
                # Blend series
                candidate_series = self.adjuster.blend_series(
                    base_series, ar1_series, alpha, self.criteria.ensure_nonnegative)
                
                # Adjust to match target properties
                candidate_series = self._precise_adjustment(
                    candidate_series, target_mean, target_variance)
                
                # Calculate error for required metrics only
                overall_error = self._calculate_overall_error(
                    candidate_series, original_data, target_mean, 
                    target_variance, target_autocorr, target_correlation)
                
                if overall_error < best_overall_error:
                    current_best = candidate_series.copy()
                    best_overall_error = overall_error
                    
                    if best_overall_error < self.criteria.target_precision:
                        break
            
            if best_overall_error < self.criteria.target_precision:
                break
        
        return current_best, best_overall_error
    
    def _precise_adjustment(self, series: np.ndarray, target_mean: float, 
                          target_variance: float) -> np.ndarray:
        """Apply precise adjustments to mean and variance."""
        # Precise adjustment to match mean exactly
        current_mean = np.mean(series)
        if abs(current_mean - target_mean) > self.criteria.target_precision * abs(target_mean):
            adjustment = target_mean - current_mean
            series = series + adjustment
            if self.criteria.ensure_nonnegative:
                series = np.maximum(series, 0)
        
        # Precise adjustment to match variance
        current_var = np.var(series, ddof=1)
        if current_var > 0 and abs(current_var - target_variance) > self.criteria.target_precision * abs(target_variance):
            var_factor = np.sqrt(target_variance / current_var)
            series_mean = np.mean(series)
            series = (series - series_mean) * var_factor + series_mean
            if self.criteria.ensure_nonnegative:
                series = np.maximum(series, 0)
        
        return series
    
    def _calculate_overall_error(self, candidate_series: np.ndarray, 
                               original_data: np.ndarray, target_mean: float,
                               target_variance: float, target_autocorr: float,
                               target_correlation: float) -> float:
        """Calculate overall error for optimization."""
        if len(candidate_series) <= 1:
            return float('inf')
        
        # Calculate current properties
        cand_mean = np.mean(candidate_series)
        cand_var = np.var(candidate_series, ddof=1)
        cand_correlation = calculate_correlation(original_data, candidate_series)
        
        # Calculate normalized errors for required metrics
        mean_error = abs(cand_mean - target_mean) / abs(target_mean) if target_mean != 0 else abs(cand_mean - target_mean)
        var_error = abs(cand_var - target_variance) / abs(target_variance) if target_variance != 0 else abs(cand_var - target_variance)
        corr_error = abs(cand_correlation - target_correlation) / abs(target_correlation) if target_correlation != 0 else abs(cand_correlation - target_correlation)
        
        # Include autocorrelation only if prioritized
        if self.criteria.prioritize_autocorrelation:
            from .statistics import calculate_autocorrelation
            cand_autocorr = calculate_autocorrelation(candidate_series)
            autocorr_error = abs(cand_autocorr - target_autocorr) / abs(target_autocorr) if target_autocorr != 0 else abs(cand_autocorr - target_autocorr)
            return max(mean_error, var_error, corr_error, autocorr_error * 3.0)  # Weight autocorr higher
        else:
            return max(mean_error, var_error, corr_error)
    
    def _final_refinement(self, series: np.ndarray, original_data: np.ndarray,
                         target_mean: float, target_variance: float,
                         target_correlation: float) -> np.ndarray:
        """Apply final refinement for required metrics."""
        print(f"  Final refinement stage for required metrics...")
        
        for fine_iter in range(50):
            current_mean = np.mean(series)
            current_var = np.var(series, ddof=1)
            current_correlation = calculate_correlation(original_data, series)
            
            # Calculate adjustment factors
            mean_factor = target_mean / current_mean if current_mean != 0 else 1
            var_factor = np.sqrt(target_variance / current_var) if current_var > 0 else 1
            
            # Apply small incremental adjustments
            adjustment_weight = 0.05  # Small adjustment to preserve correlation
            
            # Adjust mean (preserves correlation exactly)
            if abs(current_mean - target_mean) > self.criteria.target_precision * abs(target_mean):
                series = series + (target_mean - current_mean)
            
            # Adjust variance gently to preserve correlation
            if abs(current_var - target_variance) > self.criteria.target_precision * abs(target_variance):
                series_mean = np.mean(series)
                series = (series - series_mean) * (1 + adjustment_weight * (var_factor - 1)) + series_mean
            
            # Ensure non-negative constraint
            if self.criteria.ensure_nonnegative:
                series = np.maximum(series, 0)
            
            # Check if precision achieved on required metrics
            validation_result = validate_generation_result(
                original_data, series, target_correlation, self.criteria)
            
            if validation_result.precision_achieved:
                print(f"    ✓ Final precision achieved after {fine_iter + 1} refinement iterations")
                break
        
        return series
    
    def _prepare_results(self, original_data: np.ndarray, generated_series: np.ndarray,
                        original_stats: Dict, target_correlation: float,
                        validation_result: ValidationResult) -> Dict:
        """Prepare final results dictionary."""
        generated_stats = calculate_series_statistics(generated_series)
        generated_stats['correlation_with_original'] = calculate_correlation(
            original_data, generated_series)
        
        results = {
            'original_data': original_data,
            'generated_series': generated_series,
            'original_stats': original_stats,
            'generated_stats': generated_stats,
            'target_correlation': target_correlation,
            'ensure_nonnegative': self.criteria.ensure_nonnegative,
            'precision_achieved': validation_result.precision_achieved,
            'error_percentages': validation_result.error_percentages,
            'validation_result': validation_result
        }
        
        # Print validation summary
        validation_result.print_summary(verbose=True)
        
        return results


class MultiAttemptOptimizer:
    """Optimizer that tries multiple attempts to achieve precision."""
    
    def __init__(self, criteria: GenerationCriteria):
        self.criteria = criteria
        
    def optimize_with_retries(self, original_data: np.ndarray,
                            target_correlation: float,
                            random_seed: Optional[int] = None) -> Optional[Dict]:
        """
        Attempt optimization multiple times until precision is achieved.
        
        Parameters:
        original_data (np.ndarray): Original time series data
        target_correlation (float): Target correlation
        random_seed (int, optional): Base random seed
        
        Returns:
        dict: Results if successful, None otherwise
        """
        for attempt in range(self.criteria.max_attempts):
            if attempt > 0:
                print(f"\n{'='*50}")
                print(f"ATTEMPT {attempt + 1}/{self.criteria.max_attempts}")
                print(f"{'='*50}")
            
            # Create engine for this attempt
            engine = OptimizationEngine(self.criteria)
            
            # Generate with different seed for each attempt
            attempt_seed = random_seed + attempt * 100 if random_seed else None
            results = engine.generate_correlated_series(
                original_data, target_correlation, attempt_seed)
            
            if results['precision_achieved']:
                return results
            elif attempt < self.criteria.max_attempts - 1:
                print(f"\n⚠️  Attempt {attempt + 1} did not achieve required precision. Retrying...")
        
        # Return best attempt even if precision not achieved
        return results if 'results' in locals() else None
