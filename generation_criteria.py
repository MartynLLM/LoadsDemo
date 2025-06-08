"""
Generation Criteria and Validation Module

Defines criteria for acceptable simulations and validation functions.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class GenerationCriteria:
    """Configuration class for generation criteria and constraints."""
    
    # Precision requirements
    target_precision: float = 0.01  # 1% precision requirement
    required_metrics: List[str] = None  # Metrics that must meet precision
    
    # Constraints
    ensure_nonnegative: bool = True
    min_correlation: float = -1.0
    max_correlation: float = 1.0
    
    # Optimization parameters
    max_iterations: int = 200
    max_attempts: int = 5
    tolerance: float = 0.005
    
    # Focus mode
    prioritize_autocorrelation: bool = False
    
    def __post_init__(self):
        if self.required_metrics is None:
            if self.prioritize_autocorrelation:
                self.required_metrics = ['mean', 'variance', 'autocorrelation', 'correlation']
            else:
                self.required_metrics = ['mean', 'variance', 'correlation']


class ValidationResult:
    """Container for validation results."""
    
    def __init__(self, error_percentages: Dict[str, float], 
                 criteria: GenerationCriteria,
                 num_negative_values: int = 0):
        self.error_percentages = error_percentages
        self.criteria = criteria
        self.num_negative_values = num_negative_values
        self._calculate_status()
    
    def _calculate_status(self):
        """Calculate validation status based on criteria."""
        target_precision_pct = self.criteria.target_precision * 100
        
        # Check required metrics
        self.metric_status = {}
        for metric in self.criteria.required_metrics:
            if metric in self.error_percentages:
                self.metric_status[metric] = self.error_percentages[metric] <= target_precision_pct
            else:
                self.metric_status[metric] = False
        
        # Check non-negativity constraint
        self.nonnegative_ok = (not self.criteria.ensure_nonnegative or 
                              self.num_negative_values == 0)
        
        # Overall precision achieved
        self.precision_achieved = (all(self.metric_status.values()) and 
                                 self.nonnegative_ok)
    
    def get_failed_metrics(self) -> List[str]:
        """Get list of metrics that failed to meet precision requirements."""
        return [metric for metric, passed in self.metric_status.items() if not passed]
    
    def get_max_error(self) -> float:
        """Get maximum error percentage among required metrics."""
        required_errors = [self.error_percentages.get(metric, float('inf')) 
                          for metric in self.criteria.required_metrics]
        return max(required_errors) if required_errors else 0.0
    
    def print_summary(self, verbose: bool = True):
        """Print validation summary."""
        print(f"\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        
        if verbose:
            print(f"{'Property':<25} {'Error %':<10} {'Target %':<10} {'Status':<8}")
            print("-" * 63)
            
            target_precision_pct = self.criteria.target_precision * 100
            
            for metric in ['mean', 'variance', 'autocorrelation', 'correlation']:
                if metric in self.error_percentages:
                    error_pct = self.error_percentages[metric]
                    is_required = metric in self.criteria.required_metrics
                    status_symbol = '✓' if self.metric_status.get(metric, False) else '✗'
                    if not is_required:
                        status_symbol = 'ⓘ'  # Informational only
                    
                    print(f"{metric.capitalize():<25} {error_pct:<10.3f} "
                          f"{'≤' + str(target_precision_pct):<10} {status_symbol:<8}")
            
            if self.criteria.ensure_nonnegative:
                status = '✓' if self.nonnegative_ok else '✗'
                print(f"{'Non-negative constraint':<25} {'N/A':<10} {'0 neg':<10} {status:<8}")
        
        # Overall result
        if self.precision_achieved:
            print("🎉 SUCCESS: All required criteria met!")
        else:
            print("⚠️  WARNING: Some criteria not met")
            failed = self.get_failed_metrics()
            if failed:
                print(f"   Failed metrics: {', '.join(failed)}")
            if not self.nonnegative_ok:
                print(f"   Negative values: {self.num_negative_values}")


def validate_generation_result(original_data: np.ndarray,
                             generated_data: np.ndarray,
                             target_correlation: float,
                             criteria: GenerationCriteria) -> ValidationResult:
    """
    Validate generated time series against criteria.
    
    Parameters:
    original_data (np.ndarray): Original time series
    generated_data (np.ndarray): Generated time series
    target_correlation (float): Target correlation with original
    criteria (GenerationCriteria): Validation criteria
    
    Returns:
    ValidationResult: Validation results
    """
    from .statistics import (calculate_series_statistics, calculate_correlation,
                           calculate_error_percentages)
    
    # Calculate statistics for both series
    original_stats = calculate_series_statistics(original_data)
    generated_stats = calculate_series_statistics(generated_data)
    
    # Add correlation with original to generated stats
    generated_stats['correlation_with_original'] = calculate_correlation(
        original_data, generated_data)
    
    # Calculate error percentages
    error_percentages = calculate_error_percentages(
        generated_stats, original_stats, target_correlation)
    
    # Create validation result
    return ValidationResult(
        error_percentages=error_percentages,
        criteria=criteria,
        num_negative_values=generated_stats['num_negative_values']
    )


def should_continue_optimization(validation_result: ValidationResult,
                               iteration: int,
                               stage: int = 0) -> bool:
    """
    Determine if optimization should continue based on current results.
    
    Parameters:
    validation_result (ValidationResult): Current validation results
    iteration (int): Current iteration number
    stage (int): Current optimization stage
    
    Returns:
    bool: True if optimization should continue
    """
    # Stop if precision achieved
    if validation_result.precision_achieved:
        return False
    
    # Stop if maximum iterations reached
    if iteration >= validation_result.criteria.max_iterations:
        return False
    
    # Continue optimization
    return True


def create_default_criteria(correlation_focused: bool = True) -> GenerationCriteria:
    """
    Create default generation criteria.
    
    Parameters:
    correlation_focused (bool): If True, prioritize correlation over autocorrelation
    
    Returns:
    GenerationCriteria: Default criteria configuration
    """
    if correlation_focused:
        return GenerationCriteria(
            required_metrics=['mean', 'variance', 'correlation'],
            prioritize_autocorrelation=False,
            max_iterations=200,
            max_attempts=5
        )
    else:
        return GenerationCriteria(
            required_metrics=['mean', 'variance', 'autocorrelation', 'correlation'],
            prioritize_autocorrelation=True,
            max_iterations=250,
            max_attempts=5
        )


def validate_input_parameters(target_correlation: float,
                            random_seed: Optional[int] = None) -> bool:
    """
    Validate input parameters before generation.
    
    Parameters:
    target_correlation (float): Target correlation value
    random_seed (int, optional): Random seed
    
    Returns:
    bool: True if parameters are valid
    """
    # Check correlation bounds
    if not -1.0 <= target_correlation <= 1.0:
        print(f"Error: Target correlation must be between -1 and 1, got {target_correlation}")
        return False
    
    # Check random seed
    if random_seed is not None and random_seed < 0:
        print(f"Warning: Negative random seed {random_seed} may cause issues")
    
    return True


def calculate_optimization_weights(criteria: GenerationCriteria) -> Dict[str, float]:
    """
    Calculate optimization weights for different metrics based on criteria.
    
    Parameters:
    criteria (GenerationCriteria): Generation criteria
    
    Returns:
    dict: Dictionary of metric weights for optimization
    """
    weights = {
        'mean': 1.0,
        'variance': 1.0,
        'correlation': 1.0,
        'autocorrelation': 3.0 if criteria.prioritize_autocorrelation else 1.0
    }
    
    return weights
