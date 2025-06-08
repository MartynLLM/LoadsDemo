"""
Main User Interface Module

Provides high-level interface functions for time series generation and analysis.
"""

import numpy as np
from typing import Dict, List, Optional, Union
from .data_io import read_flow_data, save_generated_series, create_output_filename
from .statistics import analyze_flow_data
from .generation_criteria import (GenerationCriteria, create_default_criteria, 
                                validate_input_parameters)
from .optimization_engine import MultiAttemptOptimizer
from .visualization import create_visualizer, plot_results


def generate_correlated_time_series(csv_file_path: str, 
                                   target_correlation: float,
                                   random_seed: Optional[int] = 42,
                                   require_1pct_precision: bool = True,
                                   correlation_focused: bool = True,
                                   ensure_nonnegative: bool = True,
                                   max_attempts: int = 5) -> Optional[Dict]:
    """
    Main function for generating correlated time series.
    
    Parameters:
    csv_file_path (str): Path to input CSV file with Date and Flow columns
    target_correlation (float): Desired correlation with original series (-1 to 1)
    random_seed (int, optional): Random seed for reproducible results
    require_1pct_precision (bool): If True, retry until 1% precision achieved on required metrics
    correlation_focused (bool): If True, prioritize correlation over autocorrelation
    ensure_nonnegative (bool): If True, ensure generated series has no negative values
    max_attempts (int): Maximum attempts to achieve precision
    
    Returns:
    dict: Results containing original and generated series with statistics, or None if failed
    """
    print("="*70)
    print("CORRELATION-FOCUSED TIME SERIES GENERATOR")
    print("="*70)
    print(f"Input file: {csv_file_path}")
    print(f"Target correlation: {target_correlation}")
    print(f"Random seed: {random_seed}")
    print(f"Precision requirement: ≤ 1% error on required metrics")
    print(f"Correlation focused: {correlation_focused}")
    print(f"Non-negative constraint: {ensure_nonnegative}")
    print()
    
    # Validate input parameters
    if not validate_input_parameters(target_correlation, random_seed):
        return None
    
    # Analyze original data
    original_stats = analyze_flow_data(csv_file_path)
    if original_stats is None:
        print("Failed to analyze original data")
        return None
    
    original_data = original_stats['data']
    
    # Create generation criteria
    criteria = create_default_criteria(correlation_focused=correlation_focused)
    criteria.ensure_nonnegative = ensure_nonnegative
    criteria.max_attempts = max_attempts if require_1pct_precision else 1
    
    # Create optimizer and generate series
    optimizer = MultiAttemptOptimizer(criteria)
    results = optimizer.optimize_with_retries(
        original_data, target_correlation, random_seed)
    
    if results:
        # Add original stats to results
        results['original_stats'] = original_stats
        
        # Save results to file
        output_filename = create_output_filename(
            target_correlation, random_seed, 
            results.get('precision_achieved', False), "corr_focused")
        
        save_generated_series(
            results['original_data'], 
            results['generated_series'], 
            output_filename,
            original_stats.get('dates'))
        
        # Print final summary
        _print_final_summary(results, output_filename)
        
        return results
    else:
        print("Failed to generate correlated series")
        return None


def batch_generate_correlations(csv_file_path: str,
                               correlation_targets: List[float],
                               random_seed: int = 42,
                               require_1pct_precision: bool = True) -> List[Dict]:
    """
    Generate multiple time series with different correlation targets.
    
    Parameters:
    csv_file_path (str): Path to input CSV file
    correlation_targets (list): List of target correlations to generate
    random_seed (int): Base random seed
    require_1pct_precision (bool): If True, require 1% precision for each series
    
    Returns:
    list: List of results dictionaries
    """
    print("="*70)
    print("BATCH CORRELATION GENERATION")
    print("="*70)
    print(f"Input file: {csv_file_path}")
    print(f"Target correlations: {correlation_targets}")
    print(f"Base random seed: {random_seed}")
    print(f"Require 1% precision: {require_1pct_precision}")
    print()
    
    results_list = []
    
    for i, target_corr in enumerate(correlation_targets):
        print(f"\n{'=' * 30} BATCH {i+1}/{len(correlation_targets)}: Correlation = {target_corr} {'=' * 30}")
        
        # Use different seed for each target
        seed = random_seed + i
        
        results = generate_correlated_time_series(
            csv_file_path=csv_file_path,
            target_correlation=target_corr,
            random_seed=seed,
            require_1pct_precision=require_1pct_precision
        )
        
        if results:
            results_list.append(results)
            achieved_corr = results['generated_stats']['correlation_with_original']
            precision_achieved = results.get('precision_achieved', False)
            
            print(f"\nBatch {i+1} Result:")
            print(f"  Target correlation: {target_corr:.3f}")
            print(f"  Achieved correlation: {achieved_corr:.6f}")
            print(f"  Precision achieved: {'✓ YES' if precision_achieved else '✗ NO'}")
        else:
            print(f"Batch {i+1} Failed")
    
    # Create error analysis plot if we have results
    if results_list:
        print(f"\nGenerating batch analysis plots...")
        visualizer = create_visualizer()
        try:
            visualizer.plot_error_analysis(results_list, save_plots=True)
        except Exception as e:
            print(f"Error creating analysis plots: {e}")
    
    print(f"\nBatch generation completed! Generated {len(results_list)}/{len(correlation_targets)} series.")
    return results_list


def analyze_existing_series(csv_file_path: str, 
                          print_summary: bool = True) -> Optional[Dict]:
    """
    Analyze an existing time series from CSV file.
    
    Parameters:
    csv_file_path (str): Path to CSV file with Date and Flow columns
    print_summary (bool): If True, print analysis summary
    
    Returns:
    dict: Analysis results or None if failed
    """
    results = analyze_flow_data(csv_file_path)
    
    if results and print_summary:
        print("="*50)
        print("TIME SERIES ANALYSIS")
        print("="*50)
        print(f"File: {csv_file_path}")
        print(f"Sample size: {results['sample_size']}")
        print(f"Date range: {'Available' if results['has_dates'] else 'Not available'}")
        print(f"Mean: {results['mean']:.6f}")
        print(f"Variance: {results['variance']:.6f}")
        print(f"Standard deviation: {results['standard_deviation']:.6f}")
        print(f"First-order autocorrelation: {results['first_order_autocorrelation']:.6f}")
        print(f"Min value: {results['min_value']:.6f}")
        print(f"Max value: {results['max_value']:.6f}")
        print(f"Negative values: {results['num_negative_values']}")
        print("="*50)
    
    return results


def create_visualization(results: Dict, 
                        plot_type: str = 'comprehensive',
                        save_plots: bool = True,
                        show_plots: bool = True) -> None:
    """
    Create visualizations for generation results.
    
    Parameters:
    results (dict): Results from time series generation
    plot_type (str): Type of plot ('comprehensive' or 'quick')
    save_plots (bool): If True, save plots to files
    show_plots (bool): If True, display plots
    """
    if results is None:
        print("No results to visualize")
        return
    
    print(f"Creating {plot_type} visualization...")
    
    try:
        plot_results(results, plot_type, save_plots, show_plots)
    except Exception as e:
        print(f"Error creating visualization: {e}")
        if plot_type == 'comprehensive':
            print("Trying quick visualization as fallback...")
            try:
                plot_results(results, 'quick', save_plots, show_plots)
            except Exception as e2:
                print(f"Error creating fallback visualization: {e2}")


def _print_final_summary(results: Dict, output_filename: str) -> None:
    """Print final summary of generation results."""
    print(f"\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    if results.get('precision_achieved', False):
        print("🎉 SUCCESS: Required precision achieved!")
    else:
        print("⚠️  Generated series with best possible precision")
    
    stats = results['generated_stats']
    print(f"✓ Generated series length: {len(results['generated_series'])}")
    print(f"✓ Achieved correlation: {stats['correlation_with_original']:.6f}")
    print(f"✓ Min value: {stats['min_value']:.6f}")
    print(f"✓ Max value: {stats['max_value']:.6f}")
    print(f"✓ Negative values: {stats['num_negative_values']}")
    print(f"✓ Output saved to: {output_filename}")
    
    # Show precision summary
    error_pcts = results.get('error_percentages', {})
    print(f"\nPrecision Summary:")
    print(f"  Mean error:           {error_pcts.get('mean', 0):.3f}% {'✓' if error_pcts.get('mean', 0) <= 1.0 else '✗'}")
    print(f"  Variance error:       {error_pcts.get('variance', 0):.3f}% {'✓' if error_pcts.get('variance', 0) <= 1.0 else '✗'}")
    print(f"  Correlation error:    {error_pcts.get('correlation', 0):.3f}% {'✓' if error_pcts.get('correlation', 0) <= 1.0 else '✗'}")
    
    # Check if autocorrelation is informational or required
    criteria = results.get('validation_result')
    if criteria and 'autocorrelation' in criteria.criteria.required_metrics:
        print(f"  Autocorrelation error: {error_pcts.get('autocorrelation', 0):.3f}% {'✓' if error_pcts.get('autocorrelation', 0) <= 1.0 else '✗'}")
    else:
        print(f"  Autocorrelation info: {error_pcts.get('autocorrelation', 0):.3f}% ⓘ (informational)")


def print_usage_examples() -> None:
    """Print usage examples for the module."""
    print("="*70)
    print("USAGE EXAMPLES")
    print("="*70)
    print("# Generate series prioritizing correlation with original:")
    print("results = generate_correlated_time_series('your_file.csv', target_correlation=0.75)")
    print()
    print("# Generate series with custom parameters:")
    print("results = generate_correlated_time_series(")
    print("    'your_file.csv', ")
    print("    target_correlation=0.5, ")
    print("    random_seed=123,")
    print("    correlation_focused=True)")
    print()
    print("# Batch generation with multiple correlation targets:")
    print("correlations = [0.9, 0.75, 0.5, 0.0, -0.3]")
    print("results_list = batch_generate_correlations('your_file.csv', correlations)")
    print()
    print("# Analyze existing time series:")
    print("analysis = analyze_existing_series('your_file.csv')")
    print()
    print("# Create visualizations:")
    print("create_visualization(results, plot_type='comprehensive')")
    print()
    print("# Access the generated data:")
    print("if results:")
    print("    original_data = results['original_data']")
    print("    generated_data = results['generated_series']")
    print("    precision_ok = results['precision_achieved']")
    print("    correlation_achieved = results['generated_stats']['correlation_with_original']")
    print("="*70)


# Convenience aliases for backward compatibility
def main(csv_file_path: str, target_correlation: float = 0.75, 
         random_seed: int = 42, require_1pct_precision: bool = True) -> Optional[Dict]:
    """
    Convenience function that maintains compatibility with original interface.
    
    This function prioritizes correlation with the original series and guarantees
    mean, variance, and correlation within 1% (autocorr reported only).
    """
    return generate_correlated_time_series(
        csv_file_path=csv_file_path,
        target_correlation=target_correlation,
        random_seed=random_seed,
        require_1pct_precision=require_1pct_precision,
        correlation_focused=True
    )


def main_enhanced_autocorr(csv_file_path: str, target_correlation: float = 0.75, 
                          random_seed: int = 42) -> Optional[Dict]:
    """
    Enhanced function focused on autocorrelation precision.
    
    This function prioritizes autocorrelation precision along with other metrics.
    """
    return generate_correlated_time_series(
        csv_file_path=csv_file_path,
        target_correlation=target_correlation,
        random_seed=random_seed,
        require_1pct_precision=True,
        correlation_focused=False  # Prioritize autocorrelation
    )
