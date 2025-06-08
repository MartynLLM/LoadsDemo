"""
Example Usage Script

Demonstrates how to use the refactored flow data analysis package.
"""

import sys
import os

# Add the package directory to the path if running as a script
# (This would not be needed if the package is properly installed)
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the main interface functions
from flow_analysis import (
    generate_correlated_time_series,
    batch_generate_correlations,
    analyze_existing_series,
    create_visualization,
    print_usage_examples,
    print_package_info
)


def example_basic_usage():
    """Example 1: Basic usage with single correlation target."""
    print("="*70)
    print("EXAMPLE 1: BASIC USAGE")
    print("="*70)
    
    # Set your CSV file path here
    csv_file = "SavjaForClaude.csv"
    
    # Generate a correlated series
    results = generate_correlated_time_series(
        csv_file_path=csv_file,
        target_correlation=0.75,
        random_seed=42,
        require_1pct_precision=True
    )
    
    if results:
        print("\n✓ Successfully generated correlated series!")
        
        # Print key results
        achieved_corr = results['generated_stats']['correlation_with_original']
        precision_ok = results['precision_achieved']
        
        print(f"Target correlation: 0.75")
        print(f"Achieved correlation: {achieved_corr:.6f}")
        print(f"Precision achieved: {'Yes' if precision_ok else 'No'}")
        
        # Create visualization
        print("\nCreating visualizations...")
        create_visualization(results, plot_type='comprehensive')
        
        return results
    else:
        print("❌ Failed to generate series")
        return None


def example_batch_processing():
    """Example 2: Batch processing with multiple correlation targets."""
    print("\n" + "="*70)
    print("EXAMPLE 2: BATCH PROCESSING")
    print("="*70)
    
    csv_file = "SavjaForClaude.csv"
    
    # Define multiple correlation targets
    correlation_targets = [0.9, 0.75, 0.5, 0.0, -0.3]
    
    # Generate multiple series
    results_list = batch_generate_correlations(
        csv_file_path=csv_file,
        correlation_targets=correlation_targets,
        random_seed=42,
        require_1pct_precision=True
    )
    
    if results_list:
        print(f"\n✓ Successfully generated {len(results_list)} series!")
        
        # Print summary
        print("\nBatch Results Summary:")
        print("-" * 50)
        for i, result in enumerate(results_list):
            target = correlation_targets[i]
            achieved = result['generated_stats']['correlation_with_original']
            precision = result['precision_achieved']
            print(f"Target {target:+5.2f} → Achieved {achieved:+7.4f} {'✓' if precision else '✗'}")
        
        return results_list
    else:
        print("❌ Failed to generate batch series")
        return None


def example_analysis_only():
    """Example 3: Analyze existing time series without generation."""
    print("\n" + "="*70)
    print("EXAMPLE 3: ANALYSIS ONLY")
    print("="*70)
    
    csv_file = "SavjaForClaude.csv"
    
    # Analyze the existing time series
    analysis_results = analyze_existing_series(
        csv_file_path=csv_file,
        print_summary=True
    )
    
    if analysis_results:
        print("\n✓ Successfully analyzed time series!")
        return analysis_results
    else:
        print("❌ Failed to analyze time series")
        return None


def example_custom_configuration():
    """Example 4: Custom configuration and advanced usage."""
    print("\n" + "="*70)
    print("EXAMPLE 4: CUSTOM CONFIGURATION")
    print("="*70)
    
    csv_file = "SavjaForClaude.csv"
    
    # Generate with custom parameters
    results = generate_correlated_time_series(
        csv_file_path=csv_file,
        target_correlation=0.8,
        random_seed=123,
        require_1pct_precision=True,
        correlation_focused=True,  # Prioritize correlation over autocorrelation
        ensure_nonnegative=True,
        max_attempts=3
    )
    
    if results:
        print("\n✓ Successfully generated with custom configuration!")
        
        # Access detailed statistics
        orig_stats = results['original_stats']
        gen_stats = results['generated_stats']
        errors = results['error_percentages']
        
        print(f"\nDetailed Statistics:")
        print(f"Original mean: {orig_stats['mean']:.6f}")
        print(f"Generated mean: {gen_stats['mean']:.6f}")
        print(f"Mean error: {errors['mean']:.3f}%")
        print(f"Variance error: {errors['variance']:.3f}%")
        print(f"Correlation error: {errors['correlation']:.3f}%")
        print(f"Autocorrelation info: {errors['autocorrelation']:.3f}%")
        
        # Create quick visualization
        create_visualization(results, plot_type='quick', save_plots=True)
        
        return results
    else:
        print("❌ Failed to generate with custom configuration")
        return None


def example_autocorr_focused():
    """Example 5: Autocorrelation-focused generation."""
    print("\n" + "="*70)
    print("EXAMPLE 5: AUTOCORRELATION-FOCUSED")
    print("="*70)
    
    csv_file = "SavjaForClaude.csv"
    
    # Generate with autocorrelation priority
    results = generate_correlated_time_series(
        csv_file_path=csv_file,
        target_correlation=0.75,
        random_seed=456,
        require_1pct_precision=True,
        correlation_focused=False,  # Prioritize autocorrelation
        ensure_nonnegative=True
    )
    
    if results:
        print("\n✓ Successfully generated autocorrelation-focused series!")
        
        # Compare autocorrelation precision
        orig_autocorr = results['original_stats']['first_order_autocorrelation']
        gen_autocorr = results['generated_stats']['first_order_autocorrelation']
        autocorr_error = results['error_percentages']['autocorrelation']
        
        print(f"\nAutocorrelation Comparison:")
        print(f"Original: {orig_autocorr:.6f}")
        print(f"Generated: {gen_autocorr:.6f}")
        print(f"Error: {autocorr_error:.3f}%")
        
        return results
    else:
        print("❌ Failed to generate autocorrelation-focused series")
        return None


def run_all_examples():
    """Run all examples in sequence."""
    print_package_info()
    
    # Run examples
    example_1_results = example_basic_usage()
    example_2_results = example_batch_processing()
    example_3_results = example_analysis_only()
    example_4_results = example_custom_configuration()
    example_5_results = example_autocorr_focused()
    
    # Final summary
    print("\n" + "="*70)
    print("EXAMPLES COMPLETED")
    print("="*70)
    
    successful_examples = sum([
        1 if example_1_results else 0,
        1 if example_2_results else 0,
        1 if example_3_results else 0,
        1 if example_4_results else 0,
        1 if example_5_results else 0
    ])
    
    print(f"Successfully completed: {successful_examples}/5 examples")
    
    if successful_examples > 0:
        print("\n✓ Check the generated CSV files and plots in your working directory!")
        print("✓ The package is working correctly!")
    else:
        print("\n❌ No examples completed successfully.")
        print("❌ Check that 'SavjaForClaude.csv' exists in your working directory.")
    
    # Print usage guide
    print("\n" + "="*70)
    print_usage_examples()


if __name__ == "__main__":
    """
    Run this script to test all functionality of the refactored package.
    
    Make sure 'SavjaForClaude.csv' is in the same directory as this script.
    """
    
    # Check if CSV file exists
    csv_file = "SavjaForClaude.csv"
    if not os.path.exists(csv_file):
        print(f"❌ Error: '{csv_file}' not found in current directory.")
        print(f"Please ensure the CSV file is in the same directory as this script.")
        print(f"Current directory: {os.getcwd()}")
        sys.exit(1)
    
    # Run all examples
    try:
        run_all_examples()
    except KeyboardInterrupt:
        print("\n\n⚠️ Examples interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nExample script completed.")
