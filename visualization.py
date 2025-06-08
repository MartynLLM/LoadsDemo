"""
Visualization Module

Handles plotting and visualization of time series comparison results.
"""

import numpy as np
from typing import Dict, Optional, List
from datetime import datetime

# Handle plotting imports gracefully
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Note: matplotlib not available, plotting functionality disabled")


class TimeSeriesVisualizer:
    """Class for visualizing time series comparison results."""
    
    def __init__(self):
        if not MATPLOTLIB_AVAILABLE:
            print("Warning: matplotlib not available. Install with: pip install matplotlib")
    
    def plot_comprehensive_comparison(self, results: Dict, save_plots: bool = True, 
                                    show_plots: bool = True) -> Optional[object]:
        """
        Create comprehensive plots comparing original and generated time series.
        
        Parameters:
        results (dict): Results from time series generation
        save_plots (bool): If True, save plots to files
        show_plots (bool): If True, display plots
        
        Returns:
        matplotlib.figure.Figure: Figure object if matplotlib available, None otherwise
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available - cannot create plots")
            return None
        
        if results is None:
            print("No results to plot")
            return None
        
        # Extract data
        original_data = results['original_data']
        generated_data = results['generated_series']
        original_stats = results['original_stats']
        generated_stats = results['generated_stats']
        target_correlation = results['target_correlation']
        
        # Get date information
        dates = original_stats.get('dates', list(range(len(original_data))))
        has_dates = original_stats.get('has_dates', False)
        
        # Sort data by time if we have dates
        original_data, generated_data, dates = self._sort_by_time(
            original_data, generated_data, dates, has_dates)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], width_ratios=[2, 1])
        
        # 1. Time series plot (main plot)
        ax1 = self._create_time_series_plot(
            fig, gs[0, :], original_data, generated_data, dates, 
            has_dates, generated_stats, target_correlation)
        
        # 2. Scatter plot (observed vs generated)
        ax2 = self._create_scatter_plot(
            fig, gs[1, 0], original_data, generated_data, generated_stats)
        
        # 3. Statistics summary
        self._create_statistics_panel(
            fig, gs[1, 1], results, original_stats, generated_stats)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save and show plots
        if save_plots:
            filename = self._create_plot_filename(target_correlation, "comprehensive")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Comprehensive plot saved as: {filename}")
        
        if show_plots:
            plt.show()
        
        return fig
    
    def plot_quick_comparison(self, results: Dict, save_plots: bool = False) -> Optional[object]:
        """
        Create a quick 2-panel comparison plot.
        
        Parameters:
        results (dict): Results from time series generation
        save_plots (bool): If True, save plot to file
        
        Returns:
        matplotlib.figure.Figure: Figure object if matplotlib available, None otherwise
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available - cannot create plots")
            return None
        
        if results is None:
            print("No results to plot")
            return None
        
        # Extract data
        original_data = results['original_data']
        generated_data = results['generated_series']
        generated_stats = results['generated_stats']
        target_correlation = results['target_correlation']
        
        # Create 2-panel plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Time series plot
        time_index = np.arange(len(original_data))
        ax1.plot(time_index, original_data, 'b-', label='Observed', linewidth=1.5, alpha=0.8)
        ax1.plot(time_index, generated_data, 'r-', label='Generated', linewidth=1.5, alpha=0.8)
        ax1.set_xlabel('Time Index')
        ax1.set_ylabel('Flow')
        ax1.set_title(f'Time Series (r={generated_stats["correlation_with_original"]:.3f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Scatter plot
        self._add_scatter_plot_content(ax2, original_data, generated_data)
        
        plt.tight_layout()
        
        # Save if requested
        if save_plots:
            filename = self._create_plot_filename(target_correlation, "quick")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Quick plot saved as: {filename}")
        
        plt.show()
        return fig
    
    def _sort_by_time(self, original_data: np.ndarray, generated_data: np.ndarray,
                     dates: List, has_dates: bool) -> tuple:
        """Sort data by time if dates are available."""
        if has_dates and dates and isinstance(dates[0], datetime):
            print("Sorting plot data by date for proper time series order...")
            # Create combined data for sorting
            combined_data = list(zip(dates, original_data, generated_data))
            # Sort by date
            combined_data.sort(key=lambda x: x[0])
            # Separate back into components
            dates, original_data, generated_data = zip(*combined_data)
            dates = list(dates)
            original_data = list(original_data)
            generated_data = list(generated_data)
            print(f"Plot data sorted from {dates[0].strftime('%d/%m/%Y')} to {dates[-1].strftime('%d/%m/%Y')}")
        
        return original_data, generated_data, dates
    
    def _create_time_series_plot(self, fig, gs_position, original_data: np.ndarray,
                               generated_data: np.ndarray, dates: List, has_dates: bool,
                               generated_stats: Dict, target_correlation: float):
        """Create the main time series plot."""
        ax1 = fig.add_subplot(gs_position)
        
        if has_dates and isinstance(dates[0], datetime):
            # Plot with actual dates
            ax1.plot(dates, original_data, 'b-', label='Observed', linewidth=1.5, alpha=0.8)
            ax1.plot(dates, generated_data, 'r-', label='Generated', linewidth=1.5, alpha=0.8)
            
            # Format x-axis to show years without overlap
            years = mdates.YearLocator()
            years_fmt = mdates.DateFormatter('%Y')
            ax1.xaxis.set_major_locator(years)
            ax1.xaxis.set_major_formatter(years_fmt)
            
            # Add month locators for better granularity if data span is short
            months = mdates.MonthLocator()
            ax1.xaxis.set_minor_locator(months)
            
            # Rotate labels if needed
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            ax1.set_xlabel('Year')
            
            # Show date range in title
            date_range = f"({dates[0].strftime('%d/%m/%Y')} to {dates[-1].strftime('%d/%m/%Y')})"
        else:
            # Plot with index
            time_index = np.arange(len(original_data))
            ax1.plot(time_index, original_data, 'b-', label='Observed', linewidth=1.5, alpha=0.8)
            ax1.plot(time_index, generated_data, 'r-', label='Generated', linewidth=1.5, alpha=0.8)
            ax1.set_xlabel('Time Index')
            date_range = ""
        
        ax1.set_ylabel('Flow')
        ax1.set_title(f'Time Series Comparison {date_range}\nTarget Correlation: {target_correlation:.3f}, '
                     f'Achieved: {generated_stats["correlation_with_original"]:.3f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        return ax1
    
    def _create_scatter_plot(self, fig, gs_position, original_data: np.ndarray,
                           generated_data: np.ndarray, generated_stats: Dict):
        """Create the scatter plot."""
        ax2 = fig.add_subplot(gs_position)
        self._add_scatter_plot_content(ax2, original_data, generated_data)
        return ax2
    
    def _add_scatter_plot_content(self, ax, original_data: np.ndarray, generated_data: np.ndarray):
        """Add content to scatter plot axis."""
        # Create scatter plot
        ax.scatter(original_data, generated_data, alpha=0.6, s=20, c='blue', edgecolors='none')
        
        # Add 1:1 line
        min_val = min(np.min(original_data), np.min(generated_data))
        max_val = max(np.max(original_data), np.max(generated_data))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='1:1 Line')
        
        # Add trend line
        correlation = np.corrcoef(original_data, generated_data)[0, 1]
        z = np.polyfit(original_data, generated_data, 1)
        p = np.poly1d(z)
        ax.plot(original_data, p(original_data), 'g-', alpha=0.8, linewidth=1.5, 
                label=f'Trend (R²={correlation**2:.3f})')
        
        ax.set_xlabel('Observed Flow')
        ax.set_ylabel('Generated Flow')
        ax.set_title('Observed vs Generated')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Make scatter plot square
        ax.set_aspect('equal', adjustable='box')
    
    def _create_statistics_panel(self, fig, gs_position, results: Dict, 
                               original_stats: Dict, generated_stats: Dict):
        """Create the statistics summary panel."""
        ax3 = fig.add_subplot(gs_position)
        ax3.axis('off')
        
        # Create statistics text
        error_pcts = results.get('error_percentages', {})
        
        stats_text = f"""Statistical Comparison:

Original Series:
  Mean: {original_stats['mean']:.4f}
  Variance: {original_stats['variance']:.4f}
  Std Dev: {original_stats['standard_deviation']:.4f}
  Autocorr: {original_stats['first_order_autocorrelation']:.4f}

Generated Series:
  Mean: {generated_stats['mean']:.4f}
  Variance: {generated_stats['variance']:.4f}
  Std Dev: {generated_stats['standard_deviation']:.4f}
  Autocorr: {generated_stats['first_order_autocorrelation']:.4f}

Precision Achieved:
  Mean error: {error_pcts.get('mean', 0):.3f}%
  Variance error: {error_pcts.get('variance', 0):.3f}%
  Correlation error: {error_pcts.get('correlation', 0):.3f}%
  Autocorr info: {error_pcts.get('autocorrelation', 0):.3f}%

Properties:
  Min value: {generated_stats['min_value']:.4f}
  Max value: {generated_stats['max_value']:.4f}
  Non-negative: {'✓' if generated_stats['num_negative_values'] == 0 else '✗'}
  Precision OK: {'✓' if results.get('precision_achieved', False) else '✗'}

Data Range: {len(results['original_data'])} points"""
        
        ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    def _create_plot_filename(self, target_correlation: float, plot_type: str) -> str:
        """Create standardized plot filename."""
        correlation_str = str(target_correlation).replace('.', 'p').replace('-', 'neg')
        return f"time_series_{plot_type}_corr_{correlation_str}.png"
    
    def plot_error_analysis(self, results_list: List[Dict], save_plots: bool = True):
        """
        Plot error analysis across multiple correlation targets.
        
        Parameters:
        results_list (list): List of results dictionaries
        save_plots (bool): If True, save plot to file
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available - cannot create error analysis plot")
            return None
        
        if not results_list:
            print("No results to analyze")
            return None
        
        # Extract data for analysis
        correlations = []
        mean_errors = []
        var_errors = []
        corr_errors = []
        autocorr_errors = []
        precision_achieved = []
        
        for result in results_list:
            correlations.append(result['target_correlation'])
            error_pcts = result.get('error_percentages', {})
            mean_errors.append(error_pcts.get('mean', 0))
            var_errors.append(error_pcts.get('variance', 0))
            corr_errors.append(error_pcts.get('correlation', 0))
            autocorr_errors.append(error_pcts.get('autocorrelation', 0))
            precision_achieved.append(result.get('precision_achieved', False))
        
        # Create error analysis plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot error trends
        ax1.plot(correlations, mean_errors, 'o-', label='Mean Error', color='blue')
        ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='1% Target')
        ax1.set_xlabel('Target Correlation')
        ax1.set_ylabel('Error %')
        ax1.set_title('Mean Error vs Target Correlation')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(correlations, var_errors, 'o-', label='Variance Error', color='green')
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='1% Target')
        ax2.set_xlabel('Target Correlation')
        ax2.set_ylabel('Error %')
        ax2.set_title('Variance Error vs Target Correlation')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3.plot(correlations, corr_errors, 'o-', label='Correlation Error', color='orange')
        ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='1% Target')
        ax3.set_xlabel('Target Correlation')
        ax3.set_ylabel('Error %')
        ax3.set_title('Correlation Error vs Target Correlation')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4.plot(correlations, autocorr_errors, 'o-', label='Autocorr Error (Info)', color='purple')
        ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='1% Reference')
        ax4.set_xlabel('Target Correlation')
        ax4.set_ylabel('Error %')
        ax4.set_title('Autocorrelation Error vs Target Correlation')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Add precision achievement markers
        for i, (corr, achieved) in enumerate(zip(correlations, precision_achieved)):
            if achieved:
                ax1.plot(corr, mean_errors[i], 'go', markersize=8, markeredgecolor='darkgreen')
                ax2.plot(corr, var_errors[i], 'go', markersize=8, markeredgecolor='darkgreen')
                ax3.plot(corr, corr_errors[i], 'go', markersize=8, markeredgecolor='darkgreen')
        
        if save_plots:
            plt.savefig('error_analysis.png', dpi=300, bbox_inches='tight')
            print("Error analysis plot saved as: error_analysis.png")
        
        plt.show()
        return fig


def create_visualizer() -> TimeSeriesVisualizer:
    """Factory function to create a visualizer."""
    return TimeSeriesVisualizer()


def plot_results(results: Dict, plot_type: str = 'comprehensive', 
                save_plots: bool = True, show_plots: bool = True) -> Optional[object]:
    """
    Convenience function to plot results.
    
    Parameters:
    results (dict): Results from time series generation
    plot_type (str): Type of plot ('comprehensive' or 'quick')
    save_plots (bool): If True, save plots to files
    show_plots (bool): If True, display plots
    
    Returns:
    matplotlib.figure.Figure: Figure object if successful, None otherwise
    """
    visualizer = create_visualizer()
    
    if plot_type == 'comprehensive':
        return visualizer.plot_comprehensive_comparison(results, save_plots, show_plots)
    elif plot_type == 'quick':
        return visualizer.plot_quick_comparison(results, save_plots)
    else:
        print(f"Unknown plot type: {plot_type}. Use 'comprehensive' or 'quick'.")
        return None