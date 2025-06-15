"""
SQLite Database Analysis Script

Provides comprehensive analysis tools for the SQLite time series database.
Demonstrates efficient querying and analysis using SQL and pandas.
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os


class SQLiteTimeSeriesAnalyzer:
    """Analyzer for SQLite time series database."""
    
    def __init__(self, db_path: str = "savja_timeseries.db"):
        """Initialize analyzer with database path."""
        self.db_path = db_path
        self.conn = None
        
    def connect(self) -> bool:
        """Connect to SQLite database."""
        try:
            if not os.path.exists(self.db_path):
                print(f"❌ Database not found: {self.db_path}")
                return False
            
            self.conn = sqlite3.connect(self.db_path)
            print(f"✅ Connected to database: {self.db_path}")
            
            # Verify tables exist
            tables = pd.read_sql(
                "SELECT name FROM sqlite_master WHERE type='table'", 
                self.conn
            )['name'].tolist()
            
            expected_tables = ['timeseries_data', 'metadata']
            missing_tables = [t for t in expected_tables if t not in tables]
            
            if missing_tables:
                print(f"❌ Missing tables: {missing_tables}")
                return False
            
            print(f"✅ Found tables: {tables}")
            return True
            
        except Exception as e:
            print(f"❌ Error connecting to database: {e}")
            return False
    
    def get_database_info(self) -> Dict:
        """Get basic information about the database."""
        if not self.conn:
            return {}
        
        try:
            # Get table sizes
            timeseries_count = pd.read_sql(
                "SELECT COUNT(*) as count FROM timeseries_data", 
                self.conn
            ).iloc[0]['count']
            
            metadata_count = pd.read_sql(
                "SELECT COUNT(*) as count FROM metadata", 
                self.conn
            ).iloc[0]['count']
            
            # Get date range
            date_range = pd.read_sql(
                "SELECT MIN(date) as min_date, MAX(date) as max_date FROM timeseries_data",
                self.conn
            ).iloc[0]
            
            # Get database size
            db_size = os.path.getsize(self.db_path) / 1024 / 1024  # MB
            
            info = {
                'timeseries_rows': timeseries_count,
                'metadata_rows': metadata_count,
                'combinations': metadata_count,
                'points_per_combination': timeseries_count // metadata_count if metadata_count > 0 else 0,
                'date_range': f"{date_range['min_date']} to {date_range['max_date']}",
                'database_size_mb': db_size
            }
            
            print(f"\n" + "="*50)
            print("DATABASE INFORMATION")
            print("="*50)
            print(f"Database file: {self.db_path}")
            print(f"Database size: {db_size:.1f} MB")
            print(f"Time series rows: {timeseries_count:,}")
            print(f"Metadata rows: {metadata_count}")
            print(f"Combinations: {metadata_count}")
            print(f"Points per combination: {info['points_per_combination']:,}")
            print(f"Date range: {info['date_range']}")
            
            return info
            
        except Exception as e:
            print(f"❌ Error getting database info: {e}")
            return {}
    
    def validate_perfect_case(self) -> bool:
        """Validate the perfect case using SQL queries."""
        print("\n" + "="*50)
        print("VALIDATING PERFECT CASE (SQL)")
        print("="*50)
        
        try:
            # Get perfect case metadata
            perfect_meta = pd.read_sql("""
                SELECT * FROM metadata 
                WHERE is_perfect_case = 1
            """, self.conn)
            
            if len(perfect_meta) == 0:
                print("❌ No perfect case found in metadata")
                return False
            
            combo_id = perfect_meta.iloc[0]['combination_id']
            print(f"Perfect case ID: {combo_id}")
            
            # Get time series data and calculate differences using SQL
            perfect_analysis = pd.read_sql(f"""
                SELECT 
                    combination_id,
                    COUNT(*) as total_points,
                    AVG(ABS(generated_value - input_value)) as avg_abs_diff,
                    MAX(ABS(generated_value - input_value)) as max_abs_diff,
                    MIN(generated_value - input_value) as min_diff,
                    MAX(generated_value - input_value) as max_diff
                FROM timeseries_data 
                WHERE combination_id = '{combo_id}'
            """, self.conn)
            
            if len(perfect_analysis) == 0:
                print("❌ No time series data found for perfect case")
                return False
            
            analysis = perfect_analysis.iloc[0]
            max_diff = analysis['max_abs_diff']
            avg_diff = analysis['avg_abs_diff']
            
            print(f"Difference analysis:")
            print(f"  Total points: {analysis['total_points']}")
            print(f"  Maximum absolute difference: {max_diff:.8f}")
            print(f"  Average absolute difference: {avg_diff:.8f}")
            print(f"  Difference range: [{analysis['min_diff']:.8f}, {analysis['max_diff']:.8f}]")
            
            # Get metadata validation
            meta_row = perfect_meta.iloc[0]
            print(f"\nMetadata validation:")
            print(f"  Correlation error: {meta_row['corr_error_pct']:.6f}%")
            print(f"  Mean error: {meta_row['mean_error_pct']:.6f}%")
            print(f"  Variance error: {meta_row['var_error_pct']:.6f}%")
            print(f"  Max error: {meta_row['max_error_pct']:.6f}%")
            
            # Determine success
            tolerance = 1e-6
            is_perfect = max_diff < tolerance
            
            if is_perfect:
                print(f"\n✅ PERFECT CASE VALIDATED!")
                print(f"   Generated series is identical to input (within {tolerance})")
                return True
            else:
                print(f"\n❌ PERFECT CASE FAILED!")
                print(f"   Difference {max_diff:.8f} exceeds tolerance {tolerance}")
                return False
                
        except Exception as e:
            print(f"❌ Error validating perfect case: {e}")
            return False
    
    def analyze_quality_metrics(self) -> Dict:
        """Analyze quality metrics using SQL aggregations."""
        print("\n" + "="*50)
        print("QUALITY METRICS ANALYSIS (SQL)")
        print("="*50)
        
        try:
            # Get overall quality summary using SQL
            quality_summary = pd.read_sql("""
                SELECT 
                    COUNT(*) as total_combinations,
                    SUM(CASE WHEN max_error_pct <= 1.0 THEN 1 ELSE 0 END) as success_1pct,
                    SUM(CASE WHEN max_error_pct <= 2.0 THEN 1 ELSE 0 END) as success_2pct,
                    SUM(CASE WHEN max_error_pct <= 5.0 THEN 1 ELSE 0 END) as success_5pct,
                    SUM(CASE WHEN negative_values_count > 0 THEN 1 ELSE 0 END) as constraint_violations,
                    SUM(CASE WHEN is_perfect_case = 1 THEN 1 ELSE 0 END) as perfect_cases,
                    AVG(mean_error_pct) as avg_mean_error,
                    AVG(var_error_pct) as avg_var_error,
                    AVG(corr_error_pct) as avg_corr_error,
                    MAX(mean_error_pct) as max_mean_error,
                    MAX(var_error_pct) as max_var_error,
                    MAX(corr_error_pct) as max_corr_error
                FROM metadata
            """, self.conn)
            
            summary = quality_summary.iloc[0]
            total = summary['total_combinations']
            
            print(f"Overall Quality Summary:")
            print(f"  Total combinations: {total}")
            print(f"  Perfect cases: {summary['perfect_cases']}")
            print(f"  Within 1% error: {summary['success_1pct']}/{total} ({summary['success_1pct']/total*100:.1f}%)")
            print(f"  Within 2% error: {summary['success_2pct']}/{total} ({summary['success_2pct']/total*100:.1f}%)")
            print(f"  Within 5% error: {summary['success_5pct']}/{total} ({summary['success_5pct']/total*100:.1f}%)")
            print(f"  Constraint violations: {summary['constraint_violations']}/{total}")
            
            print(f"\nError Statistics:")
            print(f"  Mean errors - Avg: {summary['avg_mean_error']:.3f}%, Max: {summary['max_mean_error']:.3f}%")
            print(f"  Variance errors - Avg: {summary['avg_var_error']:.3f}%, Max: {summary['max_var_error']:.3f}%")
            print(f"  Correlation errors - Avg: {summary['avg_corr_error']:.3f}%, Max: {summary['max_corr_error']:.3f}%")
            
            # Quality by correlation target
            quality_by_corr = pd.read_sql("""
                SELECT 
                    target_correlation,
                    COUNT(*) as count,
                    AVG(max_error_pct) as avg_max_error,
                    SUM(CASE WHEN max_error_pct <= 1.0 THEN 1 ELSE 0 END) as success_1pct
                FROM metadata
                GROUP BY target_correlation
                ORDER BY target_correlation
            """, self.conn)
            
            print(f"\nQuality by Correlation Target:")
            for _, row in quality_by_corr.iterrows():
                success_rate = row['success_1pct'] / row['count'] * 100
                print(f"  Correlation {row['target_correlation']:.1f}: "
                      f"Avg error {row['avg_max_error']:.2f}%, "
                      f"Success rate {success_rate:.1f}%")
            
            return summary.to_dict()
            
        except Exception as e:
            print(f"❌ Error analyzing quality metrics: {e}")
            return {}
    
    def get_combination_data(self, combination_id: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Get time series and metadata for a specific combination using SQL."""
        try:
            # Get time series data
            ts_data = pd.read_sql(f"""
                SELECT date, input_value, generated_value,
                       (generated_value - input_value) as difference
                FROM timeseries_data 
                WHERE combination_id = '{combination_id}'
                ORDER BY date
            """, self.conn)
            
            # Get metadata
            meta_data = pd.read_sql(f"""
                SELECT * FROM metadata 
                WHERE combination_id = '{combination_id}'
            """, self.conn)
            
            if len(meta_data) > 0:
                meta_data = meta_data.iloc[0]
            else:
                meta_data = None
            
            return ts_data, meta_data
            
        except Exception as e:
            print(f"❌ Error getting combination data: {e}")
            return pd.DataFrame(), None
    
    def find_best_worst_combinations(self) -> Dict:
        """Find best and worst performing combinations using SQL."""
        print("\n" + "="*50)
        print("BEST AND WORST COMBINATIONS")
        print("="*50)
        
        try:
            # Best combinations (lowest max error)
            best_combos = pd.read_sql("""
                SELECT combination_id, mean_percentage, variance_percentage, 
                       target_correlation, max_error_pct, is_perfect_case
                FROM metadata
                ORDER BY max_error_pct ASC
                LIMIT 5
            """, self.conn)
            
            # Worst combinations (highest max error)
            worst_combos = pd.read_sql("""
                SELECT combination_id, mean_percentage, variance_percentage,
                       target_correlation, max_error_pct, is_perfect_case
                FROM metadata
                ORDER BY max_error_pct DESC
                LIMIT 5
            """, self.conn)
            
            print("Best 5 combinations (lowest error):")
            for _, row in best_combos.iterrows():
                perfect_flag = " (Perfect case)" if row['is_perfect_case'] else ""
                print(f"  {row['combination_id']}: {row['max_error_pct']:.4f}% error{perfect_flag}")
            
            print("\nWorst 5 combinations (highest error):")
            for _, row in worst_combos.iterrows():
                print(f"  {row['combination_id']}: {row['max_error_pct']:.4f}% error")
            
            return {
                'best_combinations': best_combos,
                'worst_combinations': worst_combos
            }
            
        except Exception as e:
            print(f"❌ Error finding best/worst combinations: {e}")
            return {}
    
    def export_analysis_reports(self, output_dir: str = "."):
        """Export various analysis reports to CSV files."""
        print(f"\n" + "="*50)
        print("EXPORTING ANALYSIS REPORTS")
        print("="*50)
        
        try:
            # Quality summary by parameters
            quality_by_params = pd.read_sql("""
                SELECT 
                    mean_percentage, variance_percentage,
                    COUNT(*) as combination_count,
                    AVG(max_error_pct) as avg_max_error,
                    SUM(CASE WHEN max_error_pct <= 1.0 THEN 1 ELSE 0 END) as success_1pct,
                    SUM(negative_values_count) as total_negative_values
                FROM metadata
                GROUP BY mean_percentage, variance_percentage
                ORDER BY mean_percentage, variance_percentage
            """, self.conn)
            
            quality_file = os.path.join(output_dir, "quality_by_parameters.csv")
            quality_by_params.to_csv(quality_file, index=False)
            print(f"✅ Quality by parameters: {quality_file}")
            
            # Correlation analysis
            correlation_analysis = pd.read_sql("""
                SELECT 
                    target_correlation,
                    COUNT(*) as combination_count,
                    AVG(achieved_correlation) as avg_achieved_correlation,
                    AVG(corr_error_pct) as avg_correlation_error,
                    MIN(achieved_correlation) as min_achieved_correlation,
                    MAX(achieved_correlation) as max_achieved_correlation
                FROM metadata
                GROUP BY target_correlation
                ORDER BY target_correlation
            """, self.conn)
            
            corr_file = os.path.join(output_dir, "correlation_analysis.csv")
            correlation_analysis.to_csv(corr_file, index=False)
            print(f"✅ Correlation analysis: {corr_file}")
            
            # Full metadata export
            metadata_file = os.path.join(output_dir, "full_metadata_export.csv")
            full_metadata = pd.read_sql("SELECT * FROM metadata ORDER BY combination_id", self.conn)
            full_metadata.to_csv(metadata_file, index=False)
            print(f"✅ Full metadata: {metadata_file}")
            
            # Perfect case detailed analysis
            perfect_case_data = pd.read_sql("""
                SELECT t.*, m.achieved_correlation, m.max_error_pct
                FROM timeseries_data t
                JOIN metadata m ON t.combination_id = m.combination_id
                WHERE m.is_perfect_case = 1
                ORDER BY t.date
            """, self.conn)
            
            if len(perfect_case_data) > 0:
                perfect_file = os.path.join(output_dir, "perfect_case_analysis.csv")
                perfect_case_data['difference'] = perfect_case_data['generated_value'] - perfect_case_data['input_value']
                perfect_case_data.to_csv(perfect_file, index=False)
                print(f"✅ Perfect case analysis: {perfect_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error exporting reports: {e}")
            return False
    
    def create_sql_examples(self):
        """Show useful SQL query examples."""
        print("\n" + "="*60)
        print("USEFUL SQL QUERY EXAMPLES")
        print("="*60)
        
        examples = [
            ("Get perfect case time series", """
                SELECT date, input_value, generated_value,
                       (generated_value - input_value) as difference
                FROM timeseries_data 
                WHERE combination_id = 'M100_V100_C1.0'
                ORDER BY date;
            """),
            
            ("Find high correlation combinations", """
                SELECT combination_id, achieved_correlation, max_error_pct
                FROM metadata 
                WHERE target_correlation >= 0.9
                ORDER BY achieved_correlation DESC;
            """),
            
            ("Get statistical summary by mean percentage", """
                SELECT mean_percentage,
                       COUNT(*) as combinations,
                       AVG(max_error_pct) as avg_error,
                       MIN(achieved_correlation) as min_correlation,
                       MAX(achieved_correlation) as max_correlation
                FROM metadata
                GROUP BY mean_percentage
                ORDER BY mean_percentage;
            """),
            
            ("Find combinations with constraint violations", """
                SELECT combination_id, negative_values_count, max_error_pct
                FROM metadata
                WHERE negative_values_count > 0
                ORDER BY negative_values_count DESC;
            """),
            
            ("Compare input vs generated for specific date", """
                SELECT combination_id, input_value, generated_value,
                       (generated_value - input_value) as difference
                FROM timeseries_data 
                WHERE date = '1980-01-01'
                ORDER BY combination_id;
            """),
            
            ("Join tables for correlation analysis", """
                SELECT m.target_correlation, m.achieved_correlation,
                       AVG(t.generated_value) as avg_generated,
                       AVG(t.input_value) as avg_input
                FROM metadata m
                JOIN timeseries_data t ON m.combination_id = t.combination_id
                GROUP BY m.combination_id
                ORDER BY m.target_correlation;
            """)
        ]
        
        for description, query in examples:
            print(f"\n-- {description}")
            print(query.strip())
    
    def run_complete_analysis(self):
        """Run complete analysis pipeline."""
        print("="*70)
        print("SQLITE COMPLETE ANALYSIS")
        print("="*70)
        
        # Connect to database
        if not self.connect():
            print("❌ Failed to connect to database. Exiting.")
            return False
        
        try:
            # Get database info
            db_info = self.get_database_info()
            
            # Validate perfect case
            perfect_ok = self.validate_perfect_case()
            
            # Analyze quality
            quality_metrics = self.analyze_quality_metrics()
            
            # Find best/worst combinations
            best_worst = self.find_best_worst_combinations()
            
            # Export reports
            self.export_analysis_reports()
            
            # Show SQL examples
            self.create_sql_examples()
            
            # Final summary
            print("\n" + "="*70)
            print("SQLITE ANALYSIS SUMMARY")
            print("="*70)
            print(f"Database: {self.db_path} ({db_info.get('database_size_mb', 0):.1f} MB)")
            print(f"Perfect case validation: {'✅ PASSED' if perfect_ok else '❌ FAILED'}")
            
            if quality_metrics:
                total = quality_metrics.get('total_combinations', 0)
                success_1pct = quality_metrics.get('success_1pct', 0)
                success_5pct = quality_metrics.get('success_5pct', 0)
                violations = quality_metrics.get('constraint_violations', 0)
                
                print(f"Overall quality: {success_1pct}/{total} within 1% error ({success_1pct/total*100:.1f}%)")
                print(f"Constraint compliance: {total - violations}/{total} combinations")
            
            print(f"\n📊 Exported analysis files:")
            print(f"  - quality_by_parameters.csv")
            print(f"  - correlation_analysis.csv")
            print(f"  - full_metadata_export.csv")
            print(f"  - perfect_case_analysis.csv")
            
            if perfect_ok and quality_metrics and success_5pct / total > 0.9:
                print(f"\n🎉 EXCELLENT SQLITE RESULTS!")
                print(f"   ✅ Perfect case working correctly")
                print(f"   ✅ >90% combinations within 5% error")
                print(f"   ✅ Database ready for production analysis")
            else:
                print(f"\n⚠️  SQLITE RESULTS NEED REVIEW")
                if not perfect_ok:
                    print(f"   ❌ Perfect case validation failed")
                if quality_metrics and success_5pct / total <= 0.9:
                    print(f"   ❌ <90% combinations within 5% error")
            
            return True
            
        finally:
            if self.conn:
                self.conn.close()
                print(f"\n🔐 Database connection closed")
    
    def __del__(self):
        """Ensure database connection is closed."""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()


def main():
    """Main analysis function."""
    print("🔍 SQLITE TIME SERIES DATABASE ANALYZER")
    print("Comprehensive analysis of SQLite time series database")
    print()
    
    # Check if database exists
    db_path = "savja_timeseries.db"
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        print("Please run the SQLite generator first:")
        print("python sqlite_timeseries_generator.py")
        return False
    
    # Initialize analyzer
    analyzer = SQLiteTimeSeriesAnalyzer(db_path)
    
    # Run analysis
    success = analyzer.run_complete_analysis()
    
    if success:
        print(f"\n✅ SQLite analysis completed successfully!")
        print(f"\n💡 Next steps:")
        print(f"1. Review exported CSV files for detailed analysis")
        print(f"2. Use SQL queries to explore specific combinations")
        print(f"3. Connect to database with your preferred tools")
        print(f"4. Create custom visualizations and reports")
    else:
        print(f"\n❌ SQLite analysis failed!")
    
    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)