"""
Performance Tracker
Tracks scoring accuracy over time
"""

import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

class PerformanceTracker:
    """Tracks fund score predictions vs actual performance"""
    
    TRACKER_DIR = 'performance_tracking'
    METRICS_FILE = os.path.join(TRACKER_DIR, 'metrics.json')
    
    def __init__(self):
        """Initialize tracker"""
        self._create_tracking_dir()
    
    def _create_tracking_dir(self):
        """Create tracking directory"""
        if not os.path.exists(self.TRACKER_DIR):
            os.makedirs(self.TRACKER_DIR)
    
    def record_snapshot(self, df_with_scores, run_name=None):
        """
        Record current snapshot of fund scores
        
        Args:
            df_with_scores: DataFrame with scores
            run_name: Optional name for this snapshot
        """
        if run_name is None:
            run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        
        timestamp = datetime.now().isoformat()
        
        # Get current metrics
        metrics_data = {
            'timestamp': timestamp,
            'run_name': run_name,
            'total_funds': len(df_with_scores),
            'avg_score': float(df_with_scores['Score'].mean()) if 'Score' in df_with_scores else None,
            'median_score': float(df_with_scores['Score'].median()) if 'Score' in df_with_scores else None,
        }
        
        # Load existing metrics
        metrics = self._load_metrics()
        if 'snapshots' not in metrics:
            metrics['snapshots'] = []
        
        metrics['snapshots'].append(metrics_data)
        
        # Save updated metrics
        self._save_metrics(metrics)
        
        # Save detailed snapshot
        snapshot_path = os.path.join(self.TRACKER_DIR, f"{run_name}.csv")
        df_with_scores.to_csv(snapshot_path, index=False)
        
        return metrics_data
    
    def add_actual_returns(self, symbol, actual_1y_return, actual_3y_return=None):
        """
        Add actual returns data for a fund
        
        Args:
            symbol: Fund ticker
            actual_1y_return: Actual 1-year return (%)
            actual_3y_return: Actual 3-year return (%)
        """
        metrics = self._load_metrics()
        
        if 'actual_returns' not in metrics:
            metrics['actual_returns'] = {}
        
        metrics['actual_returns'][symbol] = {
            'timestamp': datetime.now().isoformat(),
            '1y_return': actual_1y_return,
            '3y_return': actual_3y_return,
        }
        
        self._save_metrics(metrics)
    
    def calculate_prediction_accuracy(self, df_with_scores):
        """
        Calculate how well scores predicted returns
        
        Args:
            df_with_scores: DataFrame with scores
            
        Returns:
            Dictionary with accuracy metrics
        """
        metrics = self._load_metrics()
        actual_returns = metrics.get('actual_returns', {})
        
        if not actual_returns:
            return {'message': 'No actual returns data yet'}
        
        # Match scores to returns
        comparisons = []
        for symbol, ret_data in actual_returns.items():
            score_row = df_with_scores[df_with_scores['Symbol'] == symbol]
            
            if not score_row.empty and 'Score' in score_row.columns:
                score = score_row.iloc[0]['Score']
                actual_1y = ret_data.get('1y_return')
                
                if score is not None and actual_1y is not None:
                    comparisons.append({
                        'symbol': symbol,
                        'score': score,
                        'actual_return': actual_1y,
                    })
        
        if not comparisons:
            return {'message': 'Not enough data for comparison'}
        
        comp_df = pd.DataFrame(comparisons)
        
        # Calculate correlation
        correlation = comp_df['score'].corr(comp_df['actual_return'])
        
        return {
            'total_comparisons': len(comp_df),
            'correlation': correlation,
            'avg_score_in_study': comp_df['score'].mean(),
            'avg_return_in_study': comp_df['actual_return'].mean(),
            'best_predictor': 'Score' if correlation > 0.5 else 'Needs improvement',
        }
    
    def _load_metrics(self):
        """Load metrics JSON"""
        if os.path.exists(self.METRICS_FILE):
            with open(self.METRICS_FILE, 'r') as f:
                return json.load(f)
        return {'snapshots': [], 'actual_returns': {}}
    
    def _save_metrics(self, metrics):
        """Save metrics JSON"""
        with open(self.METRICS_FILE, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
    
    def get_history(self):
        """Get snapshot history"""
        metrics = self._load_metrics()
        return metrics.get('snapshots', [])
    
    def get_accuracy_report(self, df_with_scores):
        """Generate accuracy report"""
        accuracy = self.calculate_prediction_accuracy(df_with_scores)
        history = self.get_history()
        
        return {
            'accuracy': accuracy,
            'snapshots': len(history),
            'latest_snapshot': history[-1] if history else None,
        }