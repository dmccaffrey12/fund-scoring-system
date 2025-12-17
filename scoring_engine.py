# scoring_engine.py
"""
Core scoring engine for passive and active funds
Handles percentile calculation and weighted score computation
"""

import pandas as pd
import numpy as np
from scipy import stats
from config import (
    PASSIVE_WEIGHTS, ACTIVE_WEIGHTS, METRIC_DIRECTION,
    MISSING_DATA_THRESHOLD, CSV_COLUMNS
)

class FundScoringEngine:
    """Main class for calculating fund scores"""
    
    def __init__(self, df):
        """Initialize with fund dataframe"""
        self.df = df.copy()
        self.passive_funds = self.df[self.df['Index Fund'] == True]
        self.active_funds = self.df[self.df['Index Fund'] == False]
        self.scores = pd.DataFrame()
        
    def calculate_all_scores(self):
        """Calculate scores for all funds"""
        scores_list = []
        
        for idx, row in self.df.iterrows():
            if row['Index Fund']:
                score_data = self._score_passive_fund(row)
            else:
                score_data = self._score_active_fund(row)
            scores_list.append(score_data)
        
        self.scores = pd.DataFrame(scores_list)
        return self.scores
    
    def _score_passive_fund(self, fund_row):
        """Score a passive index fund"""
        symbol = fund_row['Symbol']
        fund_type = 'Passive'
        category = fund_row['Category Name']
        
        # Get category data (only passive funds)
        category_data = self.passive_funds[
            self.passive_funds['Category Name'] == category
        ]
        
        if len(category_data) < 2:
            # Not enough funds in category
            return {
                'Symbol': symbol,
                'Name': fund_row['Name'],
                'Fund_Type': fund_type,
                'Category': category,
                'Score': np.nan,
                'Percentile': np.nan,
                'Data_Completeness': 0,
                'Status': 'Insufficient data'
            }
        
        # Define metrics for passive funds
        metrics = [
            'Net Expense Ratio',
            'Tracking Error (vs Category) (3Y)',
            'Tracking Error (vs Category) (5Y)',
            'Tracking Error (vs Category) (10Y)',
            'R-Squared (vs Category) (5Y)',
            'Share Class Assets Under Management',
            'Downside (vs Category) (5Y)',
            'Downside (vs Category) (10Y)',
            'Max Drawdown (5Y)',
            'Max Drawdown (10Y)',
        ]
        
        # Calculate percentiles within category
        percentiles = {}
        valid_metrics = 0
        
        for metric in metrics:
            if pd.notna(fund_row[metric]):
                valid_metrics += 1
                category_values = category_data[metric].dropna()
                if len(category_values) > 1:
                    percentile = self._calculate_percentile(
                        fund_row[metric], 
                        category_values,
                        METRIC_DIRECTION.get(self._metric_to_key(metric), 1)
                    )
                    percentiles[metric] = percentile
                else:
                    percentiles[metric] = 50.0
            else:
                percentiles[metric] = np.nan
        
        # Check data completeness
        data_completeness = valid_metrics / len(metrics)
        if data_completeness < (1 - MISSING_DATA_THRESHOLD):
            return {
                'Symbol': symbol,
                'Name': fund_row['Name'],
                'Fund_Type': fund_type,
                'Category': category,
                'Score': np.nan,
                'Percentile': np.nan,
                'Data_Completeness': data_completeness,
                'Status': 'Insufficient data (>30% missing)'
            }
        
        # Calculate weighted score with rescaling
        score = self._calculate_weighted_score(percentiles, PASSIVE_WEIGHTS)
        percentile_rank = self._rank_fund_percentile(
            symbol, category, fund_type, score
        )
        
        return {
            'Symbol': symbol,
            'Name': fund_row['Name'],
            'Fund_Type': fund_type,
            'Category': category,
            'Score': score,
            'Percentile': percentile_rank,
            'Data_Completeness': data_completeness,
            'Status': 'Complete',
            'Expense_Ratio': fund_row['Net Expense Ratio'],
        }
    
    def _score_active_fund(self, fund_row):
        """Score an active fund"""
        symbol = fund_row['Symbol']
        fund_type = 'Active'
        category = fund_row['Category Name']
        
        # Get category data (only active funds)
        category_data = self.active_funds[
            self.active_funds['Category Name'] == category
        ]
        
        if len(category_data) < 2:
            return {
                'Symbol': symbol,
                'Name': fund_row['Name'],
                'Fund_Type': fund_type,
                'Category': category,
                'Score': np.nan,
                'Percentile': np.nan,
                'Data_Completeness': 0,
                'Status': 'Insufficient data'
            }
        
        # Define metrics for active funds
        metrics = [
            'Net Expense Ratio',
            'Information Ratio (vs Category) (3Y)',
            'Information Ratio (vs Category) (5Y)',
            'Information Ratio (vs Category) (10Y)',
            'Historical Sortino (3Y)',
            'Historical Sortino (5Y)',
            'Historical Sortino (10Y)',
            'Max Drawdown (5Y)',
            'Max Drawdown (10Y)',
            'Downside (vs Category) (5Y)',
            'Downside (vs Category) (10Y)',
            '3 Year Total Returns (Daily)',
            '5 Year Total Returns (Daily)',
            '10 Year Total Returns (Daily)',
            'Upside (vs Category) (5Y)',
            'Upside (vs Category) (10Y)',
        ]
        
        # Calculate percentiles within category
        percentiles = {}
        valid_metrics = 0
        
        for metric in metrics:
            if pd.notna(fund_row[metric]):
                valid_metrics += 1
                category_values = category_data[metric].dropna()
                if len(category_values) > 1:
                    percentile = self._calculate_percentile(
                        fund_row[metric],
                        category_values,
                        METRIC_DIRECTION.get(self._metric_to_key(metric), 1)
                    )
                    percentiles[metric] = percentile
                else:
                    percentiles[metric] = 50.0
            else:
                percentiles[metric] = np.nan
        
        # Check data completeness
        data_completeness = valid_metrics / len(metrics)
        if data_completeness < (1 - MISSING_DATA_THRESHOLD):
            return {
                'Symbol': symbol,
                'Name': fund_row['Name'],
                'Fund_Type': fund_type,
                'Category': category,
                'Score': np.nan,
                'Percentile': np.nan,
                'Data_Completeness': data_completeness,
                'Status': 'Insufficient data (>30% missing)'
            }
        
        # Calculate weighted score with rescaling
        score = self._calculate_weighted_score(percentiles, ACTIVE_WEIGHTS)
        percentile_rank = self._rank_fund_percentile(
            symbol, category, fund_type, score
        )
        
        return {
            'Symbol': symbol,
            'Name': fund_row['Name'],
            'Fund_Type': fund_type,
            'Category': category,
            'Score': score,
            'Percentile': percentile_rank,
            'Data_Completeness': data_completeness,
            'Status': 'Complete',
            'Expense_Ratio': fund_row['Net Expense Ratio'],
        }
    
    def _calculate_percentile(self, value, comparison_set, direction=1):
        """
        Calculate percentile rank for a value within a set
        direction: 1 for higher is better, -1 for lower is better
        """
        if pd.isna(value) or len(comparison_set) == 0:
            return 50.0
        
        if direction == -1:
            # Lower is better: invert the ranking
            rank = stats.percentileofscore(comparison_set, value, kind='rank')
            return 100 - rank
        else:
            # Higher is better
            return stats.percentileofscore(comparison_set, value, kind='rank')
    
    def _metric_to_key(self, metric_name):
        """Convert metric name to config key"""
        for key, value in CSV_COLUMNS.items():
            if value == metric_name:
                return key
        return None
    
    def _calculate_weighted_score(self, percentiles, weights):
        """Calculate weighted score, rescaling weights for missing data"""
        valid_percentiles = {}
        valid_weights = {}
        
        for metric_key, weight in weights.items():
            # Find matching metric name
            metric_name = CSV_COLUMNS.get(metric_key)
            if metric_name and not pd.isna(percentiles.get(metric_name)):
                valid_percentiles[metric_key] = percentiles[metric_name]
                valid_weights[metric_key] = weight
        
        if not valid_percentiles:
            return np.nan
        
        # Rescale weights to sum to 1
        total_weight = sum(valid_weights.values())
        rescaled_weights = {k: v / total_weight for k, v in valid_weights.items()}
        
        # Calculate weighted score
        score = sum(
            valid_percentiles[k] * rescaled_weights[k]
            for k in valid_percentiles.keys()
        )
        
        return score
    
    def _rank_fund_percentile(self, symbol, category, fund_type, score):
        """Get fund's percentile rank within its category and type"""
        if fund_type == 'Passive':
            subset = self.scores[
                (self.scores['Category'] == category) &
                (self.scores['Fund_Type'] == 'Passive')
            ]
        else:
            subset = self.scores[
                (self.scores['Category'] == category) &
                (self.scores['Fund_Type'] == 'Active')
            ]
        
        scores_in_category = subset['Score'].dropna()
        
        if len(scores_in_category) < 2 or pd.isna(score):
            return np.nan
        
        percentile = stats.percentileofscore(
            scores_in_category, score, kind='rank'
        )
        return percentile
    
    def get_fund_details(self, symbol):
        """Get detailed scoring breakdown for a fund"""
        fund_data = self.df[self.df['Symbol'] == symbol].iloc[0]
        score_data = self.scores[self.scores['Symbol'] == symbol].iloc[0]
        
        return {
            'fund_data': fund_data,
            'score_data': score_data,
        }
    
    def get_category_stats(self, category, fund_type='Active'):
        """Get statistics for all funds in a category"""
        if fund_type == 'Passive':
            category_scores = self.scores[
                (self.scores['Category'] == category) &
                (self.scores['Fund_Type'] == 'Passive')
            ]
        else:
            category_scores = self.scores[
                (self.scores['Category'] == category) &
                (self.scores['Fund_Type'] == 'Active')
            ]
        
        valid_scores = category_scores['Score'].dropna()
        
        if len(valid_scores) == 0:
            return None
        
        return {
            'avg_score': valid_scores.mean(),
            'min_score': valid_scores.min(),
            'max_score': valid_scores.max(),
            'median_score': valid_scores.median(),
            'std_score': valid_scores.std(),
            'num_funds': len(valid_scores),
        }
