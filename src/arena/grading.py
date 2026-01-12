"""
Grading logic for Arena evaluation.

This module handles saving and loading evaluation results from the Arena
interface, tracking user votes and statistics.
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import ArenaConfig


class ArenaGrader:
    """
    Manages grading results from Arena evaluations.
    
    Tracks user votes comparing baseline vs challenger models and
    saves statistics to CSV.
    """
    
    def __init__(self, results_path: Optional[Path] = None):
        """
        Initialize the grader.
        
        Args:
            results_path: Path to results CSV file. If None, uses config default.
        """
        self.results_path = results_path or ArenaConfig.RESULTS_CSV
        
        # Initialize results file if it doesn't exist
        if not self.results_path.exists():
            self._initialize_results_file()
    
    def _initialize_results_file(self):
        """Create results CSV with headers."""
        with open(self.results_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "query_image",
                "baseline_method",
                "challenger_method",
                "user_choice",
                "baseline_is_row_a"
            ])
    
    def save_vote(
        self,
        query_image: str,
        baseline_method: str,
        challenger_method: str,
        user_choice: str,
        baseline_is_row_a: bool
    ):
        """
        Save a user vote to the results file.
        
        Args:
            query_image: Path or name of the query image
            baseline_method: Name of baseline method (e.g., "Base CLIP")
            challenger_method: Name of challenger method (e.g., "Fine-Tuned CLIP")
            user_choice: User's choice ("A" or "B")
            baseline_is_row_a: Whether baseline was shown in row A
        """
        timestamp = datetime.now().isoformat()
        
        with open(self.results_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                query_image,
                baseline_method,
                challenger_method,
                user_choice,
                baseline_is_row_a
            ])
    
    def get_statistics(self) -> dict:
        """
        Get aggregated statistics from all votes.
        
        Returns:
            Dictionary with statistics
        """
        if not self.results_path.exists():
            return {
                "total_votes": 0,
                "baseline_wins": 0,
                "challenger_wins": 0,
                "baseline_win_rate": 0.0,
                "challenger_win_rate": 0.0
            }
        
        df = pd.read_csv(self.results_path)
        
        if len(df) == 0:
            return {
                "total_votes": 0,
                "baseline_wins": 0,
                "challenger_wins": 0,
                "baseline_win_rate": 0.0,
                "challenger_win_rate": 0.0
            }
        
        # Determine which method won each vote
        def get_winner(row):
            if row["baseline_is_row_a"]:
                baseline_row = "A"
            else:
                baseline_row = "B"
            
            if row["user_choice"] == baseline_row:
                return "baseline"
            else:
                return "challenger"
        
        df["winner"] = df.apply(get_winner, axis=1)
        
        total_votes = len(df)
        baseline_wins = (df["winner"] == "baseline").sum()
        challenger_wins = (df["winner"] == "challenger").sum()
        
        return {
            "total_votes": total_votes,
            "baseline_wins": baseline_wins,
            "challenger_wins": challenger_wins,
            "baseline_win_rate": baseline_wins / total_votes if total_votes > 0 else 0.0,
            "challenger_win_rate": challenger_wins / total_votes if total_votes > 0 else 0.0
        }
    
    def print_statistics(self):
        """Print current statistics to console."""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("Arena Statistics")
        print("="*60)
        print(f"Total Votes: {stats['total_votes']}")
        print(f"Baseline Wins: {stats['baseline_wins']} ({stats['baseline_win_rate']*100:.1f}%)")
        print(f"Challenger Wins: {stats['challenger_wins']} ({stats['challenger_win_rate']*100:.1f}%)")
        print("="*60 + "\n")

