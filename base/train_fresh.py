#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FRESH START TRAINING - Train từ đầu với fixed code
"""

import os
import sys
sys.path.append('.')

from advanced_trainer import TrainingManager

def train_fresh():
    """Train fresh từ đầu với tất cả fixes"""
    
    print("🚀 FRESH START TRAINING")
    print("=" * 50)
    print("✅ Simple safe exploration (no stuck)")
    print("✅ Fixed epsilon decay (no catastrophic forgetting)")
    print("✅ Balanced training parameters")
    print("=" * 50)
    
    # Create training manager
    manager = TrainingManager()
    
    # Start fresh training với reasonable parameters
    results = manager.train_single_level(
        level_name='Level1',
        algorithm='DQN',
        total_episodes=3000,    # Moderate length
        eval_interval=300,      # Frequent evaluation để monitor
        eval_games=50          # Good sample size
    )
    
    if results and results['evaluation_results']:
        final_eval = results['evaluation_results'][-1]
        final_win_rate = final_eval['win_rate']
        
        print(f"\n🎉 FRESH TRAINING COMPLETED!")
        print(f"📊 Final win rate: {final_win_rate:.1%}")
        
        if final_win_rate >= 0.3:
            print(f"🏆 EXCELLENT! Ready for report")
        elif final_win_rate >= 0.15:
            print(f"✅ GOOD! Consider more training") 
        else:
            print(f"📈 PROGRESS! Check parameters")
    
    return results

if __name__ == "__main__":
    train_fresh()
