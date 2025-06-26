#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UPDATE PLOT - Vẽ lại biểu đồ với data đầy đủ đến episode 16000
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

def load_results():
    """Load training results"""
    results_file = "training_results/Double_DQN_Level1_results.json"
    
    if not os.path.exists(results_file):
        print(f"❌ Results file not found: {results_file}")
        return None
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print(f"📊 Loaded results:")
    print(f"   Episode scores: {len(results.get('episode_scores', []))}")
    print(f"   Evaluations: {len(results.get('evaluation_results', []))}")
    print(f"   Final episode: {results.get('final_episode', 'N/A')}")
    print(f"   Best win rate: {results.get('best_win_rate', 0):.2%}")
    
    return results

def plot_training_results(results):
    """Vẽ biểu đồ training results"""
    
    algorithm = results['algorithm']
    level = results['level']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{algorithm} - {level} Training Results (Updated)', fontsize=16)
    
    # 1. Episode scores với moving average
    episode_scores = results.get('episode_scores', [])
    if episode_scores:
        episodes = range(len(episode_scores))
        ax1.plot(episodes, episode_scores, alpha=0.3, color='lightblue', label='Episode Scores')
        ax1.set_title('Episode Scores')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Score')
        
        # Moving average
        window = 100
        if len(episode_scores) >= window:
            moving_avg = np.convolve(episode_scores, np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(episode_scores)), moving_avg, 'r-', linewidth=2, label='Moving Average (100)')
            ax1.legend()
    else:
        ax1.text(0.5, 0.5, 'No episode scores data', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Episode Scores')
    
    # 2. Win Rate Over Time
    eval_results = results.get('evaluation_results', [])
    if eval_results:
        eval_episodes = [r['episode'] for r in eval_results]
        win_rates = [r['win_rate'] for r in eval_results]
        
        ax2.plot(eval_episodes, win_rates, 'g-o', linewidth=2, markersize=4)
        ax2.set_title('Win Rate Over Time')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Win Rate')
        ax2.set_ylim(0, max(win_rates) * 1.1 if win_rates else 1)
        ax2.grid(True, alpha=0.3)
        
        # Highlight best point
        best_idx = win_rates.index(max(win_rates))
        ax2.plot(eval_episodes[best_idx], win_rates[best_idx], 'r*', markersize=15, label=f'Best: {max(win_rates):.1%}')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No evaluation data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Win Rate Over Time')
    
    # 3. Average Score Over Time
    if eval_results:
        avg_scores = [r['avg_score'] for r in eval_results]
        
        ax3.plot(eval_episodes, avg_scores, 'b-o', linewidth=2, markersize=4)
        ax3.set_title('Average Score Over Time')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Average Score')
        ax3.grid(True, alpha=0.3)
        
        # Add target line (win threshold = 6)
        ax3.axhline(y=6, color='red', linestyle='--', alpha=0.7, label='Win Threshold (6)')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'No evaluation data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Average Score Over Time')
    
    # 4. Steps to Win (when won)
    if eval_results:
        steps_to_win = []
        episodes_with_wins = []
        for r in eval_results:
            if r.get('avg_steps_when_won', 0) > 0:
                steps_to_win.append(r['avg_steps_when_won'])
                episodes_with_wins.append(r['episode'])
        
        if steps_to_win:
            ax4.plot(episodes_with_wins, steps_to_win, 'purple', marker='o', linewidth=2, markersize=4)
            ax4.set_title('Average Steps to Win (When Won)')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Steps to Win')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No wins recorded', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Average Steps to Win (When Won)')
    else:
        ax4.text(0.5, 0.5, 'No evaluation data', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Average Steps to Win (When Won)')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = f"plots/{algorithm}_{level}_training.png"
    os.makedirs("plots", exist_ok=True)
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_file

def main():
    print("📈 UPDATE TRAINING PLOT")
    print("=" * 50)
    print("✅ Vẽ lại biểu đồ với data đầy đủ")
    print("✅ Bao gồm evaluation tại episode 16000 (32% win rate)")
    print("=" * 50)
    
    # Load results
    results = load_results()
    if not results:
        return
    
    # Generate updated plot
    plot_file = plot_training_results(results)
    
    print(f"\n🎉 PLOT UPDATED SUCCESSFULLY!")
    print(f"📊 Saved to: {plot_file}")
    
    # Show summary
    eval_results = results.get('evaluation_results', [])
    if eval_results:
        print(f"\n📈 Training Summary:")
        print(f"   Episodes trained: 0 → {results.get('final_episode', 'N/A')}")
        print(f"   Evaluations: {len(eval_results)}")
        print(f"   Best win rate: {results.get('best_win_rate', 0):.2%} at episode {results.get('best_model_episode', 'N/A')}")
        
        # Latest evaluation
        latest = eval_results[-1]
        print(f"   Latest evaluation: {latest['win_rate']:.2%} win rate at episode {latest['episode']}")
        print(f"   Latest avg score: {latest['avg_score']:.1f}/6")
    
    print(f"\n🎯 PLOT FEATURES:")
    print(f"✅ Episode scores với moving average")
    print(f"✅ Win rate progression (highlight best point)")
    print(f"✅ Average score progression")
    print(f"✅ Steps to win analysis")
    print(f"✅ Data range: 0 → 16000 episodes")

if __name__ == "__main__":
    main() 