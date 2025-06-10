import pygame
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from game_level1 import Level1AI
from agent import DQNAgent
import time
import os

def fast_train():
    """Training script tối ưu cho Level 1 - train nhanh hơn"""
    
    # OPTIMIZED HYPERPARAMETERS cho training nhanh
    config = {
        'learning_rate': 0.001,  # Tăng LR để học nhanh hơn
        'gamma': 0.95,           # Discount factor vừa phải
        'epsilon_start': 1.0,
        'epsilon_end': 0.05,     # Không quá thấp 
        'epsilon_decay': 0.998,  # Decay nhanh hơn
        'memory_size': 50000,    # Memory vừa đủ
        'batch_size': 64,        # Batch size tối ưu
        'target_update': 10,     # Update target network thường xuyên
        'curriculum_threshold': 0.7,  # Win rate để tăng difficulty
        'max_episodes': 800,     # Ít episodes hơn
        'save_interval': 50,     # Save thường xuyên
        'early_stop_threshold': 0.85,  # Stop sớm khi đạt performance tốt
    }
    
    print("🚀 Starting FAST Training for Level 1!")
    print(f"📊 Config: {config}")
    
    # Initialize game và agent
    game = Level1AI()
    agent = DQNAgent(state_size=15, action_size=4, config=config)
    
    # Training metrics
    scores = deque(maxlen=100)
    win_rates = deque(maxlen=50)
    avg_scores = []
    avg_win_rates = []
    episode_times = []
    
    # Curriculum learning tracking
    curriculum_wins = 0
    curriculum_games = 0
    difficulty_progression = []
    
    start_time = time.time()
    best_avg_score = -float('inf')
    
    for episode in range(config['max_episodes']):
        episode_start = time.time()
        
        # Reset game
        game.reset()
        state = agent.get_state(game)
        total_reward = 0
        steps = 0
        
        while True:
            # Get action
            action = agent.get_action(state)
            
            # Take action
            reward, done, score = game.play_step(action)
            next_state = agent.get_state(game)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train agent
            if len(agent.memory) > config['batch_size']:
                agent.replay(config['batch_size'])
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # Track metrics
        scores.append(total_reward)
        is_win = score > 0  # Win if reached food
        
        # Curriculum learning
        curriculum_games += 1
        if is_win:
            curriculum_wins += 1
        
        # Update curriculum every 20 games
        if curriculum_games >= 20:
            win_rate = curriculum_wins / curriculum_games
            win_rates.append(win_rate)
            
            # Tăng difficulty nếu win rate cao
            if win_rate >= config['curriculum_threshold'] and game.difficulty_level < game.max_difficulty:
                game.difficulty_level += 1
                agent.update_curriculum()  # Reset epsilon một phần khi tăng difficulty
                print(f"🎯 Difficulty increased to {game.difficulty_level}! Win rate: {win_rate:.2f}")
            
            curriculum_wins = 0
            curriculum_games = 0
        
        difficulty_progression.append(game.difficulty_level)
        
        # Calculate running averages
        if len(scores) >= 10:
            avg_score = np.mean(list(scores)[-10:])
            avg_scores.append(avg_score)
            
            if len(win_rates) > 0:
                avg_win_rate = np.mean(list(win_rates)[-5:])
                avg_win_rates.append(avg_win_rate)
            else:
                avg_win_rates.append(0)
        
        episode_time = time.time() - episode_start
        episode_times.append(episode_time)
        
        # Update target network
        if episode % config['target_update'] == 0:
            agent.update_target_network()
        
        # Logging
        if episode % 10 == 0:
            avg_score = np.mean(list(scores)[-10:]) if len(scores) >= 10 else np.mean(scores)
            avg_time = np.mean(episode_times[-10:])
            current_win_rate = win_rates[-1] if win_rates else 0
            
            print(f"Episode {episode:4d} | "
                  f"Score: {total_reward:7.1f} | "
                  f"Avg Score: {avg_score:7.1f} | "
                  f"Win Rate: {current_win_rate:.2f} | "
                  f"Difficulty: {game.difficulty_level} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Time: {avg_time:.1f}s")
        
        # Save model
        if episode % config['save_interval'] == 0 and episode > 0:
            agent.save(f'fast_level1_episode_{episode}.pth')
            print(f"💾 Model saved at episode {episode}")
        
        # Early stopping
        if len(avg_scores) >= 50:
            recent_avg = np.mean(avg_scores[-20:])
            if recent_avg > best_avg_score:
                best_avg_score = recent_avg
                agent.save('fast_level1_best.pth')
            
            # Stop if performance is very good
            if (len(win_rates) >= 10 and 
                np.mean(list(win_rates)[-5:]) >= config['early_stop_threshold'] and
                game.difficulty_level >= 3):
                print(f"🎉 Early stopping! Excellent performance achieved!")
                print(f"Win rate: {np.mean(list(win_rates)[-5:]):.2f} at difficulty {game.difficulty_level}")
                break
    
    total_time = time.time() - start_time
    
    # Final save
    agent.save('fast_level1_final.pth')
    
    # Results summary
    print("\n" + "="*60)
    print("🏁 FAST TRAINING COMPLETED!")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"🎯 Episodes completed: {episode+1}")
    print(f"🏆 Final difficulty: {game.difficulty_level}")
    if win_rates:
        print(f"📈 Final win rate: {win_rates[-1]:.2f}")
    if avg_scores:
        print(f"💰 Best average score: {best_avg_score:.1f}")
    print(f"⚡ Average episode time: {np.mean(episode_times):.1f}s")
    
    # Plot results
    plot_training_results(avg_scores, avg_win_rates, difficulty_progression, episode_times)
    
    return agent

def plot_training_results(avg_scores, avg_win_rates, difficulty_progression, episode_times):
    """Vẽ biểu đồ kết quả training"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Average scores
    if avg_scores:
        ax1.plot(avg_scores, 'b-', alpha=0.7)
        ax1.set_title('Average Score (Last 10 episodes)')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Score')
        ax1.grid(True, alpha=0.3)
    
    # Win rates
    if avg_win_rates:
        ax2.plot(avg_win_rates, 'g-', alpha=0.7)
        ax2.set_title('Win Rate')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Win Rate')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
    
    # Difficulty progression
    ax3.plot(difficulty_progression, 'r-', alpha=0.7)
    ax3.set_title('Difficulty Progression')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Difficulty Level')
    ax3.set_ylim(0.5, 4.5)
    ax3.grid(True, alpha=0.3)
    
    # Episode times
    if episode_times:
        ax4.plot(episode_times, 'purple', alpha=0.7)
        ax4.set_title('Episode Duration')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Time (seconds)')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fast_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Training plots saved as 'fast_training_results.png'")

def test_trained_agent():
    """Test agent đã train"""
    print("\n🧪 Testing trained agent...")
    
    game = Level1AI()
    agent = DQNAgent(state_size=15, action_size=4)
    
    # Load best model
    try:
        agent.load('fast_level1_best.pth')
        print("✅ Loaded best model")
    except:
        try:
            agent.load('fast_level1_final.pth')
            print("✅ Loaded final model")
        except:
            print("❌ No trained model found!")
            return
    
    agent.epsilon = 0  # No exploration cho testing
    
    # Test ở tất cả difficulty levels
    for difficulty in range(1, 5):
        wins = 0
        total_games = 20
        total_score = 0
        
        print(f"\n🎯 Testing at Difficulty {difficulty}...")
        
        for game_num in range(total_games):
            game.reset()
            game.set_difficulty(difficulty)
            state = agent.get_state(game)
            episode_score = 0
            
            for step in range(800):  # Max steps
                action = agent.get_action(state)
                reward, done, score = game.play_step(action)
                state = agent.get_state(game)
                episode_score += reward
                
                if done:
                    if score > 0:  # Won
                        wins += 1
                    total_score += episode_score
                    break
        
        win_rate = wins / total_games
        avg_score = total_score / total_games
        
        print(f"   Win rate: {win_rate:.2f} ({wins}/{total_games})")
        print(f"   Average score: {avg_score:.1f}")

if __name__ == "__main__":
    print("🚀 Fast Training Script for Level 1")
    print("Choose option:")
    print("1. Train new agent")
    print("2. Test existing agent")
    
    choice = input("Enter choice (1/2): ").strip()
    
    if choice == "1":
        agent = fast_train()
    elif choice == "2":
        test_trained_agent()
    else:
        print("Invalid choice!") 