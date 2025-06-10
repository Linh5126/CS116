import pygame
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from game_level1 import Level1AI
from agent import Agent
import time

def simple_train():
    """Simple training cho Level 1 với 4 enemies speed = 5"""
    
    print("🚀 Simple Training: 4 Fast Enemies (speed=5)")
    print("="*50)
    
    # Initialize
    game = Level1AI()
    agent = Agent()
    
    print(f"✅ Game: {game.w}x{game.h}")
    print(f"✅ Enemies: 4 active, all speed = 5")
    print(f"✅ Food: ({game.food_x}, {game.food_y})")
    print(f"✅ Spawn: ({game.spawnpoint_x}, {game.spawnpoint_y})")
    
    # Training settings
    max_episodes = 600  # Fewer episodes
    save_interval = 50
    target_update = 10
    
    # Metrics
    scores = deque(maxlen=100)
    wins = 0
    total_games = 0
    episode_times = []
    avg_scores = []
    win_rates = []
    
    start_time = time.time()
    best_score = -float('inf')
    
    for episode in range(max_episodes):
        episode_start = time.time()
        
        game.reset()
        state = agent.get_state(game)
        total_reward = 0
        steps = 0
        
        while True:
            # Get action
            action = agent.get_action(state)
            
            # Take step
            reward, done, score = game.play_step(action)
            next_state = agent.get_state(game)
            
            # Remember
            agent.remember(state, action, reward, next_state, done)
            
            # Train
            if len(agent.memory) > 1000:
                agent.train_long_memory()
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # Track stats
        scores.append(total_reward)
        total_games += 1
        
        is_win = score > 0
        if is_win:
            wins += 1
        
        # Calculate metrics
        if len(scores) >= 10:
            avg_score = np.mean(list(scores)[-10:])
            avg_scores.append(avg_score)
            
            if avg_score > best_score:
                best_score = avg_score
                agent.model.save('best_simple_level1.pth')
        
        win_rate = wins / total_games
        win_rates.append(win_rate)
        
        episode_time = time.time() - episode_start
        episode_times.append(episode_time)
        
        # Update target network
        if episode % target_update == 0:
            agent.update_target()
        
        # Logging
        if episode % 10 == 0:
            current_avg = np.mean(list(scores)[-10:]) if len(scores) >= 10 else total_reward
            avg_time = np.mean(episode_times[-10:])
            
            print(f"Episode {episode:4d} | "
                  f"Score: {total_reward:7.1f} | "
                  f"Avg: {current_avg:7.1f} | "
                  f"Win Rate: {win_rate:.3f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Time: {avg_time:.1f}s")
        
        # Save periodically
        if episode % save_interval == 0 and episode > 0:
            agent.model.save(f'simple_level1_ep{episode}.pth')
            print(f"💾 Saved at episode {episode}")
        
        # Early stop if excellent performance
        if len(win_rates) >= 50 and np.mean(win_rates[-20:]) >= 0.8:
            print(f"🎉 Excellent performance! Win rate: {np.mean(win_rates[-20:]):.3f}")
            break
    
    # Final results
    total_time = time.time() - start_time
    agent.model.save('final_simple_level1.pth')
    
    print("\n" + "="*50)
    print("🏁 TRAINING COMPLETED!")
    print(f"⏱️  Time: {total_time/60:.1f} minutes")
    print(f"🎯 Episodes: {episode+1}")
    print(f"🏆 Final win rate: {win_rate:.3f}")
    print(f"💰 Best avg score: {best_score:.1f}")
    print(f"⚡ Avg time/episode: {np.mean(episode_times):.1f}s")
    
    # Plot results
    plot_simple_results(avg_scores, win_rates, episode_times)
    
    return agent

def plot_simple_results(avg_scores, win_rates, episode_times):
    """Plot training results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Average scores
    if avg_scores:
        ax1.plot(avg_scores, 'b-', alpha=0.8, linewidth=2)
        ax1.set_title('Average Score (Last 10 episodes)')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Score')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Win rates
    if win_rates:
        ax2.plot(win_rates, 'g-', alpha=0.8, linewidth=2)
        ax2.set_title('Win Rate')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Win Rate')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50%')
        ax2.legend()
    
    # Episode times
    if episode_times:
        ax3.plot(episode_times, 'purple', alpha=0.7)
        ax3.set_title('Episode Duration')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Time (seconds)')
        ax3.grid(True, alpha=0.3)
    
    # Summary stats
    if avg_scores and win_rates:
        ax4.text(0.1, 0.8, f"Final Win Rate: {win_rates[-1]:.3f}", fontsize=12, transform=ax4.transAxes)
        ax4.text(0.1, 0.7, f"Best Avg Score: {max(avg_scores):.1f}", fontsize=12, transform=ax4.transAxes)
        ax4.text(0.1, 0.6, f"Episodes: {len(win_rates)}", fontsize=12, transform=ax4.transAxes)
        ax4.text(0.1, 0.5, f"Avg Time: {np.mean(episode_times):.1f}s", fontsize=12, transform=ax4.transAxes)
        ax4.set_title('Training Summary')
        ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('simple_level1_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Results saved as 'simple_level1_results.png'")

def test_simple_agent():
    """Test trained agent"""
    print("\n🧪 Testing Simple Agent...")
    
    game = Level1AI()
    agent = Agent()
    
    # Load best model
    try:
        agent.model.load('best_simple_level1.pth')
        print("✅ Loaded best model")
    except:
        try:
            agent.model.load('final_simple_level1.pth')
            print("✅ Loaded final model")
        except:
            print("❌ No model found!")
            return
    
    agent.epsilon = 0  # No exploration
    
    # Test games
    wins = 0
    total_games = 20
    total_score = 0
    
    for game_num in range(total_games):
        game.reset()
        state = agent.get_state(game)
        episode_score = 0
        
        for step in range(700):
            action = agent.get_action(state)
            reward, done, score = game.play_step(action)
            state = agent.get_state(game)
            episode_score += reward
            
            if done:
                if score > 0:
                    wins += 1
                    print(f"Game {game_num+1}: WON! (score: {episode_score:.1f})")
                else:
                    print(f"Game {game_num+1}: Lost (score: {episode_score:.1f})")
                total_score += episode_score
                break
    
    win_rate = wins / total_games
    avg_score = total_score / total_games
    
    print(f"\n📊 TEST RESULTS:")
    print(f"   Win rate: {win_rate:.3f} ({wins}/{total_games})")
    print(f"   Average score: {avg_score:.1f}")

if __name__ == "__main__":
    print("🎮 Simple Training for Level 1 (4 Fast Enemies)")
    print("1. Train new agent")
    print("2. Test existing agent")
    
    choice = input("Choose (1/2): ").strip()
    
    if choice == "1":
        simple_train()
    elif choice == "2":
        test_simple_agent()
    else:
        print("Invalid choice!") 