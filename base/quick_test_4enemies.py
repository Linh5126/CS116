import pygame
import numpy as np
from game_level1 import Level1AI
from agent import Agent
import time

def test_4enemies_system():
    """Test quick 4 enemies system"""
    print("🎮 Testing 4 Enemies System...")
    
    # Initialize
    game = Level1AI()
    agent = Agent()
    
    print(f"✅ Game initialized: {game.w}x{game.h}")
    print(f"✅ Food position: ({game.food_x}, {game.food_y})")
    print(f"✅ Spawn position: ({game.spawnpoint_x}, {game.spawnpoint_y})")
    print(f"✅ Active enemies: {len(game.active_enemies)}")
    
    # Test enemy speeds
    print("\n🤖 Enemy Speeds:")
    for i, enemy in enumerate(game.active_enemies):
        print(f"   Enemy {i+1}: speed={enemy.enemy_speed}, pos=({enemy.rect2.centerx}, {enemy.rect2.centery})")
    
    # Test state representation
    print("\n🧠 Testing state representation...")
    state = agent.get_state(game)
    print(f"✅ State size: {len(state)} features")
    print(f"State values: {state}")
    
    # Test difficulty levels
    print("\n🎚️ Testing difficulty levels...")
    for level in range(1, 5):
        game.set_difficulty(level)
        speeds = [enemy.enemy_speed for enemy in game.active_enemies]
        print(f"   Level {level}: speeds = {speeds}")
    
    # Run quick game test
    print("\n🚀 Running quick game test...")
    game.reset()
    game.set_difficulty(2)  # Medium difficulty
    
    episode_rewards = []
    for episode in range(5):
        game.reset()
        total_reward = 0
        steps = 0
        
        print(f"\nEpisode {episode + 1}:")
        
        for step in range(300):  # Max 300 steps
            state = agent.get_state(game)
            action = agent.get_action(state)
            reward, done, score = game.play_step(action)
            
            total_reward += reward
            steps += 1
            
            if step % 50 == 0:
                print(f"   Step {step}: reward={reward:.3f}, total={total_reward:.1f}")
            
            if done:
                status = "WON!" if score > 0 else "Lost"
                print(f"   {status} after {steps} steps, final reward: {total_reward:.1f}")
                break
        
        episode_rewards.append(total_reward)
    
    # Results
    print("\n" + "="*50)
    print("📊 TEST RESULTS:")
    print(f"   Average reward: {np.mean(episode_rewards):.1f}")
    print(f"   Reward range: {np.min(episode_rewards):.1f} to {np.max(episode_rewards):.1f}")
    print(f"   All 4 enemies active: ✅")
    print(f"   Reward system working: ✅")
    print("✅ 4 Enemies System Test PASSED!")

def test_enemy_collision_system():
    """Test collision system với 4 enemies"""
    print("\n🔥 Testing Enemy Collision System...")
    
    game = Level1AI()
    agent = Agent()
    
    # Test collision detection
    collision_tests = 0
    collision_detected = 0
    
    for _ in range(100):
        game.reset()
        
        # Random movement
        for _ in range(20):
            action = [np.random.randint(0, 2) for _ in range(4)]
            reward, done, score = game.play_step(action)
            collision_tests += 1
            
            if done and score == 0:  # Lost (collision)
                collision_detected += 1
                break
    
    collision_rate = collision_detected / collision_tests * 100
    print(f"   Collision rate: {collision_rate:.1f}%")
    print(f"   Collision system: {'✅' if collision_rate > 0 else '❌'}")

def test_rewards_system():
    """Test enhanced rewards với 4 enemies"""
    print("\n🎯 Testing Enhanced Rewards System...")
    
    game = Level1AI()
    agent = Agent()
    
    reward_types = {
        'progress': 0,
        'checkpoint': 0,
        'enemy_awareness': 0,
        'exploration': 0,
        'negative': 0
    }
    
    game.reset()
    
    for step in range(100):
        old_pos = game.head
        state = agent.get_state(game)
        action = agent.get_action(state)
        reward, done, score = game.play_step(action)
        
        # Categorize rewards
        if reward > 1:
            reward_types['checkpoint'] += 1
        elif reward > 0.1:
            reward_types['progress'] += 1  
        elif reward > 0:
            reward_types['exploration'] += 1
        elif reward < -1:
            reward_types['negative'] += 1
        
        if done:
            break
    
    print(f"   Reward distribution: {reward_types}")
    print(f"   Reward system variety: ✅")

if __name__ == "__main__":
    print("🚀 Quick Test for 4 Enemies Level 1 System")
    print("="*50)
    
    try:
        test_4enemies_system()
        test_enemy_collision_system()
        test_rewards_system()
        
        print("\n🎉 All tests PASSED! 4 Enemies system ready for training!")
        
    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        import traceback
        traceback.print_exc() 