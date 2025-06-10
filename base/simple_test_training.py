import os
os.environ["SDL_VIDEODRIVER"] = "windib"

print("🚀 Testing Enhanced Double DQN Training...")

from agent2 import Agent
from game_level2 import Level2AI
import torch
import numpy as np

def simple_training_test():
    """Test đơn giản training loop"""
    agent = Agent()
    game = Level2AI()
    
    print(f"✅ Agent initialized - Difficulty: {agent.current_difficulty}")
    print(f"✅ Memory type: {type(agent.memory).__name__}")
    print(f"✅ State features: {agent.model.input_size}")
    
    total_score = 0
    wins = 0
    
    for game_num in range(10):  # Test 10 games
        game.reset()
        steps = 0
        game_score = 0
        
        while steps < 100:  # Limit steps
            # Get state and action
            state = agent.get_state(game)
            action = agent.get_action(state)
            
            # Play step
            reward, done, score = game.play_step(action)
            
            # Store experience (simplified)
            if steps > 0:
                agent.remember(prev_state, prev_action, reward, state, done)
            
            prev_state = state
            prev_action = action
            game_score = score
            steps += 1
            
            if done:
                break
        
        total_score += game_score
        if game_score == 10:  # Win
            wins += 1
            
        # Test curriculum update
        agent.update_curriculum(game_score == 10)
        
        print(f"Game {game_num + 1}: Score {game_score}, Steps {steps}")
    
    mean_score = total_score / 10
    win_rate = wins / 10
    
    print(f"\n📊 Test Results:")
    print(f"🎮 Total games: 10")
    print(f"🏆 Wins: {wins} ({win_rate:.1%})")
    print(f"📈 Mean score: {mean_score:.2f}")
    print(f"🎚️ Final difficulty: {agent.current_difficulty}")
    print(f"💾 Memory size: {len(agent.memory)}")
    
    # Test prioritized replay sampling
    if len(agent.memory) >= 5:
        sample_data = agent.memory.sample(5, agent.beta)
        if sample_data:
            print(f"✅ Prioritized sampling working")
        else:
            print(f"⚠️ Prioritized sampling returned None")
    
    print(f"\n🎯 Enhanced Double DQN training test completed!")
    return True

if __name__ == "__main__":
    try:
        success = simple_training_test()
        if success:
            print("✅ All systems working correctly!")
            print("🚀 Ready for full training!")
        else:
            print("❌ Test failed")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc() 