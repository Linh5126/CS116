import os
os.environ["SDL_VIDEODRIVER"] = "windib"

print("🧪 Testing Enhanced Double DQN...")

try:
    from agent2 import Agent
    from game_level1 import Level1AI
    from game_level2 import Level2AI
    
    print("✅ Imports successful")
    
    # Test agent creation
    agent = Agent()
    print(f"✅ Agent created with {agent.model.input_size} input features")
    
    # Test Level 1
    game1 = Level1AI()
    state1 = agent.get_state(game1)
    print(f"✅ Level 1 state shape: {state1.shape}")
    
    # Test Level 2
    game2 = Level2AI()
    state2 = agent.get_state(game2)
    print(f"✅ Level 2 state shape: {state2.shape}")
    
    # Test memory type
    print(f"✅ Memory type: {type(agent.memory).__name__}")
    print(f"✅ Memory capacity: {agent.memory.capacity}")
    
    # Test curriculum
    print(f"✅ Initial difficulty: {agent.current_difficulty}")
    
    print("\n🎯 Enhanced Double DQN is working correctly!")
    print("🚀 Ready for training với Level 1 và Level 2!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc() 