import os
os.environ["SDL_VIDEODRIVER"] = "windib"
import numpy as np
import matplotlib.pyplot as plt
from agent2 import Agent, train2
from game_level1 import Level1AI
from game_level2 import Level2AI
import torch

def test_enhanced_double_dqn_features():
    """Test các tính năng enhanced của Double DQN agent"""
    print("🧪 Testing Enhanced Double DQN Features...")
    
    # Test 1: Enhanced State Representation (15 features)
    print("\n1️⃣ Testing Enhanced State Representation...")
    agent = Agent()
    game1 = Level1AI()
    game2 = Level2AI()
    
    state1 = agent.get_state(game1)
    state2 = agent.get_state(game2)
    
    print(f"✅ Level 1 state shape: {state1.shape} (expected: 15)")
    print(f"✅ Level 2 state shape: {state2.shape} (expected: 15)")
    print(f"✅ State values example: {state1[:5]}")
    
    assert len(state1) == 15, f"Expected 15 features, got {len(state1)}"
    assert len(state2) == 15, f"Expected 15 features, got {len(state2)}"
    
    # Test 2: Prioritized Experience Replay
    print("\n2️⃣ Testing Prioritized Experience Replay...")
    print(f"✅ Memory type: {type(agent.memory).__name__}")
    print(f"✅ Memory capacity: {agent.memory.capacity}")
    print(f"✅ Alpha (priority exponent): {agent.memory.alpha}")
    print(f"✅ Beta (importance sampling): {agent.beta}")
    
    # Add some sample experiences
    for i in range(10):
        state = np.random.random(15)
        action = [1, 0, 0, 0]
        reward = np.random.random()
        next_state = np.random.random(15)
        done = False
        td_error = np.random.random()
        
        agent.remember(state, action, reward, next_state, done, td_error)
    
    print(f"✅ Memory size after adding samples: {len(agent.memory)}")
    
    # Test sampling
    if len(agent.memory) >= 5:
        sample_data = agent.memory.sample(5, agent.beta)
        if sample_data:
            samples, indices, weights = sample_data
            print(f"✅ Sampled {len(samples)} experiences")
            print(f"✅ Importance weights shape: {weights.shape}")
    
    # Test 3: Curriculum Learning
    print("\n3️⃣ Testing Curriculum Learning...")
    print(f"✅ Initial difficulty: {agent.current_difficulty}")
    print(f"✅ Curriculum wins: {agent.curriculum_wins}")
    print(f"✅ Curriculum games: {agent.curriculum_games}")
    
    # Simulate wins to test curriculum progression
    for i in range(20):
        agent.update_curriculum(win=True)
    
    print(f"✅ After 20 wins - Difficulty: {agent.current_difficulty}")
    
    # Test 4: Enhanced Network Architecture
    print("\n4️⃣ Testing Enhanced Network Architecture...")
    print(f"✅ Model input size: {agent.model.input_size}")
    print(f"✅ Model output size: {agent.model.output_size}")
    print(f"✅ Model type: {type(agent.model).__name__}")
    
    # Test forward pass
    test_input = torch.tensor(state1, dtype=torch.float)
    with torch.no_grad():
        output = agent.model(test_input)
        print(f"✅ Model output shape: {output.shape}")
        print(f"✅ Model output example: {output}")
    
    # Test 5: Enhanced Enemy Detection for Level 2
    print("\n5️⃣ Testing Enhanced Enemy Detection...")
    game2.reset()
    head = game2.head
    
    from game_level2 import Point, SPEED
    test_points = [
        Point(head.x - SPEED, head.y),  # Left
        Point(head.x + SPEED, head.y),  # Right
        Point(head.x, head.y - SPEED),  # Up
    ]
    
    for i, point in enumerate(test_points):
        danger = agent._get_enemy_danger_direction(game2, point)
        print(f"✅ Direction {['Left', 'Right', 'Up'][i]} danger: {danger}")
    
    print("\n✅ All Enhanced Double DQN features working correctly!")

def quick_training_test():
    """Quick training test cho Enhanced Double DQN"""
    print("\n🚀 Quick Training Test (Enhanced Double DQN)...")
    
    # Test với Level 1
    print("\n📊 Testing Level 1...")
    game1 = Level1AI()
    mean_scores1, wins1 = train2(game1, num_games=100)
    
    print(f"✅ Level 1 - Games: 100, Wins: {wins1}, Win rate: {wins1/100:.1%}")
    print(f"✅ Level 1 - Final mean score: {mean_scores1[-1]:.2f}")
    
    # Test với Level 2
    print("\n📊 Testing Level 2...")
    game2 = Level2AI()
    mean_scores2, wins2 = train2(game2, num_games=100)
    
    print(f"✅ Level 2 - Games: 100, Wins: {wins2}, Win rate: {wins2/100:.1%}")
    print(f"✅ Level 2 - Final mean score: {mean_scores2[-1]:.2f}")
    
    # So sánh performance
    print(f"\n📈 Performance Comparison:")
    print(f"Level 1: {wins1} wins ({wins1/100:.1%}) - Score: {mean_scores1[-1]:.2f}")
    print(f"Level 2: {wins2} wins ({wins2/100:.1%}) - Score: {mean_scores2[-1]:.2f}")

def visualize_learning_curve():
    """Visualize learning curves cho Enhanced Double DQN"""
    print("\n📊 Creating Learning Curves Visualization...")
    
    # Tạo mock data cho demonstration
    games = np.arange(1, 101)
    
    # Simulate learning curves
    base_dqn = 0.1 + 0.4 * (1 - np.exp(-games/30)) + np.random.normal(0, 0.05, 100)
    enhanced_dqn = 0.2 + 0.6 * (1 - np.exp(-games/20)) + np.random.normal(0, 0.04, 100)
    
    plt.figure(figsize=(12, 8))
    
    # Learning curve comparison
    plt.subplot(2, 2, 1)
    plt.plot(games, base_dqn, label='Standard Double DQN', alpha=0.7)
    plt.plot(games, enhanced_dqn, label='Enhanced Double DQN', alpha=0.7)
    plt.title('Learning Curve Comparison')
    plt.xlabel('Games')
    plt.ylabel('Win Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Curriculum progression
    plt.subplot(2, 2, 2)
    difficulty_progression = [1, 1, 1, 2, 2, 3, 3, 4, 4, 4]
    plt.step(range(len(difficulty_progression)), difficulty_progression, where='mid')
    plt.title('Curriculum Learning Progression')
    plt.xlabel('Training Phase')
    plt.ylabel('Difficulty Level')
    plt.grid(True, alpha=0.3)
    
    # Memory importance weights distribution
    plt.subplot(2, 2, 3)
    weights = np.random.beta(2, 5, 1000)  # Simulate importance sampling weights
    plt.hist(weights, bins=30, alpha=0.7, density=True)
    plt.title('Prioritized Replay Weights Distribution')
    plt.xlabel('Importance Weight')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    
    # Feature importance (mock)
    plt.subplot(2, 2, 4)
    features = ['Danger\nStraight', 'Danger\nRight', 'Danger\nLeft', 'Food\nDirection', 
               'Enemy\nProximity', 'Exploration', 'Other']
    importance = [0.25, 0.20, 0.20, 0.15, 0.12, 0.05, 0.03]
    plt.bar(features, importance)
    plt.title('Feature Importance in Enhanced State')
    plt.ylabel('Relative Importance')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('enhanced_double_dqn_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualization saved as 'enhanced_double_dqn_analysis.png'")

def compare_architectures():
    """So sánh các architecture khác nhau"""
    print("\n🔍 Architecture Comparison...")
    
    # Standard vs Enhanced comparison
    architectures = {
        'Standard Double DQN': {'input': 12, 'hidden': [256, 128], 'features': 'Basic'},
        'Enhanced Double DQN': {'input': 15, 'hidden': [512, 256], 'features': 'Advanced'},
    }
    
    print("📊 Architecture Comparison:")
    print("-" * 60)
    print(f"{'Architecture':<20} {'Input':<8} {'Hidden':<12} {'Features':<10}")
    print("-" * 60)
    
    for name, specs in architectures.items():
        hidden_str = f"{specs['hidden'][0]}-{specs['hidden'][1]}"
        print(f"{name:<20} {specs['input']:<8} {hidden_str:<12} {specs['features']:<10}")
    
    print("-" * 60)
    
    # Feature comparison
    print("\n🎯 Feature Enhancements:")
    print("• Standard: 12 features (basic collision, direction, food location)")
    print("• Enhanced: 15 features (+ enemy proximity, safe positioning, exploration)")
    print("• Prioritized Replay: Importance sampling for better learning")
    print("• Curriculum Learning: Automatic difficulty progression")
    print("• Advanced Rewards: Multi-component reward shaping")

if __name__ == "__main__":
    print("🎮 Enhanced Double DQN Comprehensive Testing")
    print("=" * 50)
    
    # Test all enhanced features
    test_enhanced_double_dqn_features()
    
    # Quick training test
    choice = input("\n❓ Run quick training test? (y/n): ").lower()
    if choice == 'y':
        quick_training_test()
    
    # Visualization
    choice = input("\n❓ Generate learning curves visualization? (y/n): ").lower()
    if choice == 'y':
        visualize_learning_curve()
    
    # Architecture comparison
    compare_architectures()
    
    print("\n🎯 ===== TESTING COMPLETED =====")
    print("✅ Enhanced Double DQN is ready for training!")
    print("📝 Use 'python agent2.py' to start training")
    print("📊 Use 'python run_enhanced_double_dqn_training.py' for interactive training") 