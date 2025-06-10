import os
os.environ["SDL_VIDEODRIVER"] = "windib"
import torch
import numpy as np
import pygame
from agent import Agent
from game_level1 import Level1AI
from game_level2 import Level2AI
import matplotlib.pyplot as plt
import time

class EnhancedAgentTester:
    def __init__(self):
        self.agent = Agent()
        print("🧪 Enhanced Agent Tester được khởi tạo")
        print(f"🎚️ Current difficulty: {self.agent.current_difficulty}")
        print(f"🧠 Model architecture: Enhanced Dueling DQN với 15 input features")
        print(f"💾 Memory type: Prioritized Experience Replay")
        
    def load_trained_model(self, model_path="model/model.pth"):
        """Load trained model nếu có"""
        try:
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.agent.model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"✅ Loaded enhanced model from {model_path}")
                else:
                    self.agent.model.load_state_dict(checkpoint)
                    print(f"✅ Loaded legacy model from {model_path}")
                self.agent.model.eval()
                return True
            else:
                print(f"⚠️ Model file {model_path} not found. Using randomly initialized model.")
                return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def test_state_representation(self, game):
        """Test enhanced state representation"""
        print("\n🔍 Testing Enhanced State Representation:")
        state = self.agent.get_state(game)
        print(f"   State shape: {state.shape}")
        print(f"   State features: {len(state)} (expected: 15)")
        print(f"   State values: {state}")
        
        # Kiểm tra từng feature
        feature_names = [
            "Danger straight", "Danger right", "Danger left",
            "Dir left", "Dir right", "Dir up", "Dir down", 
            "Food left", "Food right", "Food up", "Food down",
            "Distance to food", "Enemy left", "Enemy right", "Enemy up"
        ]
        
        print("   📊 Feature breakdown:")
        for i, (name, value) in enumerate(zip(feature_names, state)):
            print(f"      {i+1:2d}. {name:<18}: {value:.3f}")
    
    def test_curriculum_learning(self, game, test_games=20):
        """Test curriculum learning mechanism"""
        print(f"\n🎓 Testing Curriculum Learning ({test_games} games per difficulty):")
        
        results = {}
        for difficulty in range(1, 5):
            print(f"\n   🎚️ Testing Difficulty Level {difficulty}:")
            game.set_difficulty(difficulty)
            self.agent.current_difficulty = difficulty
            
            wins = 0
            scores = []
            steps_list = []
            
            for test_game in range(test_games):
                game.reset()
                game_over = False
                steps = 0
                
                while not game_over and steps < 1000:
                    state = self.agent.get_state(game)
                    action = self.agent.get_action(state)
                    reward, game_over, score = game.play_step(action)
                    steps += 1
                
                if score >= 10:
                    wins += 1
                    print(f"      Game {test_game+1:2d}: ✅ WIN  (Score: {score:2d}, Steps: {steps:3d})")
                else:
                    print(f"      Game {test_game+1:2d}: ❌ FAIL (Score: {score:2d}, Steps: {steps:3d})")
                
                scores.append(score)
                steps_list.append(steps)
            
            win_rate = wins / test_games
            avg_score = np.mean(scores)
            avg_steps = np.mean(steps_list)
            
            results[difficulty] = {
                'win_rate': win_rate,
                'avg_score': avg_score,
                'avg_steps': avg_steps,
                'wins': wins,
                'total': test_games
            }
            
            print(f"   📈 Results: Win Rate {win_rate:.1%} ({wins}/{test_games}), Avg Score: {avg_score:.1f}, Avg Steps: {avg_steps:.0f}")
        
        return results
    
    def test_prioritized_replay(self):
        """Test prioritized experience replay functionality"""
        print(f"\n🔄 Testing Prioritized Experience Replay:")
        print(f"   Memory type: {type(self.agent.memory).__name__}")
        print(f"   Memory size: {len(self.agent.memory)}")
        print(f"   Beta (importance sampling): {self.agent.beta:.3f}")
        
        # Test memory operations
        if len(self.agent.memory) > 0:
            try:
                sample = self.agent.memory.sample(min(10, len(self.agent.memory)), self.agent.beta)
                if sample:
                    samples, indices, weights = sample
                    print(f"   ✅ Prioritized sampling working: {len(samples)} samples")
                    print(f"   Importance weights range: {weights.min():.3f} - {weights.max():.3f}")
                else:
                    print(f"   ⚠️ Memory too small for sampling")
            except Exception as e:
                print(f"   ❌ Error in prioritized sampling: {e}")
        else:
            print(f"   ℹ️ Memory is empty (expected for fresh agent)")
    
    def visualize_performance(self, results):
        """Visualize curriculum learning results"""
        if not results:
            return
            
        difficulties = list(results.keys())
        win_rates = [results[d]['win_rate'] for d in difficulties]
        avg_scores = [results[d]['avg_score'] for d in difficulties]
        avg_steps = [results[d]['avg_steps'] for d in difficulties]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Win rates
        ax1.bar(difficulties, win_rates, color='green', alpha=0.7)
        ax1.set_title('Win Rate by Difficulty')
        ax1.set_xlabel('Difficulty Level')
        ax1.set_ylabel('Win Rate')
        ax1.set_ylim(0, 1)
        for i, v in enumerate(win_rates):
            ax1.text(difficulties[i], v + 0.02, f'{v:.1%}', ha='center')
        
        # Average scores
        ax2.bar(difficulties, avg_scores, color='blue', alpha=0.7)
        ax2.set_title('Average Score by Difficulty')
        ax2.set_xlabel('Difficulty Level')
        ax2.set_ylabel('Average Score')
        for i, v in enumerate(avg_scores):
            ax2.text(difficulties[i], v + 0.2, f'{v:.1f}', ha='center')
        
        # Average steps
        ax3.bar(difficulties, avg_steps, color='orange', alpha=0.7)
        ax3.set_title('Average Steps by Difficulty')
        ax3.set_xlabel('Difficulty Level')
        ax3.set_ylabel('Average Steps')
        for i, v in enumerate(avg_steps):
            ax3.text(difficulties[i], v + 10, f'{v:.0f}', ha='center')
        
        # Summary table
        ax4.axis('tight')
        ax4.axis('off')
        
        table_data = []
        for d in difficulties:
            r = results[d]
            table_data.append([
                f"Level {d}",
                f"{r['win_rate']:.1%}",
                f"{r['avg_score']:.1f}",
                f"{r['avg_steps']:.0f}"
            ])
        
        table = ax4.table(cellText=table_data,
                         colLabels=['Difficulty', 'Win Rate', 'Avg Score', 'Avg Steps'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax4.set_title('Performance Summary', pad=20)
        
        plt.tight_layout()
        save_path = 'enhanced_agent_test_results.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Performance chart saved to {save_path}")
        plt.show()
    
    def interactive_demo(self, game, difficulty=2):
        """Interactive demo để xem agent chơi"""
        print(f"\n🎮 Interactive Demo (Difficulty {difficulty}):")
        print("Press SPACE to pause/resume, ESC to quit, 1-4 to change difficulty")
        
        game.set_difficulty(difficulty)
        self.agent.current_difficulty = difficulty
        game.reset()
        
        game_over = False
        steps = 0
        paused = False
        clock = pygame.time.Clock()
        
        while not game_over and steps < 2000:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        print(f"{'⏸️ Paused' if paused else '▶️ Resumed'}")
                    elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4]:
                        new_diff = int(event.unicode)
                        if 1 <= new_diff <= 4:
                            difficulty = new_diff
                            game.set_difficulty(difficulty)
                            self.agent.current_difficulty = difficulty
                            print(f"🎚️ Changed to difficulty {difficulty}")
            
            if not paused:
                state = self.agent.get_state(game)
                action = self.agent.get_action(state)
                reward, game_over, score = game.play_step(action)
                steps += 1
                
                # Print real-time info
                if steps % 50 == 0:
                    print(f"   Step {steps}: Score {score}, Reward {reward:.2f}")
            
            clock.tick(20)  # Limit FPS for better viewing
        
        result = "WIN" if score >= 10 else "FAIL"
        print(f"🏁 Demo ended: {result} (Score: {score}, Steps: {steps})")

def main():
    print("🚀 Enhanced DQN Agent Testing Suite")
    print("="*50)
    
    # Initialize tester
    tester = EnhancedAgentTester()
    
    # Load trained model if available
    tester.load_trained_model()
    
    # Initialize game
    game = Level1AI()
    print(f"🎮 Initialized {type(game).__name__}")
    
    # Test enhanced state representation
    tester.test_state_representation(game)
    
    # Test prioritized replay
    tester.test_prioritized_replay()
    
    # Test curriculum learning
    print("\n" + "="*50)
    results = tester.test_curriculum_learning(game, test_games=10)
    
    # Visualize results
    tester.visualize_performance(results)
    
    # Interactive demo
    print("\n" + "="*50)
    demo_choice = input("🎮 Run interactive demo? (y/n): ").lower().strip()
    if demo_choice == 'y':
        try:
            difficulty = int(input("Choose difficulty (1-4): ") or "2")
            difficulty = max(1, min(4, difficulty))
        except:
            difficulty = 2
        
        tester.interactive_demo(game, difficulty)
    
    print("\n🎊 Testing completed!")
    pygame.quit()

if __name__ == "__main__":
    main() 