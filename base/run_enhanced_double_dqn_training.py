#!/usr/bin/env python3
"""
🎮 Enhanced Double DQN Interactive Training Script
Hỗ trợ cả Level 1 và Level 2 với tất cả enhanced features
"""

import os
os.environ["SDL_VIDEODRIVER"] = "windib"
import argparse
import sys
from agent2 import train2, Agent
from game_level1 import Level1AI
from game_level2 import Level2AI

def print_banner():
    """In banner cho Enhanced Double DQN"""
    print("=" * 70)
    print("🎮 ENHANCED DOUBLE DQN TRAINING - WORLD'S HARDEST GAME 🎮")
    print("=" * 70)
    print("🚀 Features: Prioritized Replay + Curriculum Learning + Enhanced State")
    print("🎯 Supports: Level 1 (4 enemies) & Level 2 (12 enemies)")
    print("🧠 Architecture: Dueling DQN với LayerNorm + Advanced Rewards")
    print("=" * 70)

def print_enhanced_features():
    """In chi tiết các enhanced features"""
    print("\n🔥 ENHANCED FEATURES:")
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│ 🧠 Enhanced State Representation (15 features)              │")
    print("│    • Collision detection (3 directions)                    │")
    print("│    • Movement direction (4 directions)                     │")
    print("│    • Food location (4 directions)                          │")
    print("│    • Normalized distance to food                           │")
    print("│    • Enemy danger detection (3 directions)                 │")
    print("│                                                             │")
    print("│ 🎯 Prioritized Experience Replay                           │")
    print("│    • TD-error based sampling priorities                    │")
    print("│    • Importance sampling with dynamic beta                 │")
    print("│    • Better sample efficiency                              │")
    print("│                                                             │")
    print("│ 📚 Automatic Curriculum Learning                           │")
    print("│    • Auto-adjusts difficulty (1→4) based on win rate      │")
    print("│    • Smart epsilon management                              │")
    print("│    • Progressive challenge system                          │")
    print("│                                                             │")
    print("│ 🎁 Advanced Reward Shaping                                 │")
    print("│    • Multi-component reward system                        │")
    print("│    • Anti-oscillation penalties                           │")
    print("│    • Strategic positioning bonuses                        │")
    print("│    • Enemy danger awareness rewards                       │")
    print("└─────────────────────────────────────────────────────────────┘")

def get_training_config():
    """Lấy configuration từ user"""
    print("\n🎯 TRAINING CONFIGURATION")
    print("-" * 30)
    
    # Chọn level
    print("📍 Select Level:")
    print("1. Level 1 (4 enemies, easier)")
    print("2. Level 2 (12 enemies, harder)")
    
    while True:
        try:
            level_choice = int(input("Choose level (1/2): "))
            if level_choice in [1, 2]:
                break
            else:
                print("❌ Invalid choice. Please enter 1 or 2.")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
    
    # Số games
    print("\n🎮 Number of Games:")
    print("1. Quick test (100 games)")
    print("2. Standard training (1000 games)")
    print("3. Extended training (2000 games)")
    print("4. Custom")
    
    while True:
        try:
            games_choice = int(input("Choose option (1/2/3/4): "))
            if games_choice == 1:
                num_games = 100
                break
            elif games_choice == 2:
                num_games = 1000
                break
            elif games_choice == 3:
                num_games = 2000
                break
            elif games_choice == 4:
                num_games = int(input("Enter custom number of games: "))
                break
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
    
    # Training mode
    print("\n🔧 Training Mode:")
    print("1. Fresh training (reset all progress)")
    print("2. Continue training (load saved state)")
    
    while True:
        try:
            mode_choice = int(input("Choose mode (1/2): "))
            if mode_choice in [1, 2]:
                fresh_training = (mode_choice == 1)
                break
            else:
                print("❌ Invalid choice. Please enter 1 or 2.")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
    
    return level_choice, num_games, fresh_training

def show_training_summary(level_choice, num_games, fresh_training):
    """Hiển thị tóm tắt training config"""
    level_name = "Level 1 (4 enemies)" if level_choice == 1 else "Level 2 (12 enemies)"
    mode_name = "Fresh Training" if fresh_training else "Continue Training"
    
    print(f"\n📋 TRAINING SUMMARY:")
    print(f"┌─────────────────────────────────────┐")
    print(f"│ Level: {level_name:<25} │")
    print(f"│ Games: {num_games:<25} │")
    print(f"│ Mode:  {mode_name:<25} │")
    print(f"│ Enhanced Features: ✅ Enabled       │")
    print(f"└─────────────────────────────────────┘")

def prepare_training(level_choice, fresh_training):
    """Chuẩn bị training environment"""
    print(f"\n🔧 Preparing Enhanced Double DQN Training...")
    
    # Tạo game instance
    if level_choice == 1:
        game = Level1AI()
        level_name = "Level 1"
        state_file = "training_state_enhanced_double_dqn_lv1.pkl"
    else:
        game = Level2AI()
        level_name = "Level 2" 
        state_file = "training_state_enhanced_double_dqn_lv2.pkl"
    
    print(f"✅ {level_name} environment initialized")
    
    # Handle fresh training
    if fresh_training:
        if os.path.exists(state_file):
            backup_file = state_file.replace('.pkl', '_backup.pkl')
            os.rename(state_file, backup_file)
            print(f"✅ Previous state backed up to {backup_file}")
        print(f"✅ Fresh training mode - starting from scratch")
    else:
        if os.path.exists(state_file):
            print(f"✅ Will continue from saved state: {state_file}")
        else:
            print(f"⚠️ No saved state found - starting fresh training")
    
    return game

def run_training(game, num_games):
    """Chạy training với enhanced monitoring"""
    print(f"\n🚀 Starting Enhanced Double DQN Training...")
    print(f"🎯 Target: {num_games} games")
    print(f"🎮 Level: {type(game).__name__}")
    print("-" * 50)
    
    try:
        # Chạy training
        mean_scores, total_wins = train2(game, num_games=num_games)
        
        # Training completed successfully
        final_win_rate = total_wins / len(mean_scores) if mean_scores else 0
        final_mean_score = mean_scores[-1] if mean_scores else 0
        
        print(f"\n🎯 ===== TRAINING COMPLETED SUCCESSFULLY =====")
        print(f"🎮 Total games played: {len(mean_scores)}")
        print(f"🏆 Total wins: {total_wins}")
        print(f"📊 Final win rate: {final_win_rate:.1%}")
        print(f"📈 Final mean score: {final_mean_score:.2f}")
        
        # Performance evaluation
        if final_win_rate >= 0.8:
            print(f"🌟 EXCELLENT! Win rate ≥ 80% - Training very successful!")
        elif final_win_rate >= 0.6:
            print(f"🎉 GOOD! Win rate ≥ 60% - Training successful!")
        elif final_win_rate >= 0.4:
            print(f"👍 FAIR! Win rate ≥ 40% - Reasonable progress!")
        else:
            print(f"📈 Keep training - More games needed for better performance")
        
        return True
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Training interrupted by user!")
        print(f"💾 Progress should be automatically saved")
        return False
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        return False

def show_final_tips():
    """Hiển thị tips cuối"""
    print(f"\n💡 TIPS FOR BETTER PERFORMANCE:")
    print(f"• 🔧 Adjust hyperparameters in agent2.py if needed")
    print(f"• 📊 Check learning curves in plots/ folder")
    print(f"• 🎥 Best games are saved as videos in videos1/ folder")
    print(f"• 💾 Training state is automatically saved every 500 games")
    print(f"• 🔄 Use 'Continue Training' to resume from where you left off")
    print(f"• 🎯 Level 2 is much harder - expect lower win rates initially")

def main():
    """Main function"""
    print_banner()
    print_enhanced_features()
    
    # Get configuration
    level_choice, num_games, fresh_training = get_training_config()
    show_training_summary(level_choice, num_games, fresh_training)
    
    # Confirm before starting
    confirm = input(f"\n❓ Start training with above configuration? (y/n): ").lower()
    if confirm != 'y':
        print(f"❌ Training cancelled by user")
        return
    
    # Prepare and run training
    game = prepare_training(level_choice, fresh_training)
    success = run_training(game, num_games)
    
    if success:
        show_final_tips()
    
    print(f"\n👋 Thank you for using Enhanced Double DQN Training!")
    print(f"🔗 For more advanced settings, edit agent2.py directly")

if __name__ == "__main__":
    # Command line arguments support
    parser = argparse.ArgumentParser(description='Enhanced Double DQN Training')
    parser.add_argument('--level', type=int, choices=[1, 2], help='Game level (1 or 2)')
    parser.add_argument('--games', type=int, help='Number of games to train')
    parser.add_argument('--fresh', action='store_true', help='Start fresh training')
    
    args = parser.parse_args()
    
    if args.level and args.games:
        # Command line mode
        print_banner()
        print(f"🚀 Running in command line mode...")
        
        game = Level1AI() if args.level == 1 else Level2AI()
        if args.fresh:
            state_file = f"training_state_enhanced_double_dqn_lv{args.level}.pkl"
            if os.path.exists(state_file):
                os.remove(state_file)
                print(f"✅ Removed previous state for fresh training")
        
        success = run_training(game, args.games)
        if success:
            show_final_tips()
    else:
        # Interactive mode
        main() 