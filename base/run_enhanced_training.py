import os
os.environ["SDL_VIDEODRIVER"] = "windib"
import sys
import argparse
from agent import train
from game_level1 import Level1AI
from game_level2 import Level2AI
import torch

def run_enhanced_training():
    """Chạy enhanced training với curriculum learning tự động"""
    
    print("🚀 ENHANCED DQN TRAINING")
    print("="*60)
    print("🔧 Các cải tiến chính:")
    print("   • Curriculum Learning tự động")
    print("   • Prioritized Experience Replay")
    print("   • Enhanced State Representation (15 features)")
    print("   • Improved Reward Shaping")
    print("   • Better Neural Network Architecture")
    print("   • Learning Rate Scheduling")
    print("="*60)
    
    # Chọn level
    while True:
        try:
            print("\n🎮 Chọn level để training:")
            print("   1. Level 1 AI")
            print("   2. Level 2 AI")
            choice = input("Nhập lựa chọn (1-2): ").strip()
            
            if choice == "1":
                game = Level1AI()
                level_name = "Level 1"
                break
            elif choice == "2":
                game = Level2AI()
                level_name = "Level 2"
                break
            else:
                print("❌ Lựa chọn không hợp lệ. Vui lòng nhập 1 hoặc 2.")
        except KeyboardInterrupt:
            print("\n👋 Thoát training.")
            return
    
    # Chọn số games
    while True:
        try:
            num_games_input = input(f"\n🎯 Số games để train (mặc định: 1000): ").strip()
            num_games = int(num_games_input) if num_games_input else 1000
            if num_games > 0:
                break
            else:
                print("❌ Số games phải lớn hơn 0.")
        except ValueError:
            print("❌ Vui lòng nhập số hợp lệ.")
        except KeyboardInterrupt:
            print("\n👋 Thoát training.")
            return
    
    # Hiển thị cấu hình
    print(f"\n⚙️ CẤU HÌNH TRAINING:")
    print(f"   🎮 Game: {level_name}")
    print(f"   🎯 Target games: {num_games}")
    print(f"   🧠 Architecture: Enhanced Dueling DQN")
    print(f"   💾 Memory: Prioritized Experience Replay")
    print(f"   🎚️ Curriculum: Tự động (1→4)")
    print(f"   📊 State features: 15")
    print(f"   ⚡ GPU: {'Available' if torch.cuda.is_available() else 'Not available'}")
    
    # Xác nhận
    confirm = input(f"\n✅ Bắt đầu training? (y/n): ").lower().strip()
    if confirm != 'y':
        print("👋 Hủy training.")
        return
    
    print(f"\n🚀 BẮT ĐẦU ENHANCED TRAINING...")
    print("="*60)
    
    try:
        # Chạy training với enhanced agent
        plot_scores, total_wins = train(game=game, num_games=num_games)
        
        print("\n" + "="*60)
        print("🎊 TRAINING HOÀN THÀNH THÀNH CÔNG!")
        print("="*60)
        print(f"📊 Kết quả cuối cùng:")
        print(f"   🏆 Total wins: {total_wins}")
        print(f"   📈 Final mean score: {plot_scores[-1] if plot_scores else 0:.2f}")
        print(f"   🎯 Games trained: {len(plot_scores)}")
        
        # Gợi ý testing
        print(f"\n💡 Để test agent đã train:")
        print(f"   python test_enhanced_agent.py")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Training bị dừng bởi người dùng.")
        print(f"   💾 Model và training state đã được lưu.")
    except Exception as e:
        print(f"\n❌ Lỗi trong quá trình training: {e}")
        import traceback
        traceback.print_exc()

def quick_test_mode():
    """Chế độ test nhanh với ít games"""
    print("🧪 QUICK TEST MODE")
    print("Training với 100 games để test nhanh các cải tiến...")
    
    game = Level1AI()
    try:
        plot_scores, total_wins = train(game=game, num_games=100)
        print(f"✅ Quick test hoàn thành! Wins: {total_wins}/100")
    except Exception as e:
        print(f"❌ Quick test failed: {e}")

def main():
    """Main function với argument parsing"""
    parser = argparse.ArgumentParser(description="Enhanced DQN Training Script")
    parser.add_argument("--quick", action="store_true", 
                       help="Chạy quick test với 100 games")
    parser.add_argument("--level", type=int, choices=[1, 2], 
                       help="Chọn level (1 hoặc 2)")
    parser.add_argument("--games", type=int, default=1000,
                       help="Số games để train (mặc định: 1000)")
    
    args = parser.parse_args()
    
    if args.quick:
        quick_test_mode()
        return
    
    # Auto mode nếu có args
    if args.level:
        game = Level1AI() if args.level == 1 else Level2AI()
        level_name = f"Level {args.level}"
        
        print(f"🚀 AUTO TRAINING: {level_name}, {args.games} games")
        try:
            plot_scores, total_wins = train(game=game, num_games=args.games)
            print(f"✅ Training hoàn thành! Wins: {total_wins}")
        except Exception as e:
            print(f"❌ Training failed: {e}")
    else:
        # Interactive mode
        run_enhanced_training()

if __name__ == "__main__":
    main() 