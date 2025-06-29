#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script để train từng level riêng lẻ với algorithm cụ thể
UPDATED: Sử dụng advanced_trainer.py cho tất cả levels
"""

import sys
from advanced_trainer_fixed import TrainingManager

def main():
    if len(sys.argv) < 3:
        print("🎮 TRAIN SINGLE LEVEL - Snake AI")
        print("=" * 50)
        print("Cách sử dụng:")
        print("python train_single.py <level> <algorithm> [episodes] [eval_interval]")
        print()
        print("Ví dụ:")
        print("python train_single.py Level1 DQN")
        print("python train_single.py Level2 Double_DQN 5000")
        print("python train_single.py Level3 DQN 5000 1000")
        print()
        print("🚀 SỬ DỤNG ADVANCED_TRAINER.PY VỚI FIXED EXPLORATION!")
        print("- Fixed safe exploration (không stuck)")
        print("- Fixed epsilon decay (không catastrophic forgetting)")
        print("- Balanced hyperparameters cho tất cả levels")
        print()
        print("Tham số:")
        print("- level: Level1, Level2, Level3")
        print("- algorithm: DQN, Double_DQN")
        print("- episodes: số episodes train (khuyến nghị: 3000-5000)")
        print("- eval_interval: đánh giá mỗi X episodes (khuyến nghị: 300-500)")
        return
    
    level_name = sys.argv[1]
    algorithm = sys.argv[2]
    
    # Default parameters
    episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    eval_interval = int(sys.argv[4]) if len(sys.argv) > 4 else 300
    eval_games = 50
    
    print(f"🚀 TRAINING SINGLE LEVEL")
    print("=" * 50)
    print(f"🎯 Level: {level_name}")
    print(f"🤖 Algorithm: {algorithm}")
    print(f"📊 Episodes: {episodes}")
    print(f"⏱️ Eval interval: {eval_interval}")
    print(f"🎮 Eval games: {eval_games}")
    print("=" * 50)
    
    try:
        # Kiểm tra tham số hợp lệ
        valid_levels = ['Level1', 'Level2', 'Level3']
        valid_algos = ['DQN', 'Double_DQN']
        
        if level_name not in valid_levels:
            print(f"❌ Level không hợp lệ: {level_name}")
            print(f"Chọn một trong: {valid_levels}")
            return
            
        if algorithm not in valid_algos:
            print(f"❌ Algorithm không hợp lệ: {algorithm}")
            print(f"Chọn một trong: {valid_algos}")
            return
        
        # Thông tin về level
        level_info = {
            'Level1': {'enemies': 6, 'difficulty': 'Easy', 'target_win_rate': 0.3},
            'Level2': {'enemies': 12, 'difficulty': 'Medium', 'target_win_rate': 0.2},
            'Level3': {'enemies': 18, 'difficulty': 'Hard', 'target_win_rate': 0.15}
        }
        
        info = level_info[level_name]
        print(f"🎯 {level_name} - {info['enemies']} enemies ({info['difficulty']})")
        print(f"🏆 Target win rate: {info['target_win_rate']:.0%}+")
        print(f"✅ Using fixed exploration (no stuck problems)")
        print()
        
        # Training với advanced_trainer.py
        manager = TrainingManager()
        results = manager.train_single_level(
            level_name=level_name,
            algorithm=algorithm,
            total_episodes=episodes,
            eval_interval=eval_interval,
            eval_games=eval_games
        )
        
        # Hiển thị kết quả
        if results and results['evaluation_results']:
            final_eval = results['evaluation_results'][-1]
            print(f"\n🎉 {level_name.upper()} TRAINING HOÀN THÀNH!")
            print("=" * 60)
            print(f"🏆 Kết quả cuối cùng:")
            print(f"   Win Rate: {final_eval['win_rate']:.2%}")
            print(f"   Avg Score: {final_eval['avg_score']:.1f}/6")
            print(f"   Games Won: {final_eval['games_won']}/{eval_games}")
            
            print(f"\n📁 Saved Files:")
            print(f"   Latest Model: saved_models/{algorithm}_{level_name}.pth")
            print(f"   Best Model: saved_models/{algorithm}_{level_name}_BEST.pth")
            print(f"   Results: training_results/{algorithm}_{level_name}_results.json")
            print(f"   Plots: plots/{algorithm}_{level_name}_training.png")
            
            # Performance assessment
            target_rate = info['target_win_rate']
            if final_eval['win_rate'] >= target_rate:
                print(f"\n🌟 EXCELLENT! Win rate ≥{target_rate:.0%} - Model đã học tốt {level_name}!")
            elif final_eval['win_rate'] >= target_rate * 0.5:
                print(f"\n✅ GOOD! Win rate ≥{target_rate*0.5:.0%} - Model có tiềm năng, có thể train thêm")
            elif final_eval['avg_score'] >= 2.5:
                print(f"\n⚠️ PROGRESS! Avg score ≥2.5 - Đang học, cần train lâu hơn")
            else:
                print(f"\n❌ NEEDS WORK! Avg score <2.5 - Cần train thêm hoặc điều chỉnh")
        
        print(f"\n🎮 Để chơi game với model đã train:")
        print(f"   python play_trained_agent.py")
        
    except KeyboardInterrupt:
        print("\n👋 Training bị dừng bởi user!")
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 