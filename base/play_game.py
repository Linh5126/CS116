#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script để chơi game với level và algorithm cụ thể
"""

import sys
import os
from play_trained_agent import GamePlayer

def main():
    if len(sys.argv) < 2:
        print("🎮 PLAY GAME - Snake AI")
        print("=" * 50)
        print("Cách sử dụng:")
        print("python play_game.py <level> <algorithm> [num_games] [show_game] [use_best]")
        print()
        print("Ví dụ:")
        print("python play_game.py Level1 DQN")
        print("python play_game.py Level2 Double_DQN 10")
        print("python play_game.py Level3 DQN 5 True best")
        print()
        print("Tham số:")
        print("- level: Level1, Level2, Level3")
        print("- algorithm: DQN, Double_DQN")
        print("- num_games: số game chơi (mặc định 5)")
        print("- show_game: hiển thị game (True/False, mặc định True)")
        print("- use_best: dùng best model (best, mặc định latest)")
        print()
        print("📁 Để xem models có sẵn:")
        print("python play_game.py list")
        return
    
    if sys.argv[1].lower() == 'list':
        print("🎮 MODELS CÓ SẴN:")
        print("=" * 50)
        
        models_dir = 'saved_models'
        if not os.path.exists(models_dir):
            print("❌ Thư mục saved_models không tồn tại!")
            return
            
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
        
        if not model_files:
            print("❌ Không có model nào được train!")
            print("Hãy train model trước bằng lệnh:")
            print("python train_single.py Level1 DQN")
            return
        
        print("Models đã train:")
        for i, model_file in enumerate(sorted(model_files), 1):
            # Parse filename: algorithm_level.pth
            name_parts = model_file.replace('.pth', '').split('_')
            if len(name_parts) >= 2:
                algorithm = name_parts[0]
                level = name_parts[1]
                print(f"{i:2d}. {level} với {algorithm}")
                print(f"     Lệnh chơi: python play_game.py {level} {algorithm}")
        
        print("\nVí dụ chơi game:")
        if model_files:
            first_model = model_files[0].replace('.pth', '').split('_')
            if len(first_model) >= 2:
                print(f"python play_game.py {first_model[1]} {first_model[0]} 10")
        return
    
    if len(sys.argv) < 3:
        print("❌ Thiếu tham số!")
        print("Cách sử dụng: python play_game.py <level> <algorithm>")
        print("Ví dụ: python play_game.py Level1 DQN")
        return
    
    level_name = sys.argv[1]
    algorithm = sys.argv[2]
    num_games = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    show_game = sys.argv[4].lower() == 'true' if len(sys.argv) > 4 else True
    use_best = sys.argv[5].lower() == 'best' if len(sys.argv) > 5 else False
    
    print(f"🎮 CHƠI GAME")
    print("=" * 50)
    print(f"🎯 Level: {level_name}")
    print(f"🤖 Algorithm: {algorithm}")
    print(f"🎲 Số games: {num_games}")
    print(f"👁️ Hiển thị: {'Có' if show_game else 'Không'}")
    print(f"📈 Model type: {'Best' if use_best else 'Latest'}")
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
        
        # Kiểm tra model tồn tại
        if use_best:
            model_path = f"saved_models/{algorithm}_{level_name}_BEST.pth"
            model_type = "best"
        else:
            model_path = f"saved_models/{algorithm}_{level_name}.pth"
            model_type = "latest"
            
        if not os.path.exists(model_path):
            print(f"❌ Model {model_type} không tồn tại: {model_path}")
            
            # Check if other model type exists
            alt_path = f"saved_models/{algorithm}_{level_name}_BEST.pth" if not use_best else f"saved_models/{algorithm}_{level_name}.pth"
            alt_type = "best" if not use_best else "latest"
            
            if os.path.exists(alt_path):
                print(f"💡 Tuy nhiên có model {alt_type}: {alt_path}")
                print(f"Hãy thử lại với tham số '{alt_type}' hoặc train lại model.")
            else:
                print(f"Hãy train model trước bằng lệnh:")
                print(f"python train_single.py {level_name} {algorithm}")
            return
        
        # Tạo player và chơi game
        player = GamePlayer()
        
        print("🎯 Bắt đầu chơi game...")
        print()
        
        results = player.play_game(
            level_name=level_name,
            algorithm=algorithm,
            num_games=num_games,
            show_game=show_game,
            model_path=model_path
        )
        
        if results:
            print("\n🎉 HOÀN THÀNH!")
            print("=" * 50)
            
            # Thống kê nhanh
            scores = [r['score'] for r in results]
            steps = [r['steps'] for r in results]
            
            # Kiểm tra thắng game (ăn hết food) - Updated thresholds
            win_threshold = 6 if level_name == 'Level1' else 6 if level_name == 'Level2' else 1
            wins = sum(1 for score in scores if score >= win_threshold)
            
            print(f"🏆 Tổng games thắng: {wins}/{num_games}")
            print(f"📊 Score cao nhất: {max(scores)}")
            print(f"👟 Steps ít nhất: {min(steps)}")
            
            if wins > 0:
                winning_results = [r for r in results if r['score'] >= win_threshold]
                avg_steps_win = sum(r['steps'] for r in winning_results) / len(winning_results)
                print(f"🎯 Steps trung bình khi thắng: {avg_steps_win:.1f}")
            
            print(f"\n📈 Để xem training progress:")
            print(f"   Mở file: plots/{algorithm}_{level_name}_training.png")
        
    except KeyboardInterrupt:
        print("\n👋 Game bị dừng bởi user!")
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 