import os
import torch
import numpy as np
from advanced_trainer import AdvancedAgent
from game_level1 import Level1AI
from game_level2 import Level2AI
from game_level3 import Level3AI
import pygame
import time

class GamePlayer:
    def __init__(self):
        self.models_dir = 'saved_models'
        self.experiences_dir = 'saved_experiences'
        
    def play_game(self, level_name='Level1', algorithm='DQN', num_games=5, show_game=True, model_path=None):
        """Chơi game với agent đã được train"""
        
        # Chọn game class
        game_classes = {
            'Level1': Level1AI,
            'Level2': Level2AI,
            'Level3': Level3AI
        }
        
        if level_name not in game_classes:
            print(f"❌ Level {level_name} không hợp lệ. Chọn: {list(game_classes.keys())}")
            return
            
        game_class = game_classes[level_name]
        
        # Tạo agent
        agent = AdvancedAgent(algorithm=algorithm)
        
        # Load model đã train
        if model_path is None:
            model_path = f"{self.models_dir}/{algorithm}_{level_name}.pth"
            
        if not os.path.exists(model_path):
            print(f"❌ Không tìm thấy model đã train tại {model_path}")
            print("Hãy chạy training trước hoặc kiểm tra đường dẫn!")
            return
            
        agent.load_model(model_path)
        
        # Load experience - tùy chọn best experience
        if '_BEST.pth' in model_path:
            exp_path = model_path.replace('.pth', '.pkl').replace('models', 'experiences')
        else:
            exp_path = f"{self.experiences_dir}/{algorithm}_{level_name}.pkl"
        
        if os.path.exists(exp_path):
            agent.load_experience(exp_path)
        else:
            print(f"⚠️ Experience file không tìm thấy: {exp_path}")
        
        print(f"🎮 Chơi {level_name} với {algorithm}")
        print(f"🧠 Agent đã train {agent.n_games} games")
        print(f"📊 Experience buffer có {len(agent.memory)} memories")
        print("=" * 50)
        
        results = []
        
        for game_idx in range(num_games):
            print(f"\n🎯 Game {game_idx + 1}/{num_games}")
            
            # Tạo game mới
            game = game_class()
            
            score = 0
            steps = 0
            start_time = time.time()
            
            # Chơi game
            while True:
                # Lấy state
                state = agent.get_state(game)
                
                # Lấy action (không exploration)
                action = agent.get_action(state, training=False)
                
                # Thực hiện action
                reward, done, current_score = game.play_step(action)
                score = current_score
                steps += 1
                
                # Hiển thị game (nếu được yêu cầu và là game đầu tiên)
                if show_game and game_idx == 0:
                    pygame.display.flip()
                    time.sleep(0.05)  # Làm chậm để có thể xem được
                
                if done:
                    break
                    
                # Tránh vòng lặp vô hạn
                if steps > 1000:
                    print("⚠️ Game quá dài, dừng lại!")
                    break
            
            end_time = time.time()
            game_time = end_time - start_time
            
            # Lưu kết quả
            result = {
                'game': game_idx + 1,
                'score': score,
                'steps': steps,
                'time': game_time
            }
            results.append(result)
            
            # In kết quả game
            print(f"📊 Score: {score}, Steps: {steps}, Time: {game_time:.2f}s")
            
            # Kiểm tra thắng thua
            win_threshold = self._get_win_threshold(level_name)
            if score >= win_threshold:
                print(f"🏆 THẮNG! (Score >= {win_threshold})")
            else:
                print(f"💔 Thua (Score < {win_threshold})")
            
            # Cleanup
            if hasattr(game, 'quit'):
                game.quit()
        
        # Tổng kết
        self._print_summary(results, level_name, algorithm)
        
        return results
    
    def _get_win_threshold(self, level_name):
        """Định nghĩa threshold để coi là thắng game"""
        # Thắng game = ăn hết food theo sequence
        thresholds = {
            'Level1': 6,   # Có 6 food positions trong sequence
            'Level2': 6,   # Cần kiểm tra trong game level2 
            'Level3': 1   # Cần kiểm tra trong game level3
        }
        return thresholds.get(level_name, 6)
    
    def _print_summary(self, results, level_name, algorithm):
        """In tổng kết kết quả"""
        print("\n" + "=" * 50)
        print("📊 TỔNG KẾT KẾT QUẢ")
        print("=" * 50)
        
        scores = [r['score'] for r in results]
        steps_list = [r['steps'] for r in results]
        times = [r['time'] for r in results]
        
        win_threshold = self._get_win_threshold(level_name)
        wins = sum(1 for score in scores if score >= win_threshold)
        win_rate = wins / len(results)
        
        print(f"🎮 Level: {level_name}")
        print(f"🤖 Algorithm: {algorithm}")
        print(f"🎯 Số game chơi: {len(results)}")
        print(f"🏆 Số game thắng: {wins}")
        print(f"📈 Tỷ lệ thắng: {win_rate:.2%}")
        print(f"📊 Score trung bình: {np.mean(scores):.1f}")
        print(f"📊 Score cao nhất: {max(scores)}")
        print(f"📊 Score thấp nhất: {min(scores)}")
        print(f"👟 Steps trung bình: {np.mean(steps_list):.1f}")
        print(f"⏱️ Thời gian trung bình: {np.mean(times):.2f}s")
        
        if wins > 0:
            winning_games = [r for r in results if r['score'] >= win_threshold]
            winning_steps = [r['steps'] for r in winning_games]
            print(f"🎯 Steps trung bình khi thắng: {np.mean(winning_steps):.1f}")
            print(f"🏃‍♂️ Ít steps nhất khi thắng: {min(winning_steps)}")
        
        print("=" * 50)
    
    def compare_algorithms(self, level_name='Level1', num_games=10):
        """So sánh 2 thuật toán DQN và Double DQN"""
        print(f"🔍 SO SÁNH THUẬT TOÁN TRÊN {level_name}")
        print("=" * 60)
        
        algorithms = ['DQN', 'Double_DQN']
        all_results = {}
        
        for algorithm in algorithms:
            print(f"\n🤖 Testing {algorithm}...")
            results = self.play_game(level_name, algorithm, num_games, show_game=False)
            if results:
                all_results[algorithm] = results
        
        if len(all_results) == 2:
            self._compare_results(all_results, level_name)
        
        return all_results
    
    def _compare_results(self, all_results, level_name):
        """So sánh kết quả giữa các thuật toán"""
        print("\n" + "🆚" * 20)
        print("SO SÁNH KẾT QUẢ")
        print("🆚" * 20)
        
        win_threshold = self._get_win_threshold(level_name)
        
        print(f"{'Metric':<20} {'DQN':<15} {'Double DQN':<15} {'Winner':<10}")
        print("-" * 65)
        
        # So sánh các metrics
        metrics = []
        
        for algorithm in ['DQN', 'Double_DQN']:
            if algorithm in all_results:
                results = all_results[algorithm]
                scores = [r['score'] for r in results]
                steps_list = [r['steps'] for r in results]
                wins = sum(1 for score in scores if score >= win_threshold)
                
                metrics.append({
                    'algorithm': algorithm,
                    'avg_score': np.mean(scores),
                    'max_score': max(scores),
                    'win_rate': wins / len(results),
                    'avg_steps': np.mean(steps_list),
                    'wins': wins
                })
        
        if len(metrics) == 2:
            dqn_metrics = metrics[0] if metrics[0]['algorithm'] == 'DQN' else metrics[1]
            ddqn_metrics = metrics[1] if metrics[1]['algorithm'] == 'Double_DQN' else metrics[0]
            
            # So sánh từng metric
            comparisons = [
                ('Win Rate', f"{dqn_metrics['win_rate']:.2%}", f"{ddqn_metrics['win_rate']:.2%}", 
                 'DQN' if dqn_metrics['win_rate'] > ddqn_metrics['win_rate'] else 'Double DQN'),
                ('Avg Score', f"{dqn_metrics['avg_score']:.1f}", f"{ddqn_metrics['avg_score']:.1f}",
                 'DQN' if dqn_metrics['avg_score'] > ddqn_metrics['avg_score'] else 'Double DQN'),
                ('Max Score', f"{dqn_metrics['max_score']}", f"{ddqn_metrics['max_score']}",
                 'DQN' if dqn_metrics['max_score'] > ddqn_metrics['max_score'] else 'Double DQN'),
                ('Avg Steps', f"{dqn_metrics['avg_steps']:.1f}", f"{ddqn_metrics['avg_steps']:.1f}",
                 'DQN' if dqn_metrics['avg_steps'] < ddqn_metrics['avg_steps'] else 'Double DQN'),  # Ít steps hơn = tốt hơn
                ('Total Wins', f"{dqn_metrics['wins']}", f"{ddqn_metrics['wins']}",
                 'DQN' if dqn_metrics['wins'] > ddqn_metrics['wins'] else 'Double DQN')
            ]
            
            for metric, dqn_val, ddqn_val, winner in comparisons:
                print(f"{metric:<20} {dqn_val:<15} {ddqn_val:<15} {winner:<10}")
            
            # Tổng kết
            print("\n🏆 TỔNG KẾT:")
            if ddqn_metrics['win_rate'] > dqn_metrics['win_rate']:
                print(f"Double DQN có tỷ lệ thắng cao hơn ({ddqn_metrics['win_rate']:.2%} vs {dqn_metrics['win_rate']:.2%})")
            elif dqn_metrics['win_rate'] > ddqn_metrics['win_rate']:
                print(f"DQN có tỷ lệ thắng cao hơn ({dqn_metrics['win_rate']:.2%} vs {ddqn_metrics['win_rate']:.2%})")
            else:
                print("Cả hai thuật toán có tỷ lệ thắng tương đương")


def main():
    """Demo chương trình"""
    player = GamePlayer()
    
    print("🎮 SNAKE AI GAME PLAYER")
    print("=" * 50)
    print("Chọn chế độ:")
    print("1. Chơi game với một thuật toán")
    print("2. So sánh 2 thuật toán")
    print("3. Demo tất cả")
    
    try:
        choice = input("\nNhập lựa chọn (1/2/3): ").strip()
        
        if choice == '1':
            # Chọn level
            print("\nChọn level:")
            print("1. Level1 (Dễ)")
            print("2. Level2 (Trung bình)")
            print("3. Level3 (Khó)")
            
            level_choice = input("Nhập lựa chọn (1/2/3): ").strip()
            level_map = {'1': 'Level1', '2': 'Level2', '3': 'Level3'}
            level_name = level_map.get(level_choice, 'Level1')
            
            # Chọn thuật toán
            print("\nChọn thuật toán:")
            print("1. DQN")
            print("2. Double DQN")
            
            algo_choice = input("Nhập lựa chọn (1/2): ").strip()
            algorithm = 'DQN' if algo_choice == '1' else 'Double_DQN'
            
            # Số game
            num_games = int(input("Số game muốn chơi (mặc định 5): ") or "5")
            
            player.play_game(level_name, algorithm, num_games, show_game=True)
            
        elif choice == '2':
            # Chọn level
            print("\nChọn level để so sánh:")
            print("1. Level1 (Dễ)")
            print("2. Level2 (Trung bình)")
            print("3. Level3 (Khó)")
            
            level_choice = input("Nhập lựa chọn (1/2/3): ").strip()
            level_map = {'1': 'Level1', '2': 'Level2', '3': 'Level3'}
            level_name = level_map.get(level_choice, 'Level1')
            
            # Số game
            num_games = int(input("Số game để so sánh (mặc định 10): ") or "10")
            
            player.compare_algorithms(level_name, num_games)
            
        elif choice == '3':
            # Demo tất cả
            print("\n🚀 DEMO TẤT CẢ LEVEL VÀ THUẬT TOÁN")
            
            levels = ['Level1', 'Level2', 'Level3']
            
            for level in levels:
                print(f"\n{'='*20} {level} {'='*20}")
                player.compare_algorithms(level, num_games=5)
                
        else:
            print("❌ Lựa chọn không hợp lệ!")
            
    except KeyboardInterrupt:
        print("\n👋 Tạm biệt!")
    except Exception as e:
        print(f"❌ Lỗi: {e}")


if __name__ == "__main__":
    main()