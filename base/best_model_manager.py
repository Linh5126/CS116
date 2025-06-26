#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BEST MODEL MANAGER - Utility để quản lý và xem thống kê best models
"""

import os
import json
from datetime import datetime

class BestModelManager:
    def __init__(self):
        self.results_dir = 'training_results'
        self.models_dir = 'saved_models'
        self.experiences_dir = 'saved_experiences'
        
    def view_all_best_models(self):
        """Hiển thị tất cả best models hiện có"""
        print("🏆 BEST MODELS SUMMARY")
        print("=" * 60)
        
        best_files = []
        for file in os.listdir(self.results_dir):
            if file.endswith('_best_stats.json'):
                best_files.append(file)
        
        if not best_files:
            print("❌ Không tìm thấy best model nào!")
            return
            
        best_files.sort()
        
        for file in best_files:
            try:
                with open(os.path.join(self.results_dir, file), 'r') as f:
                    stats = json.load(f)
                    
                algo = stats.get('algorithm', 'Unknown')
                level = stats.get('level_name', 'Unknown')
                win_rate = stats.get('best_win_rate', 0) * 100
                episode = stats.get('best_model_episode', 0)
                avg_score = stats.get('avg_score', 0)
                timestamp = stats.get('timestamp', 'Unknown')
                
                # Format timestamp
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    time_str = dt.strftime("%Y-%m-%d %H:%M")
                except:
                    time_str = timestamp[:16] if len(timestamp) > 16 else timestamp
                
                print(f"📊 {algo} - {level}")
                print(f"   🏆 Win Rate: {win_rate:.1f}% | Avg Score: {avg_score:.1f}")
                print(f"   📈 Episode: {episode} | Date: {time_str}")
                print(f"   📁 Model: {self.models_dir}/{algo}_{level}_BEST.pth")
                print()
                
            except Exception as e:
                print(f"⚠️ Error reading {file}: {e}")
                
    def compare_algorithms(self, level_name='Level1'):
        """So sánh các algorithms cho một level"""
        print(f"🔄 ALGORITHM COMPARISON - {level_name}")
        print("=" * 50)
        
        algorithms = ['DQN', 'Double_DQN']
        results = []
        
        for algo in algorithms:
            stats_file = f"{self.results_dir}/{algo}_{level_name}_best_stats.json"
            if os.path.exists(stats_file):
                try:
                    with open(stats_file, 'r') as f:
                        stats = json.load(f)
                        results.append({
                            'algorithm': algo,
                            'win_rate': stats.get('best_win_rate', 0) * 100,
                            'episode': stats.get('best_model_episode', 0),
                            'avg_score': stats.get('avg_score', 0)
                        })
                except:
                    print(f"⚠️ Could not read stats for {algo}")
            else:
                print(f"❌ No best model found for {algo}")
                
        if results:
            # Sort by win rate
            results.sort(key=lambda x: x['win_rate'], reverse=True)
            
            print("\n📊 Rankings:")
            for i, result in enumerate(results, 1):
                win_rate = result['win_rate']
                algo = result['algorithm']
                episode = result['episode']
                avg_score = result['avg_score']
                
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
                print(f"{medal} #{i} {algo}: {win_rate:.1f}% win rate, {avg_score:.1f} avg score (ep {episode})")
        else:
            print("❌ No results found for comparison")
            
    def reset_best_stats(self, algorithm, level_name):
        """Reset best stats cho algorithm và level cụ thể"""
        stats_file = f"{self.results_dir}/{algorithm}_{level_name}_best_stats.json"
        
        if os.path.exists(stats_file):
            print(f"⚠️ Are you sure you want to reset best stats for {algorithm} - {level_name}?")
            print(f"This will allow worse models to become 'best' again.")
            confirm = input("Type 'yes' to confirm: ")
            
            if confirm.lower() == 'yes':
                os.remove(stats_file)
                print(f"✅ Reset best stats for {algorithm} - {level_name}")
                print(f"Next training will start tracking from 0% win rate")
            else:
                print("❌ Reset cancelled")
        else:
            print(f"❌ No best stats found for {algorithm} - {level_name}")
            
    def backup_best_models(self):
        """Backup tất cả best models"""
        from datetime import datetime
        import shutil
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"backup_best_models_{timestamp}"
        
        os.makedirs(backup_dir, exist_ok=True)
        
        # Backup best model files
        best_models = []
        for file in os.listdir(self.models_dir):
            if '_BEST.pth' in file:
                best_models.append(file)
                shutil.copy(os.path.join(self.models_dir, file), backup_dir)
                
        # Backup best experiences
        for file in os.listdir(self.experiences_dir):
            if '_BEST.pkl' in file:
                shutil.copy(os.path.join(self.experiences_dir, file), backup_dir)
                
        # Backup best stats
        for file in os.listdir(self.results_dir):
            if '_best_stats.json' in file:
                shutil.copy(os.path.join(self.results_dir, file), backup_dir)
                
        print(f"✅ Backup completed: {backup_dir}")
        print(f"📁 Backed up {len(best_models)} best models and related files")

def main():
    manager = BestModelManager()
    
    while True:
        print("\n🏆 BEST MODEL MANAGER")
        print("=" * 30)
        print("1. View all best models")
        print("2. Compare algorithms (Level1)")
        print("3. Compare algorithms (Level2)")
        print("4. Compare algorithms (Level3)")
        print("5. Reset best stats")
        print("6. Backup best models")
        print("7. Exit")
        
        try:
            choice = input("\nChọn option (1-7): ").strip()
            
            if choice == '1':
                manager.view_all_best_models()
            elif choice == '2':
                manager.compare_algorithms('Level1')
            elif choice == '3':
                manager.compare_algorithms('Level2')
            elif choice == '4':
                manager.compare_algorithms('Level3')
            elif choice == '5':
                print("\nAvailable combinations:")
                manager.view_all_best_models()
                algo = input("Enter algorithm (DQN/Double_DQN): ").strip()
                level = input("Enter level (Level1/Level2/Level3): ").strip()
                manager.reset_best_stats(algo, level)
            elif choice == '6':
                manager.backup_best_models()
            elif choice == '7':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice!")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 