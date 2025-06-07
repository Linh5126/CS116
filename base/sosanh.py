import sys
from agent import train
from agent2 import train2
from game_level1 import Level1AI
from game_level2 import Level2AI
from helper import plot_dual_mean_scores

# Bản đồ tên level
LEVEL_MAP = {
    "level1": Level1AI,
    "level2": Level2AI
}

def main():
    if len(sys.argv) != 3:
        print("❌ Cách dùng: py sosanh.py level1 100")
        return

    level_name = sys.argv[1].lower()
    num_games = int(sys.argv[2])

    if level_name not in LEVEL_MAP:
        print("❌ Level không hợp lệ. Chọn: level1 hoặc level2")
        return

    level_class = LEVEL_MAP[level_name]

    print(f"🚀 Đang train DQN trên {level_name} ({num_games} ván)...")
    mean_scores_dqn, total_dqn = train(level_class(), num_games)

    print(f"🚀 Đang train Double DQN trên {level_name} ({num_games} ván)...")
    mean_scores_ddqn, total_ddqn = train2(level_class(), num_games)

    print(f"✅ Tổng số màn thắng DQN: {total_dqn}")
    print(f"✅ Tổng số màn thắng Double DQN: {total_ddqn}")

    # Tự lưu lại đồ thịthị        
    if isinstance(level_class(), Level1AI): final_chart_path = f"plots/dqn_vs_db_dqn_lv1_{num_games}.png"
    elif isinstance(level_class(), Level2AI): final_chart_path = f"plots/dqn_vs_db_dqn_lv2_{num_games}.png"
    plot_dual_mean_scores(mean_scores_dqn, mean_scores_ddqn, label1='DQN', label2='Double DQN', save_path=final_chart_path)

if __name__ == "__main__":
    main()
