import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from IPython import display
import os


plt.ion()

def plot(scores, mean_scores, a = '', nw=0, save_path=None):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title(a)
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    if nw is not None:
        plt.text(0.02, 0.95, f"number_game_win: {nw}", transform=plt.gca().transAxes,
                 fontsize=10, color='black', bbox=dict(facecolor='white', alpha=0.5))
    plt.show()
    plt.pause(.1)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"📊 Biểu đồ đã được lưu tại: {save_path}")
# vẽ trên cùng 1 đồ thị
def plot_dual_mean_scores(mean_scores_dqn, mean_scores_ddqn, total_dqn=0, total_ddqn=0, label1='DQN', label2='Double DQN', save_path=None):
    min_len = min(len(mean_scores_dqn), len(mean_scores_ddqn))
    mean_scores_dqn = mean_scores_dqn[:min_len]
    mean_scores_ddqn = mean_scores_ddqn[:min_len]

    plt.figure(figsize=(10, 6))
    plt.title("DQN vs Double DQN")
    plt.xlabel("Number of Games")
    plt.ylabel("Score")

    plt.plot(mean_scores_dqn, label=f"{label1} (Total Wins: {total_dqn})", color='blue')
    plt.plot(mean_scores_ddqn, label=f"{label2} (Total Wins: {total_ddqn})", color='green')
    plt.ylim(ymin=0)

    # Thêm text ở góc trên trái
    plt.text(0.01, 0.95,
             f"{label1} Wins: {total_dqn}\n{label2} Wins: {total_ddqn}",
             transform=plt.gca().transAxes,
             fontsize=10,
             verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.7))

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"📊 Biểu đồ so sánh đã được lưu tại: {save_path}")
    input("Press Enter to exit and close the plot...")
