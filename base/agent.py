import os
os.environ["SDL_VIDEODRIVER"] = "windib"
import torch
import random
import numpy as np
from collections import deque
from game_level1 import Level1AI, Direction, Point
from game_level2 import Level2AI
from model import Linear_QNet
from helper import plot
import sys
from trainer import QTrainer
import pickle
import pygame
import glob
import subprocess

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(12, 256, 128, 4)
        self.target_model = Linear_QNet(12, 256, 128, 4)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.soft_update(tau=0.01)

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def soft_update(self, tau=0.01):
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save_state(self, filename="training_state.pkl"):
        data = {
            "memory": self.memory,
            "epsilon": self.epsilon
        }
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        print(f"Training state saved to {filename}")

    def load_state(self, filename="training_state.pkl"):
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                data = pickle.load(f)
                self.memory = data.get("memory", deque(maxlen=MAX_MEMORY))
                self.epsilon = data.get("epsilon", 0)
            print(f"Training state loaded from {filename}, memory size: {len(self.memory)}")
        else:
            print(f"No saved training state found at {filename}")

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            (dir_r and game.is_collision(point_r)),
            (dir_l and game.is_collision(point_l)),
            (dir_u and game.is_collision(point_u)),
            (dir_d and game.is_collision(point_d)),

            dir_l,
            dir_r,
            dir_u,
            dir_d,

            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones, self.target_model)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step([state], [action], [reward], [next_state], [done], self.target_model)

    def get_action(self, state):
        self.epsilon = max(10, 80 - self.n_games)
        final_move = [0, 0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

folder = "videos2"
if not os.path.exists(folder):
    os.makedirs(folder)

def save_video_from_frames(frames, filename):
    import cv2
    if not frames:
        return
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))
    for f in frames:
        out.write(f)
    for _ in range(5):  # Giữ frame cuối 1s
        out.write(frames[-1])
    out.release()
    print(f"🎥 Saved best game video to {filename}")

def delete_old_videos(filename, prefix='best_gamelv1_'):
    folder = os.path.dirname(filename)
    if folder == '':
        folder = '.'  # nếu filename chỉ là tên file ở thư mục hiện tại

    # Tạo pattern cho file muốn xóa: chỉ xóa file bắt đầu bằng prefix và kết thúc .mp4
    pattern = os.path.join(folder, f'{prefix}*.mp4')

    for fpath in glob.glob(pattern):
        # Không xóa file đang lưu (filename)
        if os.path.abspath(fpath) != os.path.abspath(filename):
            try:
                os.remove(fpath)
                print(f"🗑️ Deleted old video file {fpath}")
            except Exception as e:
                print(f"⚠️ Could not delete {fpath}: {e}")

def open_video_file(filepath):
    full_path = os.path.abspath(filepath)
    if sys.platform == "win32":  # Windows
        os.startfile(full_path)
    elif sys.platform == "darwin":  # macOS
        subprocess.run(["open", full_path])
    else:  # Linux, ...
        subprocess.run(["xdg-open", full_path])

def train(game=Level1AI(), num_games=1000):
    nw=0
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    if isinstance(game, Level1AI): agent.load_state("training_state_dqn_lv1.pkl")
    elif isinstance(game, Level2AI): agent.load_state("training_state_dqn_lv2.pkl")
    a= 'DQN'
    frames = []
    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)
        # Luu video
        pygame.display.flip()
        frame = game.get_frame()
        frames.append(frame)

        if done:
            game.reset()
            agent.n_games += 1
            agent.epsilon = max(0.01, 0.995 ** agent.n_games)
            agent.train_long_memory()

            if agent.n_games % 10 == 0:
                agent.soft_update(tau=0.01)

            # Luu Kinh nghiem
            #if agent.n_games % 5 == 0:
            #   if isinstance(game, Level1AI): agent.save_state("training_state_dqn_lv1.pkl")
            #   elif isinstance(game, Level2AI): agent.save_state("training_state_dqn_lv2.pkl")

            if score > record:
                record = score
                agent.model.save()
                if isinstance(game, Level1AI): 
                    last_best_video = f"videos1/best_gamelv1_dqn_{agent.n_games}.mp4"
                    save_video_from_frames(frames, last_best_video)
                    delete_old_videos(last_best_video, prefix='best_gamelv1_dqn_')
                elif isinstance(game, Level2AI):
                    last_best_video = f"videos1/best_gamelv2_dqn_{agent.n_games}.mp4"
                    save_video_from_frames(frames, last_best_video)
                    delete_old_videos(last_best_video, prefix='best_gamelv2_dqn_')
            if score == 6: nw = nw + 1 
            frames = []
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores, a, nw)

            # Stop after enough games
            if agent.n_games >= num_games:
                break

    # Tự lưu lại đồ thịthị        
    if isinstance(game, Level1AI): final_chart_path = f"plots/dqn_lv1_{num_games}.png"
    elif isinstance(game, Level2AI): final_chart_path = f"plots/dqn_lv2_{num_games}.png"
    plot(plot_scores, plot_mean_scores, a='DQN Training Result', nw=nw , save_path=final_chart_path)

    print("Total number of victories: ", nw)
    print("Training finished.")
    # phát video màn chơi tốt nhất
    #if last_best_video is not None and os.path.exists(last_best_video):
    #    print(f"▶️ Opening best video: {last_best_video}")
    #    open_video_file(last_best_video)
    #lưu lại đường giá trị trung bình
    return plot_mean_scores, nw
    
if __name__ == '__main__':
    train()
