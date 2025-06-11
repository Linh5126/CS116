import os
os.environ["SDL_VIDEODRIVER"] = "windib"
import torch
import random
import numpy as np
from collections import deque
from game_level1 import Level1AI, Direction, Point, SPEED
from game_level2 import Level2AI
from game_level3 import Level3AI
from model import Linear_QNet
from helper import plot
import sys
from trainer import QTrainer
import pickle
import pygame
import glob
import subprocess
import math

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done, td_error=1.0):
        max_priority = max(self.priorities) if self.priorities else 1.0
        priority = max_priority if td_error is None else abs(td_error) + 1e-6
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(priority ** self.alpha)
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
            self.priorities[self.position] = priority ** self.alpha
            self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) < batch_size:
            return None
            
        priorities = np.array(self.priorities[:len(self.buffer)])
        probabilities = priorities / priorities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        
        # Importance sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights = weights / weights.max()
        
        return samples, indices, weights
    
    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = (abs(td_error) + 1e-6) ** self.alpha
    
    def __len__(self):
        return len(self.buffer)

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0.95  # Bắt đầu với exploration cao hơn
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998  # Decay chậm hơn
        self.gamma = 0.95  # Tăng gamma để quan tâm long-term hơn
        
        # Prioritized Experience Replay
        self.memory = PrioritizedReplayBuffer(MAX_MEMORY)
        self.beta = 0.4  # Importance sampling beta
        self.beta_increment = 0.001
        
        # Enhanced state representation (15 features instead of 11)
        self.model = Linear_QNet(15, 512, 256, 4)
        self.target_model = Linear_QNet(15, 512, 256, 4)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        
        # Performance tracking
        self.total_wins = 0
        
        # Learning rate scheduling
        self.initial_lr = LR
        self.lr_decay = 0.9999
        
        # Performance tracking
        self.recent_scores = deque(maxlen=100)
        self.score_improvement_threshold = 10
        


    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def soft_update(self, tau=0.005):  # Soft update với tau nhỏ hơn
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def adjust_learning_rate(self):
        """Điều chỉnh learning rate theo thời gian"""
        new_lr = self.initial_lr * (self.lr_decay ** self.n_games)
        for param_group in self.trainer.optimizer.param_groups:
            param_group['lr'] = max(new_lr, 0.0001)  # Minimum LR

    def save_state(self, filename="training_state.pkl"):
        data = {
            "memory": self.memory,
            "epsilon": self.epsilon,
            "total_wins": self.total_wins
        }
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        print(f"Training state saved to {filename}")

    def load_state(self, filename="training_state.pkl"):
        if os.path.exists(filename):
            try:
                with open(filename, "rb") as f:
                    data = pickle.load(f)
                    
                    # Handle compatibility with old deque-based memory
                    old_memory = data.get("memory", deque(maxlen=MAX_MEMORY))
                    if isinstance(old_memory, deque):
                        print(f"⚠️ Converting old deque memory to PrioritizedReplayBuffer...")
                        # Convert old deque to new prioritized memory
                        self.memory = PrioritizedReplayBuffer(MAX_MEMORY)
                        for experience in list(old_memory):
                            if len(experience) == 5:  # (state, action, reward, next_state, done)
                                state, action, reward, next_state, done = experience
                                # Check if state has correct size, if not skip
                                if hasattr(state, 'shape') and len(state) == 15:
                                    self.memory.push(*experience)
                                elif hasattr(state, '__len__') and len(state) == 15:
                                    self.memory.push(*experience)
                        print(f"✅ Converted compatible experiences to prioritized memory")
                    else:
                        self.memory = old_memory
                    
                    self.epsilon = data.get("epsilon", 0.5)
                    self.total_wins = data.get("total_wins", 0)
                    
                print(f"Training state loaded from {filename}")
                print(f"⚠️ Note: Old experience data with incompatible state size was skipped")
            except Exception as e:
                print(f"⚠️ Error loading training state: {e}")
                print(f"Starting with fresh state...")
                self.memory = PrioritizedReplayBuffer(MAX_MEMORY)
                self.epsilon = 0.95
                self.total_wins = 0
        else:
            print(f"No saved training state found at {filename}")

    def get_state(self, game):
        head = game.head
        
        # Kiểm tra collision ở 4 hướng
        point_l = Point(head.x - SPEED, head.y)
        point_r = Point(head.x + SPEED, head.y)
        point_u = Point(head.x, head.y - SPEED)
        point_d = Point(head.x, head.y + SPEED)
        
        # Hướng hiện tại
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        
        # Enhanced state với thêm thông tin về enemy và wall distance
        state = [
            # Danger straight, right, left (3 features)
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),
            
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),
            
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction (4 features)
            dir_l, dir_r, dir_u, dir_d,
            
            # Food location (4 features)
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y,  # food down
            
            # Distance to food (normalized) (1 feature)
            min(1.0, math.sqrt((game.food.x - head.x)**2 + (game.food.y - head.y)**2) / 1000),
            
            # Enemy danger in 4 directions (3 features)
            game.is_collision_enemy_at(point_l) if hasattr(game, 'is_collision_enemy_at') else False,
            game.is_collision_enemy_at(point_r) if hasattr(game, 'is_collision_enemy_at') else False,
            game.is_collision_enemy_at(point_u) if hasattr(game, 'is_collision_enemy_at') else False,
        ]
        
        return np.array(state, dtype=float)

    def remember(self, state, action, reward, next_state, done, td_error=None):
        self.memory.push(state, action, reward, next_state, done, td_error)

    def train_long_memory(self):
        if len(self.memory) < BATCH_SIZE:
            return
            
        # Prioritized sampling
        batch_data = self.memory.sample(BATCH_SIZE, self.beta)
        if batch_data is None:
            return
            
        samples, indices, weights = batch_data
        
        states, actions, rewards, next_states, dones = zip(*samples)
        
        # Compute TD errors for priority updates
        td_errors = self.trainer.train_step_prioritized(
            states, actions, rewards, next_states, dones, 
            self.target_model, weights
        )
        
        # Update priorities
        self.memory.update_priorities(indices, td_errors)
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)

    def train_short_memory(self, state, action, reward, next_state, done):
        td_error = self.trainer.train_step([state], [action], [reward], [next_state], [done], self.target_model)
        # Store với TD error
        self.remember(state, action, reward, next_state, done, td_error)

    def get_action(self, state):
        # Standard epsilon-greedy action selection
        if random.random() < self.epsilon:
            move = random.randint(0, 3)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
        
        final_move = [0, 0, 0, 0]
        final_move[move] = 1
        return final_move

folder = "videos1"
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
    nw = 0
    plot_rewards = []
    plot_mean_rewards = []
    total_reward = 0
    record = 0  
    agent = Agent()
    
    # Load previous state
    if isinstance(game, Level1AI): 
        agent.load_state("training_state_dqn_lv1.pkl")
    elif isinstance(game, Level2AI): 
        agent.load_state("training_state_dqn_lv2.pkl")
    elif isinstance(game, Level3AI): 
        agent.load_state("training_state_dqn_lv3.pkl")
    
    frames = []
    last_best_video = None
    consecutive_wins = 0
    
    print(f"🚀 Bắt đầu training")
    
    while True:
        
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        
        # Train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        
        # Save frame for video
        pygame.display.flip()
        frame = game.get_frame()
        frames.append(frame)
        
        if done:
            game.reset()
            agent.n_games += 1
            
            # Count wins
            win = score >= 10
            if win:
                consecutive_wins += 1
                nw += 1
                agent.total_wins += 1
            else:
                consecutive_wins = 0
            
            # Adaptive epsilon decay based on performance
            if consecutive_wins >= 3:
                # Nhanh decay khi đang thắng liên tiếp
                agent.epsilon = max(agent.epsilon_min, agent.epsilon * 0.99)
            else:
                # Decay bình thường
                agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
            
            # Adjust learning rate
            agent.adjust_learning_rate()
            
            # Train long memory with prioritized replay
            agent.train_long_memory()
            
            # Soft update target network
            if agent.n_games % 10 == 0:
                agent.soft_update()
            
            # Hard update target network periodically
            if agent.n_games % 100 == 0:
                agent.update_target()
            
            # Enhanced reward tracking
            agent.recent_scores.append(score)
            
            # Save best model and video
            if score > record:
                record = score
                agent.model.save()
                if isinstance(game, Level1AI): 
                    last_best_video = f"videos1/best_gamelv1_dqn_{agent.n_games}_score{score}.mp4"
                    save_video_from_frames(frames, last_best_video)
                    delete_old_videos(last_best_video, prefix='best_gamelv1_dqn_')
                elif isinstance(game, Level2AI):
                    last_best_video = f"videos1/best_gamelv2_dqn_{agent.n_games}_score{score}.mp4"
                    save_video_from_frames(frames, last_best_video)
                    delete_old_videos(last_best_video, prefix='best_gamelv2_dqn_')
                elif isinstance(game, Level3AI):
                    last_best_video = f"videos1/best_gamelv3_dqn_{agent.n_games}_score{score}.mp4"
                    save_video_from_frames(frames, last_best_video)
                    delete_old_videos(last_best_video, prefix='best_gamelv3_dqn_')
                
                print(f"🏆 Kỷ lục mới! Score: {score}")
            
            frames = []
            plot_rewards.append(reward)
            total_reward += reward
            mean_reward = total_reward / agent.n_games
            plot_mean_rewards.append(mean_reward)
            
            # Enhanced progress reporting
            if agent.n_games % 50 == 0:
                recent_mean = np.mean(agent.recent_scores) if agent.recent_scores else 0
                win_rate = nw / agent.n_games
                current_lr = agent.trainer.optimizer.param_groups[0]['lr']
                
                print(f"🎮 Game {agent.n_games}")
                print(f"   📈 Score: {score}, Reward: {reward:.2f}, Mean Reward: {mean_reward:.2f}, Recent Mean: {recent_mean:.2f}")
                print(f"   🏅 Record: {record}, Wins: {nw} ({win_rate:.1%})")
                print(f"   🎚️ Epsilon: {agent.epsilon:.3f}")
                print(f"   📚 Learning Rate: {current_lr:.6f}, Memory: {len(agent.memory)}")
                print(f"   🔄 Consecutive Wins: {consecutive_wins}")
            
            # Save training state periodically
            if agent.n_games % 200 == 0:
                if isinstance(game, Level1AI): 
                    agent.save_state("training_state_dqn_lv1.pkl")
                elif isinstance(game, Level2AI): 
                    agent.save_state("training_state_dqn_lv2.pkl")
                elif isinstance(game, Level3AI): 
                    agent.save_state("training_state_dqn_lv3.pkl")
            
            plot(plot_rewards, plot_mean_rewards, 'DQN Training - Rewards', nw)
            
            # Early stopping conditions
            if agent.n_games >= 200:
                recent_win_rate = sum(1 for s in list(agent.recent_scores)[-50:] if s >= 10) / min(50, len(agent.recent_scores))
                
                # Stop if good performance
                if recent_win_rate >= 0.8 and agent.n_games >= 500:
                    print(f"✅ Training hoàn thành! Win rate {recent_win_rate:.1%}")
                    break
            
            # Maximum games limit
            if agent.n_games >= num_games:
                print(f"⏰ Đạt giới hạn {num_games} games")
                break
    
    # Final save and results
    if isinstance(game, Level1AI): 
        final_chart_path = f"plots/dqn_lv1_curriculum_{num_games}.png"
        agent.save_state("training_state_dqn_lv1.pkl")
    elif isinstance(game, Level2AI): 
        final_chart_path = f"plots/dqn_lv2_curriculum_{num_games}.png"
        agent.save_state("training_state_dqn_lv2.pkl")
    elif isinstance(game, Level3AI): 
        final_chart_path = f"plots/dqn_lv3_curriculum_{num_games}.png"
        agent.save_state("training_state_dqn_lv3.pkl")
    
    plot(plot_rewards, plot_mean_rewards, 
         a='DQN Training Results - Rewards', 
         nw=nw, save_path=final_chart_path)
    
    # Detailed final statistics
    final_win_rate = nw / agent.n_games
    recent_win_rate = sum(1 for s in list(agent.recent_scores)[-50:] if s >= 10) / min(50, len(agent.recent_scores))
    
    print(f"\n" + "="*60)
    print(f"🎊 TRAINING HOÀN THÀNH!")
    print(f"="*60)
    print(f"📊 Thống kê tổng thể:")
    print(f"   🏆 Tổng victories: {nw}/{agent.n_games} ({final_win_rate:.1%})")
    print(f"   📈 Recent win rate (50 games cuối): {recent_win_rate:.1%}")
    print(f"   🎯 Best score: {record}")
    print(f"   📊 Final mean reward: {mean_reward:.2f}")
    print(f"   🧠 Final epsilon: {agent.epsilon:.3f}")
    print(f"   💾 Memory size: {len(agent.memory)}")
    
    if last_best_video:
        print(f"   🎥 Best video: {last_best_video}")
        try:
            open_video_file(last_best_video)
        except:
            pass
    
    return plot_mean_rewards, nw

    
if __name__ == '__main__':
    train()