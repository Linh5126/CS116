import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
from datetime import datetime
from collections import deque
from game_level1 import Level1AI, Direction, Point
from game_level2 import Level2AI  
from game_level3 import Level3AI
from model import Linear_QNet
from trainer import QTrainer
from trainer2_fixed import QTrainer as DoubleQTrainer
import pygame
import cv2

class AdvancedAgent:
    def __init__(self, algorithm='DQN'):
        self.algorithm = algorithm  # 'DQN' hoặc 'Double_DQN'
        self.n_games = 0
        
        # AGGRESSIVE: Exact parameters từ version đạt 50% win rate
        self.epsilon = 0.5   # Start much lower  
        self.epsilon_min = 0.01  # Very low minimum
        self.epsilon_decay = 0.990  # Much faster decay (không dùng với adaptive)
        self.gamma = 0.99  
        
        # IMPROVED: Enhanced memory management
        self.memory = deque(maxlen=200000)  
        self.priority_memory = deque(maxlen=30000)  # High priority experiences
        self.success_memory = deque(maxlen=20000)
        self.success_count = 0
        
        # IMPROVED: More features for better decision making
        self.model = Linear_QNet(22, 256, 128, 4)  # 22 enhanced features
        self.target_model = Linear_QNet(22, 256, 128, 4)
        
        # IMPROVED: Stable learning rates to prevent forgetting
        if algorithm == 'DQN':
            self.trainer = QTrainer(self.model, lr=0.0005, gamma=self.gamma)  # More stable
        else:  # Double DQN
            self.trainer = DoubleQTrainer(self.model, lr=0.0005, gamma=self.gamma)
            
        self.update_target_network()
        
        # IMPROVED: Smart exploration tracking
        self.state_visits = {}  # Track state visitation for better exploration
        
    def update_target_network(self):
        """Copy weights từ main network sang target network"""
        self.target_model.load_state_dict(self.model.state_dict())
        
    def soft_update_target(self, tau=0.01):
        """Soft update target network"""
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def get_enhanced_state(self, game):
        """IMPROVED STATE với 22 features cho hiệu suất cao hơn"""
        head = game.snake[0]
        
        BLOCK_SIZE = 20
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        if hasattr(game, 'food'):
            food_x, food_y = game.food.x, game.food.y
        else:
            food_x, food_y = game.food_x, game.food_y

            # IMPROVED STATE: 22 enhanced features
        state = [
            # Basic danger detection (4 features)
            game.is_collision(point_r),  # Danger RIGHT [0] 
            game.is_collision(point_d),  # Danger DOWN [1]
            game.is_collision(point_l),  # Danger LEFT [2]
            game.is_collision(point_u),  # Danger UP [3]

            # Current direction (4 features)
            dir_r, dir_d, dir_l, dir_u,

            # Food direction (4 features)
            food_x < head.x,  # food left
            food_x > head.x,  # food right
            food_y < head.y,  # food up  
            food_y > head.y,  # food down
            
            # Enhanced distance features (3 features)
            min(abs(food_x - head.x) / 1000.0, 1.0),  # X distance [12]
            min(abs(food_y - head.y) / 500.0, 1.0),   # Y distance [13]
            min(np.sqrt((food_x - head.x)**2 + (food_y - head.y)**2) / 500.0, 1.0),  # Euclidean distance [14]
            
            # Advanced game state (4 features)
            min(game.score / 6.0, 1.0),  # Progress 0-1 [15]
            self._get_closest_enemy_distance(game) / 100.0,  # Normalized distance [16] 
            self._predict_enemy_collision_risk(game),  # Risk prediction [17]
            self._get_safe_direction_count(game),  # Safe directions [18]
            
            # IMPROVED FEATURES (3 additional features)
            self._get_path_efficiency(game, food_x, food_y),  # Path planning [19]
            self._get_area_control(game),  # Space control [20]
            self._get_momentum_score(game)  # Movement momentum [21]
        ]

        return np.array(state, dtype=np.float32)
    
    def _get_path_efficiency(self, game, food_x, food_y):
        """IMPROVED: Tính hiệu quả đường đi đến food"""
        head = game.head
        # Simple A* heuristic: Manhattan + obstacles penalty
        manhattan_dist = abs(food_x - head.x) + abs(food_y - head.y)
        
        # Check if direct path has obstacles
        steps_x = 1 if food_x > head.x else (-1 if food_x < head.x else 0)
        steps_y = 1 if food_y > head.y else (-1 if food_y < head.y else 0)
        
        obstacle_penalty = 0
        # Sample a few points along the path
        for i in range(1, min(int(manhattan_dist/20), 5)):
            test_x = head.x + i * steps_x * 20
            test_y = head.y + i * steps_y * 20
            test_point = Point(test_x, test_y)
            if game.is_collision(test_point):
                obstacle_penalty += 0.2
                
        return max(0, 1.0 - obstacle_penalty)  # Higher = better path
    
    def _get_area_control(self, game):
        """IMPROVED: Tính khả năng kiểm soát không gian"""
        head = game.head
        BLOCK_SIZE = 20
        
        # Count accessible area in 3x3 grid around head
        accessible = 0
        total = 0
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                test_point = Point(head.x + dx * BLOCK_SIZE, head.y + dy * BLOCK_SIZE)
                total += 1
                if not game.is_collision(test_point):
                    accessible += 1
                    
        return accessible / total if total > 0 else 0
    
    def _get_momentum_score(self, game):
        """IMPROVED: Tính điểm momentum dựa trên hướng di chuyển"""
        # Reward consistent movement towards food
        head = game.head
        
        if hasattr(game, 'food'):
            food_x, food_y = game.food.x, game.food.y
        else:
            food_x, food_y = game.food_x, game.food_y
            
        # Calculate if current direction is towards food
        food_direction_x = 1 if food_x > head.x else (-1 if food_x < head.x else 0)
        food_direction_y = 1 if food_y > head.y else (-1 if food_y < head.y else 0)
        
        momentum = 0.5  # Base momentum
        
        if game.direction == Direction.RIGHT and food_direction_x > 0:
            momentum += 0.3
        elif game.direction == Direction.LEFT and food_direction_x < 0:
            momentum += 0.3
        elif game.direction == Direction.DOWN and food_direction_y > 0:
            momentum += 0.3
        elif game.direction == Direction.UP and food_direction_y < 0:
            momentum += 0.3
            
        return min(momentum, 1.0)

    def get_state(self, game):
        """Wrapper để sử dụng enhanced state"""
        return self.get_enhanced_state(game)

    def get_action(self, state, training=True, game_state=None):
        """SIMPLE SAFE EXPLORATION - Back to 50% win rate version"""
        final_move = [0, 0, 0, 0]
        
        # Get Q-values from model
        state_tensor = torch.tensor(state, dtype=torch.float)
        q_values = self.model(state_tensor).detach().numpy()
        
        if training and random.random() < self.epsilon:
            # SIMPLE SAFE EXPLORATION - như version đạt 50% win rate
            if game_state is not None:
                # Get safe actions only
                head = game_state.head
                safe_actions = []
                
                # Check all 4 directions: [RIGHT, DOWN, LEFT, UP]
                directions = [(20, 0), (0, 20), (-20, 0), (0, -20)]
                for i, (dx, dy) in enumerate(directions):
                    from game_level1 import Point
                    next_pos = Point(head.x + dx, head.y + dy)
                    if not game_state.is_collision(next_pos):
                        safe_actions.append(i)
                
                # Choose random from safe actions - SIMPLE & EFFECTIVE
                if safe_actions:
                    move = random.choice(safe_actions)
                else:
                    move = random.randint(0, 3)  # Fallback
            else:
                # Pure random when no game state
                move = random.randint(0, 3)
        else:
            # Exploitation: follow Q-values
            move = np.argmax(q_values)
        
        final_move[move] = 1
        return final_move

    def remember_priority(self, state, action, reward, next_state, done):
        """IMPROVED: Lưu experiences với priority cho training tốt hơn"""
        experience = (state, action, reward, next_state, done)
        
        # Standard memory
        self.memory.append(experience)
        
        # Priority memory for important experiences
        if reward > 5:  # High reward = success
            self.priority_memory.append(experience)
        elif done and reward > 0:  # Episode end with positive reward
            self.priority_memory.append(experience)

    def train_with_enhanced_replay(self, batch_size=128):
        """IMPROVED: Training với enhanced replay mixing"""
        if len(self.memory) < batch_size:
            return
            
        # Mix different types of experiences for better learning
        normal_size = batch_size // 2
        priority_size = batch_size - normal_size
        
        batch = []
        
        # Normal experiences
        if len(self.memory) >= normal_size:
            batch.extend(random.sample(self.memory, normal_size))
        
        # Priority + Success experiences
        priority_and_success = list(self.priority_memory) + list(self.success_memory)
        if len(priority_and_success) >= priority_size:
            batch.extend(random.sample(priority_and_success, priority_size))
        elif len(priority_and_success) > 0:
            batch.extend(priority_and_success)
        
        if len(batch) == 0:
            return
            
        states, actions, rewards, next_states, dones = zip(*batch)
        self.trainer.train_step(states, actions, rewards, next_states, dones, self.target_model)

    def adaptive_epsilon_decay(self, win_rate):
        """AGGRESSIVE: Much faster epsilon decay for real learning"""
        if win_rate > 0.25:  # Excellent performance -> very fast decay
            decay_rate = 0.95   # Very fast
        elif win_rate > 0.15:  # Good performance -> fast decay
            decay_rate = 0.97   # Fast
        elif win_rate > 0.05:  # Some progress -> moderate decay
            decay_rate = 0.985  # Moderate
        elif win_rate > 0.01:  # Minimal progress -> slow decay
            decay_rate = 0.99   # Slow but still decreasing
        else:  # No progress -> still decay but slower
            decay_rate = 0.995  # Still decay to force exploitation
            
        # AGGRESSIVE: Much lower minimum epsilon for more exploitation
        self.epsilon = max(0.01, self.epsilon * decay_rate)  # Min 1% only
        print(f"🔄 Epsilon decay: {self.epsilon:.3f} (win_rate: {win_rate:.1%}, decay: {decay_rate})")

    def remember(self, state, action, reward, next_state, done):
        """Lưu experience vào memory"""
        self.memory.append((state, action, reward, next_state, done))

    def remember_success(self, episode_memory, final_score):
        """EXPERT: Lưu successful episodes để replay"""
        if final_score >= 4:  # Good episodes (reached checkpoint 4+)
            self.success_memory.extend(episode_memory)
            self.success_count += 1

    def train_step(self, state, action, reward, next_state, done):
        """Train model với một step"""
        self.trainer.train_step([state], [action], [reward], [next_state], [done], self.target_model)

    def train_replay(self, batch_size=1000):
        """Train model với batch từ memory"""
        if len(self.memory) < batch_size:
            return
            
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        self.trainer.train_step(states, actions, rewards, next_states, dones, self.target_model)
        
    def train_replay_mini(self, batch_size=64):
        """Train model với smaller batch mỗi step cho better learning"""
        if len(self.memory) < batch_size:
            return
            
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        self.trainer.train_step(states, actions, rewards, next_states, dones, self.target_model)
    
    def train_with_success_replay(self, batch_size=128):
        """EXPERT: Train với success experiences"""
        if len(self.memory) < batch_size:
            return
            
        # 50% normal memory, 50% success memory
        normal_size = batch_size // 2
        success_size = batch_size - normal_size
        
        batch = []
        
        # Normal experiences  
        if len(self.memory) >= normal_size:
            batch.extend(random.sample(self.memory, normal_size))
        
        # Success experiences
        if len(self.success_memory) >= success_size:
            batch.extend(random.sample(self.success_memory, success_size))
        elif len(self.success_memory) > 0:
            batch.extend(random.sample(self.success_memory, len(self.success_memory)))
        
        if len(batch) == 0:
            return
            
        states, actions, rewards, next_states, dones = zip(*batch)
        self.trainer.train_step(states, actions, rewards, next_states, dones, self.target_model)

    def save_model(self, filepath):
        """Lưu model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'n_games': self.n_games,
            'epsilon': self.epsilon
        }, filepath)
        
    def load_model(self, filepath):
        """Load model"""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.n_games = checkpoint.get('n_games', 0)
            self.epsilon = checkpoint.get('epsilon', 0.1)
            print(f"✅ Loaded model from {filepath}")
        else:
            print(f"⚠️ Model file {filepath} not found")

    def save_experience(self, filepath):
        """Lưu kinh nghiệm (memory)"""
        with open(filepath, 'wb') as f:
            pickle.dump(list(self.memory), f)
        print(f"💾 Saved experience to {filepath}")
        
    def load_experience(self, filepath):
        """Load kinh nghiệm (memory)"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                memory_list = pickle.load(f)
                self.memory = deque(memory_list, maxlen=100000)
            print(f"✅ Loaded experience from {filepath}, size: {len(self.memory)}")
        else:
            print(f"⚠️ Experience file {filepath} not found")

    def _get_closest_enemy_distance(self, game):
        """Tính khoảng cách đến enemy gần nhất"""
        head = game.head
        min_dist = 200.0  # Max distance
        
        if hasattr(game, 'active_enemies') and game.active_enemies:
            for enemy in game.active_enemies:
                if hasattr(enemy, 'rect2'):
                    dist = np.sqrt((enemy.rect2.centerx - head.x)**2 + (enemy.rect2.centery - head.y)**2)
                    min_dist = min(min_dist, dist)
        elif hasattr(game, 'enemies') and game.enemies:
            for enemy in game.enemies:
                if hasattr(enemy, 'rect2'):
                    dist = np.sqrt((enemy.rect2.centerx - head.x)**2 + (enemy.rect2.centery - head.y)**2)
                    min_dist = min(min_dist, dist)
        
        return min_dist
    
    def _predict_enemy_collision_risk(self, game):
        """Dự đoán risk collision với enemies"""
        head = game.head
        risk_score = 0.0
        
        enemies = game.active_enemies if hasattr(game, 'active_enemies') else game.enemies if hasattr(game, 'enemies') else []
        
        for enemy in enemies:
            if hasattr(enemy, 'rect2'):
                enemy_x, enemy_y = enemy.rect2.centerx, enemy.rect2.centery
                # Simple collision prediction
                if abs(enemy_x - head.x) < 60 and abs(enemy_y - head.y) < 60:
                    risk_score += 1.0
        
        return min(risk_score / 4.0, 1.0)  # Normalize
    
    def _get_safe_direction_count(self, game):
        """Đếm số hướng an toàn có thể đi"""
        head = game.head
        BLOCK_SIZE = 20
        
        directions = [
            Point(head.x + BLOCK_SIZE, head.y),  # RIGHT
            Point(head.x, head.y + BLOCK_SIZE),  # DOWN  
            Point(head.x - BLOCK_SIZE, head.y),  # LEFT
            Point(head.x, head.y - BLOCK_SIZE),  # UP
        ]
        
        safe_count = 0
        for point in directions:
            if not game.is_collision(point):
                safe_count += 1
                
        return safe_count / 4.0  # Normalize to 0-1


class GameEvaluator:
    def __init__(self):
        self.evaluation_results = []
        
    def evaluate_agent(self, agent, game_class, level_name, num_games=10):
        """Đánh giá agent trên một level"""
        results = {
            'level': level_name,
            'algorithm': agent.algorithm,
            'games_won': 0,
            'total_games': num_games,
            'total_scores': [],
            'steps_to_win': [],
            'avg_score': 0,
            'avg_steps_when_won': 0,
            'win_rate': 0
        }
        
        print(f"🎮 Evaluating {agent.algorithm} on {level_name} ({num_games} games)...")
        
        scores_detail = []
        for game_idx in range(num_games):
            game = game_class()
            score = 0
            steps = 0
            
            while True:
                state = agent.get_state(game)
                action = agent.get_action(state, training=False)  # No exploration
                reward, done, current_score = game.play_step(action)
                score = current_score
                steps += 1
                
                if done:
                    break
                    
                # Tránh vòng lặp vô hạn
                if steps > 1000:
                    break
            
            results['total_scores'].append(score)
            scores_detail.append(score)
            
            # Kiểm tra thắng game (score > threshold)
            win_threshold = self._get_win_threshold(level_name)
            if score >= win_threshold:
                results['games_won'] += 1
                results['steps_to_win'].append(steps)
                print(f"  🏆 WIN! Game {game_idx+1}: Score {score} in {steps} steps")
            
            game.quit() if hasattr(game, 'quit') else None
        
        # Tính toán kết quả
        results['avg_score'] = np.mean(results['total_scores'])
        results['win_rate'] = results['games_won'] / num_games
        if results['steps_to_win']:
            results['avg_steps_when_won'] = np.mean(results['steps_to_win'])
        
        # Debug info
        print(f"  📊 Scores: {scores_detail}")
        print(f"  🎯 Wins: {results['games_won']}/{num_games} = {results['win_rate']:.1%}")
        
        return results
    
    def _get_win_threshold(self, level_name):
        """Định nghĩa threshold để coi là thắng game"""
        # Thắng game = ăn hết food theo sequence
        thresholds = {
            'Level1': 6,   # Có 6 food positions trong sequence
            'Level2': 6,   # Cập nhật theo user setting 
            'Level3': 1    # Cập nhật theo user setting - dễ test
        }
        return thresholds.get(level_name, 6)

    def is_game_won(self, game, level_name):
        """Kiểm tra xem đã thắng game chưa (ăn hết food)"""
        win_threshold = self._get_win_threshold(level_name)
        return game.score >= win_threshold


class TrainingManager:
    def __init__(self):
        self.results_dir = 'training_results'
        self.models_dir = 'saved_models'
        self.experiences_dir = 'saved_experiences'
        self.plots_dir = 'plots'
        
        # Tạo directories
        for dir_path in [self.results_dir, self.models_dir, self.experiences_dir, self.plots_dir]:
            os.makedirs(dir_path, exist_ok=True)
            
        self.evaluator = GameEvaluator()
        
    def train_single_level(self, level_name='Level1', algorithm='DQN', total_episodes=5000, eval_interval=1000, eval_games=100):
        """IMPROVED: Train với persistent best model tracking"""
        
        # Game selection
        game_classes = {
            'Level1': Level1AI,
            'Level2': Level2AI,
            'Level3': Level3AI
        }
        
        if level_name not in game_classes:
            print(f"❌ Level {level_name} không hợp lệ. Chọn: {list(game_classes.keys())}")
            return None
            
        game_class = game_classes[level_name]
        
        # Create agent and game
        agent = AdvancedAgent(algorithm=algorithm)
        game = game_class()
        
        # File paths
        model_path = f"{self.models_dir}/{algorithm}_{level_name}.pth"
        exp_path = f"{self.experiences_dir}/{algorithm}_{level_name}.pkl"
        best_stats_path = f"{self.results_dir}/{algorithm}_{level_name}_best_stats.json"
        
        # FIXED: Load existing model và experience nếu có
        if os.path.exists(model_path):
            agent.load_model(model_path)
            print(f"✅ Loaded existing model from {model_path}")
        
        if os.path.exists(exp_path):
            agent.load_experience(exp_path)
            print(f"✅ Loaded existing experience from {exp_path}")
        
        print(f"🎯 Training {algorithm} on {level_name}")
        print(f"📊 Episodes: {total_episodes}, Eval interval: {eval_interval}")
        print(f"🎮 Starting from episode: {agent.n_games}")
        
        # Training metrics will be loaded from previous results above
        
        # FIXED: Persistent best model tracking - Load previous best stats
        best_win_rate = 0.0
        best_model_episode = 0
        previous_best_info = ""
        
        if os.path.exists(best_stats_path):
            try:
                with open(best_stats_path, 'r') as f:
                    best_stats = json.load(f)
                    best_win_rate = best_stats.get('best_win_rate', 0.0)
                    best_model_episode = best_stats.get('best_model_episode', 0)
                    previous_best_info = f" (Previous best: {best_win_rate:.2%} at episode {best_model_episode})"
                    print(f"📊 Loaded previous best stats: {best_win_rate:.2%} win rate at episode {best_model_episode}")
            except Exception as e:
                print(f"⚠️ Could not load best stats: {e}")
                best_win_rate = 0.0
                
        print(f"🏆 Current best threshold: {best_win_rate:.2%}{previous_best_info}")
        
        # FIXED: Load previous training results for complete plotting
        episode_scores = []
        evaluation_results = []
        last_eval_win_rate = 0.0  # Track last evaluation win rate
        results_file = f"{self.results_dir}/{algorithm}_{level_name}_results.json"
        
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    previous_results = json.load(f)
                    episode_scores = previous_results.get('episode_scores', [])
                    evaluation_results = previous_results.get('evaluation_results', [])
                    
                    # CRITICAL FIX: Load last evaluation win rate for epsilon decay
                    if evaluation_results:
                        last_eval_win_rate = evaluation_results[-1]['win_rate']
                        print(f"🔄 Loaded last eval win rate: {last_eval_win_rate:.1%} for epsilon decay")
                    
                    print(f"📈 Loaded previous training data: {len(episode_scores)} episodes, {len(evaluation_results)} evaluations")
            except Exception as e:
                print(f"⚠️ Could not load previous results: {e}")
                episode_scores = []
                evaluation_results = []
                last_eval_win_rate = 0.0
        else:
            print(f"📊 Starting fresh training - no previous data found")
        
        episode = agent.n_games
        start_episode = episode
        
        while episode < start_episode + total_episodes:
            # Training episode
            score = self._train_episode(agent, game)
            episode_scores.append(score)
            episode += 1
            
            # AGGRESSIVE: Epsilon decay từ version đạt 50% win rate
            if episode % 100 == 0:
                agent.adaptive_epsilon_decay(last_eval_win_rate)
            
            # IMPROVED progress report
            if episode % 100 == 0:
                recent_avg = np.mean(episode_scores[-100:]) if len(episode_scores) >= 100 else np.mean(episode_scores)
                success_rate = agent.success_count / max(episode - start_episode, 1) * 100
                print(f"Episode {episode}: Avg={recent_avg:.1f}, Success={success_rate:.1f}%, ε={agent.epsilon:.3f}")
            
            # Evaluation
            if episode % eval_interval == 0:
                print(f"📊 Episode {episode}, evaluating...")
                
                # Always save latest model và experience
                agent.save_model(model_path)
                agent.save_experience(exp_path)
                
                # Evaluate với nhiều games hơn để catch wins
                eval_result = self.evaluator.evaluate_agent(agent, game_class, level_name, max(eval_games, 20))
                eval_result['episode'] = episode
                evaluation_results.append(eval_result)
                
                current_win_rate = eval_result['win_rate']
                last_eval_win_rate = current_win_rate  # Update for epsilon decay
                print(f"Win Rate: {current_win_rate:.2%}, Avg Score: {eval_result['avg_score']:.1f}")
                if eval_result['avg_steps_when_won'] > 0:
                    print(f"Avg Steps to Win: {eval_result['avg_steps_when_won']:.1f}")
                
                # 🏆 FIXED: Persistent best model saving - only when truly better
                if current_win_rate > best_win_rate:
                    print(f"🎉 NEW RECORD! {current_win_rate:.2%} > {best_win_rate:.2%}")
                    best_win_rate = current_win_rate
                    best_model_episode = episode
                    
                    # Save best model with special name
                    best_model_path = f"{self.models_dir}/{algorithm}_{level_name}_BEST.pth"
                    best_exp_path = f"{self.experiences_dir}/{algorithm}_{level_name}_BEST.pkl"
                    
                    agent.save_model(best_model_path)
                    agent.save_experience(best_exp_path)
                    
                    # FIXED: Save best stats persistently
                    best_stats = {
                        'best_win_rate': best_win_rate,
                        'best_model_episode': best_model_episode,
                        'algorithm': algorithm,
                        'level_name': level_name,
                        'timestamp': datetime.now().isoformat(),
                        'avg_score': eval_result['avg_score'],
                        'games_won': eval_result['games_won'],
                        'total_games': eval_result['total_games']
                    }
                    
                    with open(best_stats_path, 'w') as f:
                        json.dump(best_stats, f, indent=2)
                    
                    print(f"🏆 NEW BEST MODEL! Win Rate: {current_win_rate:.2%} (Episode {episode})")
                    print(f"📁 Best model saved to: {best_model_path}")
                    print(f"💾 Best stats saved to: {best_stats_path}")
                    
                    # EXPERT SUCCESS CRITERIA
                    if current_win_rate >= 0.6:  # 60% win rate = EXPERT level!
                        print(f"🎉 EXPERT LEVEL ACHIEVED! Win rate ≥60%!")
                        print(f"🛑 Stopping to prevent overtraining")
                        break
                    elif current_win_rate >= 0.3:  # 30% = very good
                        print(f"✅ VERY GOOD! Win rate ≥30% - Continue for EXPERT level")
                else:
                    print(f"📊 Current best: {best_win_rate:.2%} (Episode {best_model_episode}) - Need {best_win_rate - current_win_rate:.2%} improvement")
                
                # STABLE: Frequent soft updates for consistent long-term learning
                if episode % max(eval_interval // 2, 100) == 0:  # Every 500 episodes for stability
                    agent.soft_update_target(tau=0.005)  # Soft update prevents catastrophic forgetting
        
        # Save final results
        final_results = {
            'algorithm': algorithm,
            'level': level_name,
            'episode_scores': episode_scores,
            'evaluation_results': evaluation_results,
            'total_episodes': total_episodes,
            'start_episode': start_episode,
            'final_episode': episode,
            'best_win_rate': best_win_rate,
            'best_model_episode': best_model_episode
        }
        
        # Save results to file
        results_file = f"{self.results_dir}/{algorithm}_{level_name}_results.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # Plot results
        self._plot_results(final_results)
        
        print(f"✅ Completed training: {algorithm} on {level_name}")
        print(f"📁 Results saved to: {results_file}")
        print(f"🤖 Latest model saved to: {model_path}")
        
        # Best model summary
        if best_win_rate > 0:
            best_model_path = f"{self.models_dir}/{algorithm}_{level_name}_BEST.pth"
            print(f"🏆 Best model saved to: {best_model_path}")
            print(f"📊 Best win rate: {best_win_rate:.2%} (Episode {best_model_episode})")
        else:
            print(f"⚠️ No good model found (best win rate: 0%)")
        
        # Cleanup
        if hasattr(game, 'quit'):
            game.quit()
        
        return final_results
    
    def _train_episode(self, agent, game):
        """RAINBOW: Train episode với enhanced techniques"""
        game.reset()
        total_score = 0
        episode_memory = []
        step_count = 0
        
        while True:
            # Get current state
            state_old = agent.get_state(game)
            
            # Get action with enhanced selection
            action = agent.get_action(state_old, training=True, game_state=game)
            
            # Perform action
            reward, done, score = game.play_step(action)
            total_score = score
            step_count += 1
            
            # Get new state
            state_new = agent.get_state(game)
            
            # RAINBOW: Enhanced reward shaping
            enhanced_reward = self._shape_reward(reward, score, step_count, done)
            
            # Store experience with priority
            experience = (state_old, action, enhanced_reward, state_new, done)
            episode_memory.append(experience)
            
            # IMPROVED: Use enhanced memory storage
            agent.remember_priority(state_old, action, enhanced_reward, state_new, done)
            
            # IMPROVED: More frequent training for faster learning
            if step_count % 2 == 0:  # Train every 2 steps for faster learning
                agent.train_with_enhanced_replay(batch_size=32)
            
            if done:
                break
        
        # IMPROVED: Store successful episodes for better learning
        agent.remember_success(episode_memory, total_score)
        
        # IMPROVED: Final enhanced training at episode end
        agent.train_with_enhanced_replay(batch_size=128)
        
        agent.n_games += 1
        
        return total_score
    
    def _shape_reward(self, original_reward, score, step_count, done):
        """IMPROVED: Enhanced reward shaping for better learning"""
        shaped_reward = original_reward
        
        # Reward shaping based on score progress
        if original_reward > 0:  # Food eaten
            # Bonus for higher scores (later checkpoints worth more)
            checkpoint_bonus = score * 0.5
            shaped_reward += checkpoint_bonus
            
            # Efficiency bonus (fewer steps = better)
            if step_count < 50:  # Fast completion
                shaped_reward += 2.0
            elif step_count < 100:
                shaped_reward += 1.0
                
        # Penalty for time wasting
        if step_count > 200 and score == 0:  # Taking too long without progress
            shaped_reward -= 0.1
            
        # Completion bonus
        if done and score >= 6:  # Completed full sequence
            shaped_reward += 10.0
        elif done and score >= 4:  # Good progress
            shaped_reward += 5.0
        elif done and score >= 2:  # Some progress
            shaped_reward += 2.0
            
        return shaped_reward
    
    def _plot_results(self, results):
        """Vẽ biểu đồ kết quả"""
        algorithm = results['algorithm']
        level = results['level']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{algorithm} - {level} Training Results', fontsize=16)
        
        # Episode scores
        episodes = range(len(results['episode_scores']))
        ax1.plot(episodes, results['episode_scores'], alpha=0.6)
        ax1.set_title('Episode Scores')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Score')
        
        # Moving average of scores
        window = 100
        if len(results['episode_scores']) >= window:
            moving_avg = np.convolve(results['episode_scores'], np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(results['episode_scores'])), moving_avg, 'r-', linewidth=2, label='Moving Average')
            ax1.legend()
        
        # Evaluation metrics
        eval_episodes = [r['episode'] for r in results['evaluation_results']]
        win_rates = [r['win_rate'] for r in results['evaluation_results']]
        avg_scores = [r['avg_score'] for r in results['evaluation_results']]
        
        ax2.plot(eval_episodes, win_rates, 'g-o', linewidth=2)
        ax2.set_title('Win Rate Over Time')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Win Rate')
        ax2.set_ylim(0, 1)
        
        ax3.plot(eval_episodes, avg_scores, 'b-o', linewidth=2)
        ax3.set_title('Average Score Over Time')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Average Score')
        
        # Steps to win (chỉ show những game thắng)
        steps_to_win = []
        episodes_with_wins = []
        for r in results['evaluation_results']:
            if r['avg_steps_when_won'] > 0:
                steps_to_win.append(r['avg_steps_when_won'])
                episodes_with_wins.append(r['episode'])
        
        if steps_to_win:
            ax4.plot(episodes_with_wins, steps_to_win, 'r-o', linewidth=2)
            ax4.set_title('Average Steps to Win (When Won)')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Steps to Win')
        else:
            ax4.text(0.5, 0.5, 'No wins recorded', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Average Steps to Win (When Won)')
        
        plt.tight_layout()
        plot_file = f"{self.plots_dir}/{algorithm}_{level}_training.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Plot saved to {plot_file}")
    
    def _compare_all_results(self, all_results):
        """So sánh kết quả của tất cả algorithms và levels"""
        print("\n📊 COMPARISON SUMMARY")
        print("=" * 80)
        
        comparison_data = []
        
        for key, results in all_results.items():
            if results['evaluation_results']:
                final_eval = results['evaluation_results'][-1]
                comparison_data.append({
                    'Algorithm': results['algorithm'],
                    'Level': results['level'],
                    'Final Win Rate': final_eval['win_rate'],
                    'Final Avg Score': final_eval['avg_score'],
                    'Avg Steps to Win': final_eval['avg_steps_when_won'] if final_eval['avg_steps_when_won'] > 0 else 'N/A',
                    'Games Won': final_eval['games_won']
                })
        
        # In bảng so sánh
        print(f"{'Algorithm':<12} {'Level':<8} {'Win Rate':<10} {'Avg Score':<10} {'Steps to Win':<12} {'Games Won':<10}")
        print("-" * 80)
        
        for data in comparison_data:
            steps_str = f"{data['Avg Steps to Win']:.1f}" if data['Avg Steps to Win'] != 'N/A' else 'N/A'
            print(f"{data['Algorithm']:<12} {data['Level']:<8} {data['Final Win Rate']:<10.2%} {data['Final Avg Score']:<10.1f} {steps_str:<12} {data['Games Won']:<10}")
        
        # Tìm best performer cho mỗi level
        print("\n🏆 BEST PERFORMERS BY LEVEL")
        print("=" * 50)
        
        levels = ['Level1', 'Level2', 'Level3']
        for level in levels:
            level_results = [d for d in comparison_data if d['Level'] == level]
            if level_results:
                # Sắp xếp theo win rate, sau đó theo avg score
                best = max(level_results, key=lambda x: (x['Final Win Rate'], x['Final Avg Score']))
                print(f"{level}: {best['Algorithm']} (Win Rate: {best['Final Win Rate']:.2%}, Score: {best['Final Avg Score']:.1f})")
        
        # Vẽ biểu đồ so sánh
        self._plot_comparison(comparison_data)
    
    def _plot_comparison(self, comparison_data):
        """Vẽ biểu đồ so sánh tất cả algorithms và levels""" 
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Algorithm Comparison Across All Levels', fontsize=16)
        
        # Chuẩn bị data cho plotting
        algorithms = ['DQN', 'Double_DQN']
        levels = ['Level1', 'Level2', 'Level3']
        
        # Win rates
        win_rates_dqn = []
        win_rates_ddqn = []
        avg_scores_dqn = []
        avg_scores_ddqn = []
        
        for level in levels:
            dqn_data = next((d for d in comparison_data if d['Algorithm'] == 'DQN' and d['Level'] == level), None)
            ddqn_data = next((d for d in comparison_data if d['Algorithm'] == 'Double_DQN' and d['Level'] == level), None)
            
            win_rates_dqn.append(dqn_data['Final Win Rate'] if dqn_data else 0)
            win_rates_ddqn.append(ddqn_data['Final Win Rate'] if ddqn_data else 0)
            avg_scores_dqn.append(dqn_data['Final Avg Score'] if dqn_data else 0)
            avg_scores_ddqn.append(ddqn_data['Final Avg Score'] if ddqn_data else 0)
        
        x = np.arange(len(levels))
        width = 0.35
        
        # Win rates comparison
        ax1.bar(x - width/2, win_rates_dqn, width, label='DQN (Improved)', alpha=0.8)
        ax1.bar(x + width/2, win_rates_ddqn, width, label='Double DQN (Improved)', alpha=0.8)
        ax1.set_xlabel('Level')
        ax1.set_ylabel('Win Rate')
        ax1.set_title('Win Rate Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(levels)
        ax1.legend()
        
        # Average scores comparison
        ax2.bar(x - width/2, avg_scores_dqn, width, label='DQN (Improved)', alpha=0.8)
        ax2.bar(x + width/2, avg_scores_ddqn, width, label='Double DQN (Improved)', alpha=0.8)
        ax2.set_xlabel('Level')
        ax2.set_ylabel('Average Score')
        ax2.set_title('Average Score Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(levels)
        ax2.legend()
        
        # Steps to win comparison (chỉ những case có thắng)
        steps_dqn = []
        steps_ddqn = []
        for level in levels:
            dqn_data = next((d for d in comparison_data if d['Algorithm'] == 'DQN' and d['Level'] == level), None)
            ddqn_data = next((d for d in comparison_data if d['Algorithm'] == 'Double_DQN' and d['Level'] == level), None)
            
            steps_dqn.append(dqn_data['Avg Steps to Win'] if dqn_data and dqn_data['Avg Steps to Win'] != 'N/A' else 0)
            steps_ddqn.append(ddqn_data['Avg Steps to Win'] if ddqn_data and ddqn_data['Avg Steps to Win'] != 'N/A' else 0)
        
        # Lọc bỏ những level không có data
        valid_levels = []
        valid_steps_dqn = []
        valid_steps_ddqn = []
        for i, level in enumerate(levels):
            if steps_dqn[i] > 0 or steps_ddqn[i] > 0:
                valid_levels.append(level)
                valid_steps_dqn.append(steps_dqn[i])
                valid_steps_ddqn.append(steps_ddqn[i])
        
        if valid_levels:
            x_valid = np.arange(len(valid_levels))
            ax3.bar(x_valid - width/2, valid_steps_dqn, width, label='DQN (Improved)', alpha=0.8)
            ax3.bar(x_valid + width/2, valid_steps_ddqn, width, label='Double DQN (Improved)', alpha=0.8)
            ax3.set_xlabel('Level')
            ax3.set_ylabel('Average Steps to Win')
            ax3.set_title('Steps to Win Comparison (Lower is Better)')
            ax3.set_xticks(x_valid)
            ax3.set_xticklabels(valid_levels)
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'No wins recorded for comparison', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Steps to Win Comparison')
        
        # Overall performance radar (total wins)
        total_wins_dqn = sum(d['Games Won'] for d in comparison_data if d['Algorithm'] == 'DQN')
        total_wins_ddqn = sum(d['Games Won'] for d in comparison_data if d['Algorithm'] == 'Double_DQN')
        
        algorithms_for_pie = ['DQN (Improved)', 'Double DQN (Improved)']
        wins_for_pie = [total_wins_dqn, total_wins_ddqn]
        
        if sum(wins_for_pie) > 0:
            ax4.pie(wins_for_pie, labels=algorithms_for_pie, autopct='%1.1f%%', startangle=90)
            ax4.set_title(f'Total Wins Distribution\n(Total: {sum(wins_for_pie)} wins)')
        else:
            ax4.text(0.5, 0.5, 'No wins recorded', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Total Wins Distribution')
        
        plt.tight_layout()
        comparison_file = f"{self.plots_dir}/algorithms_comparison.png"
        plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Comparison plot saved to {comparison_file}")


if __name__ == "__main__":
    print("🚀 IMPROVED SNAKE AI TRAINING SYSTEM")
    print("🎯 TARGET: 70-80% WIN RATE")
    print("=" * 60)
    print("🧠 Features: 22 enhanced inputs")
    print("🏗️ Architecture: Enhanced 256→128 network")
    print("📚 Strategy: Smart replay + Priority experiences")
    print("🔍 Exploration: Adaptive epsilon + Smart exploration")
    print("🎁 Rewards: Enhanced reward shaping")
    print("=" * 60)
    
    # Tạo training manager
    manager = TrainingManager()
    
    # Algorithm selection
    import sys
    if len(sys.argv) > 1:
        algorithm = sys.argv[1]
    else:
        algorithm = 'DQN'  # Default to improved DQN
    
    print(f"🚀 Selected Algorithm: {algorithm}")
    
    # IMPROVED training setup
    results = manager.train_single_level(
        level_name='Level1',      # Train Level1
        algorithm=algorithm,      # Use improved algorithm  
        total_episodes=3000,      # IMPROVED: Efficient training
        eval_interval=250,        # IMPROVED: Frequent evaluation
        eval_games=100            # Thorough evaluation
    )
    
    print(f"\n🎉 {algorithm} Training completed!")
    
    # IMPROVED performance assessment
    if results and results['evaluation_results']:
        final_eval = results['evaluation_results'][-1]
        final_win_rate = final_eval['win_rate']
        
        if final_win_rate >= 0.7:
            print(f"🌟 MASTER STATUS! Win rate: {final_win_rate:.1%}")
            print(f"🎮 Ready for gameplay: python play_trained_agent.py")
        elif final_win_rate >= 0.6:
            print(f"🏆 EXPERT STATUS ACHIEVED! Win rate: {final_win_rate:.1%}")
            print(f"🎮 Ready for gameplay: python play_trained_agent.py")
        elif final_win_rate >= 0.4:
            print(f"✅ VERY GOOD! Win rate: {final_win_rate:.1%}")
            print(f"💡 Continue training for EXPERT level")
        elif final_win_rate >= 0.2:
            print(f"⚠️ GOOD PROGRESS! Win rate: {final_win_rate:.1%}")
            print(f"🔄 Run again for better performance")
        else:
            print(f"❌ NEEDS MORE WORK! Win rate: {final_win_rate:.1%}")
            print(f"💡 Try adjusting hyperparameters or more training")
    
    print(f"\n📁 Check results in:")
    print(f"- Models: {manager.models_dir}/")
    print(f"- Results: {manager.results_dir}/")
    print(f"- Plots: {manager.plots_dir}/")
    
    print(f"\n🎯 Usage:")
    print(f"- Train Improved DQN: python advanced_trainer.py DQN")
    print(f"- Train Improved Double DQN: python advanced_trainer.py Double_DQN")