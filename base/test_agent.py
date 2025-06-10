import os
os.environ["SDL_VIDEODRIVER"] = "windib"
import torch
import numpy as np
from game_level1 import Level1AI
from model import Linear_QNet
import pygame

class TestAgent:
    def __init__(self, model_path='model/model.pth'):
        self.model = Linear_QNet(20, 512, 256, 128, 4)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
            print(f"✅ Model loaded from {model_path}")
        else:
            print(f"❌ Model không tìm thấy tại {model_path}")
    
    def get_state(self, game):
        """Copy state function từ Agent"""
        head = game.snake[0]
        SPEED = 15
        
        # Kiểm tra collision ở 4 hướng
        from game_level1 import Point, Direction
        import math
        
        point_l = Point(head.x - SPEED, head.y)
        point_r = Point(head.x + SPEED, head.y)
        point_u = Point(head.x, head.y - SPEED)
        point_d = Point(head.x, head.y + SPEED)
        
        # Hướng hiện tại
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        
        # Kiểm tra danger từ walls
        danger_straight = (dir_r and game.is_collision(point_r)) or \
                         (dir_l and game.is_collision(point_l)) or \
                         (dir_u and game.is_collision(point_u)) or \
                         (dir_d and game.is_collision(point_d))
        
        danger_right = (dir_u and game.is_collision(point_r)) or \
                      (dir_d and game.is_collision(point_l)) or \
                      (dir_l and game.is_collision(point_u)) or \
                      (dir_r and game.is_collision(point_d))
        
        danger_left = (dir_d and game.is_collision(point_r)) or \
                     (dir_u and game.is_collision(point_l)) or \
                     (dir_r and game.is_collision(point_u)) or \
                     (dir_l and game.is_collision(point_d))
        
        # Kiểm tra danger từ enemies ở các hướng
        danger_enemy_straight = (dir_r and game.is_collision_enemy_at(point_r)) or \
                               (dir_l and game.is_collision_enemy_at(point_l)) or \
                               (dir_u and game.is_collision_enemy_at(point_u)) or \
                               (dir_d and game.is_collision_enemy_at(point_d))
        
        danger_enemy_right = (dir_u and game.is_collision_enemy_at(point_r)) or \
                            (dir_d and game.is_collision_enemy_at(point_l)) or \
                            (dir_l and game.is_collision_enemy_at(point_u)) or \
                            (dir_r and game.is_collision_enemy_at(point_d))
        
        danger_enemy_left = (dir_d and game.is_collision_enemy_at(point_r)) or \
                           (dir_u and game.is_collision_enemy_at(point_l)) or \
                           (dir_r and game.is_collision_enemy_at(point_u)) or \
                           (dir_l and game.is_collision_enemy_at(point_d))
        
        # Khoảng cách đến food (normalized)
        food_distance = math.sqrt((game.food.x - head.x)**2 + (game.food.y - head.y)**2)
        food_distance_norm = food_distance / 1000.0
        
        # Hướng đến food
        dx = game.food.x - head.x
        dy = game.food.y - head.y
        if food_distance > 0:
            food_dir_x = dx / food_distance
            food_dir_y = dy / food_distance
        else:
            food_dir_x = food_dir_y = 0
        
        # Vị trí tương đối
        pos_x_norm = head.x / game.w
        pos_y_norm = head.y / game.h
        
        state = [
            danger_straight, danger_right, danger_left,
            danger_enemy_straight, danger_enemy_right, danger_enemy_left,
            dir_l, dir_r, dir_u, dir_d,
            game.food.x < head.x, game.food.x > head.x,
            game.food.y < head.y, game.food.y > head.y,
            food_distance_norm, food_dir_x, food_dir_y,
            pos_x_norm, pos_y_norm, game.frame_iteration / 500.0
        ]
        
        return np.array(state, dtype=float)
    
    def get_action(self, state):
        """Greedy action - không exploration"""
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        move = torch.argmax(prediction).item()
        
        final_move = [0, 0, 0, 0]
        final_move[move] = 1
        return final_move
    
    def test_difficulty(self, difficulty_level, num_tests=10):
        """Test agent với một difficulty level cụ thể"""
        game = Level1AI()
        game.set_difficulty(difficulty_level)
        
        wins = 0
        total_score = 0
        total_steps = 0
        
        print(f"\n🧪 Testing Difficulty Level {difficulty_level}...")
        
        for test in range(num_tests):
            game.reset()
            game_over = False
            steps = 0
            
            while not game_over and steps < 1000:  # Max 1000 steps per test
                state = self.get_state(game)
                action = self.get_action(state)
                reward, game_over, score = game.play_step(action)
                steps += 1
            
            if score >= 10:
                wins += 1
                print(f"  Test {test+1}: ✅ WIN (Score: {score}, Steps: {steps})")
            else:
                print(f"  Test {test+1}: ❌ FAIL (Score: {score}, Steps: {steps})")
            
            total_score += score
            total_steps += steps
        
        win_rate = wins / num_tests
        avg_score = total_score / num_tests
        avg_steps = total_steps / num_tests
        
        print(f"📊 Results for Difficulty {difficulty_level}:")
        print(f"   Win Rate: {win_rate:.1%} ({wins}/{num_tests})")
        print(f"   Average Score: {avg_score:.1f}")
        print(f"   Average Steps: {avg_steps:.0f}")
        
        return win_rate, avg_score, avg_steps

def main():
    print("🤖 Testing Trained DQN Agent")
    
    # Load agent
    agent = TestAgent()
    
    # Test all difficulty levels
    results = {}
    for difficulty in range(1, 5):
        results[difficulty] = agent.test_difficulty(difficulty, num_tests=5)
    
    # Summary
    print("\n" + "="*50)
    print("📈 SUMMARY RESULTS:")
    print("="*50)
    for diff, (win_rate, avg_score, avg_steps) in results.items():
        print(f"Difficulty {diff}: Win Rate {win_rate:.1%}, Avg Score {avg_score:.1f}, Avg Steps {avg_steps:.0f}")
    
    # Interactive mode
    print("\n🎮 Interactive Mode - Press ENTER to watch agent play...")
    input("Press ENTER to continue...")
    
    difficulty = int(input("Chọn difficulty level (1-4): ") or "1")
    game = Level1AI()
    game.set_difficulty(difficulty)
    
    print(f"🕹️  Watching agent play at difficulty {difficulty}...")
    print("Press ESC to quit")
    
    game.reset()
    game_over = False
    steps = 0
    
    while not game_over and steps < 2000:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    game_over = True
        
        state = agent.get_state(game)
        action = agent.get_action(state)
        reward, game_over, score = game.play_step(action)
        steps += 1
        
        pygame.time.delay(50)  # Slow down for viewing
    
    print(f"🏁 Game ended! Score: {score}, Steps: {steps}")
    pygame.quit()

if __name__ == '__main__':
    main() 