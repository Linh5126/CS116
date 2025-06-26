import os
os.environ["SDL_VIDEODRIVER"] = "windib"
import pygame
import math
from enum import Enum
from collections import namedtuple, deque
import numpy as np
from pytmx.util_pygame import load_pygame
from enemy import Enemy
import cv2

pygame.init()
font = pygame.font.Font('arial.ttf', 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)
BLOCK_SIZE = 32
SPEED = 10
reward = 0

class Level3AI:

    def __init__(self, w=1320, h=720):  # Updated size to match TMX map (21*64, 11*64)
        self.w = w
        self.h = h
        # Optimized spawn point - gần hơn với main corridor
        self.spawnpoint_x = 675  # Gần entrance hơn
        self.spawnpoint_y = 375  # Aligned với corridor 
        # Optimized food position - ngắn path hơn nhưng vẫn challenging
        self.food_x = 590  # Gần hơn nhưng vẫn trong safe zone
        self.food_y = 210   # Aligned với corridor
        
        # init display
        self.screen = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Worlds Hardest Game - Level 3')
        self.clock = pygame.time.Clock()
        self.hascollided = False
        
        # Load TMX map
        self.tmx_data = load_pygame('Tiles/level3.tmx')
        self.sprite_group = pygame.sprite.Group()
        self.tile_rect = []
        self.collider_rects = []  # List of tile colliders from TMX
        self.setup_tiles()
        
        # Khởi tạo enemies với speed = 5
        self.enemies = []
        self.active_enemies = []  # Sẽ được set trong reset()
        
        self.reset()

    class Tile(pygame.sprite.Sprite):
        def __init__(self, pos, surf, groups):
            super().__init__(groups)
            self.image = surf
            self.rect = self.image.get_rect(topleft=pos)

    def setup_tiles(self):
        """Setup tiles from TMX map"""
        for layer in self.tmx_data.visible_layers:
            if hasattr(layer, 'data'):
                # Vẽ tất cả tile (main, background)
                for x_val, y_val, surf in layer.tiles():
                    if surf is not None:
                        pos = (x_val * 64, y_val * 64)
                        self.Tile(pos=pos, surf=surf, groups=self.sprite_group)
                        self.tile_rect.append(pygame.Rect(x_val * 64, y_val * 64, 64, 64))
                
                # Thêm collider cho layer "Wall"
                if layer.name.lower() == "wall":
                    for x_val, y_val in enumerate_tiles(layer):
                        tile_id = layer.data[y_val][x_val]
                        if tile_id != 0:  # Tile khác 0 là tường
                            wall_rect = pygame.Rect(x_val * 64, y_val * 64, 64, 64)
                            self.collider_rects.append(wall_rect)

    def drawColliders(self):
        """Debug function to draw colliders"""
        for rect in self.collider_rects:
            pygame.draw.rect(self.screen, (255, 0, 0), rect, 2)
    
    def reset(self):
        # init game state - đặt enemies ở vị trí strategic
        self.enemies = [
            Enemy(608, 285, 14, 14, direction=1),
            #Enemy(608, 350, 14, 14, direction=4),
            Enemy(608, 415, 14, 14, direction=4),
            #Enemy(608, 480, 14, 14, direction=4),
            Enemy(672, 480, 14, 14, direction=3),
            #Enemy(736, 480, 14, 14, direction=3),
            Enemy(800, 480, 14, 14, direction=2),
            #Enemy(800, 415, 14, 14, direction=2),
            Enemy(800, 350, 14, 14, direction=2),
            #Enemy(800, 285, 14, 14, direction=2),
            Enemy(736, 285, 14, 14, direction=1)
        ]
        self.set_enemy_speeds()
        
        # Tất cả enemies đều active
        self.active_enemies = self.enemies
        
        self.direction = Direction.RIGHT
        self.head = Point(self.spawnpoint_x, self.spawnpoint_y)
        self.snake = [self.head]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        # Reset visited positions
        self.visited = set()
        # Reset step count for this episode
        self.steps_in_episode = 0

    def _place_food(self):
        self.food = Point(self.food_x, self.food_y)
        self.food_rect = pygame.Rect(self.food_x, self.food_y, BLOCK_SIZE, BLOCK_SIZE)
        
        if self.food in self.snake:
            self._place_food()

    def set_enemy_speeds(self):
        for enemy in self.enemies:
            enemy.enemy_speed = 5  # Speed cao hơn cho Level 3 - balanced với agent speed 10

    def play_step(self, action):
        self.frame_iteration += 1
        self.steps_in_episode += 1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # Lưu thông tin cũ
        old_head = self.head
        old_dist = math.sqrt((self.food.x - old_head.x)**2 + (self.food.y - old_head.y)**2)
        
        # Di chuyển
        self._move(action)
        self.head_rect = pygame.Rect(self.head.x, self.head.y, BLOCK_SIZE, BLOCK_SIZE)
        self.snake.insert(0, Point(self.head.x, self.head.y))
        
        # REWARD SYSTEM cho 6 enemies với pattern cụ thể
        reward = -0.0025  # Base penalty điều chỉnh cho 6 enemies
        game_over = False
        
        # Fixed timeout cho single difficulty
        base_timeout = 650  # Fixed timeout
        distance_factor = old_dist / 200  
        timeout_limit = int(base_timeout + distance_factor)
        
        if self.frame_iteration > timeout_limit:
            game_over = True
            reward = -8  # Smaller timeout penalty
            return reward, game_over, self.score
        
        # Enemy collision penalty - trung bình với 6 enemies có pattern
        if self.is_collision_enemy():
            game_over = True
            reward = -11  # Penalty vừa phải cho 6 enemies có pattern
            return reward, game_over, self.score
        
        # Wall collision penalty
        if self.is_collision_wall():
            game_over = True
            reward = -6  # Smaller wall penalty
            return reward, game_over, self.score
        
        # WIN CONDITION - Bonus cho 6 enemies với pattern strategic
        if self.head_rect.colliderect(self.food_rect):
            self.score += 10
            # WIN REWARDS điều chỉnh cho 6 enemies
            base_reward = 190  # Reward tăng lên phù hợp với 6 enemies
            time_bonus = max(0, (timeout_limit - self.frame_iteration) * 0.2)
            efficiency_bonus = max(0, (600 - self.steps_in_episode) * 0.1)
            survival_bonus = 45  # Bonus tăng cho 6 enemies
            
            total_reward = base_reward + time_bonus + efficiency_bonus + survival_bonus
            reward = total_reward
            game_over = True
            self.snake.pop()
            return reward, game_over, self.score
        else:
            self.snake.pop()
        
        # ENHANCED DISTANCE-BASED REWARDS
        new_dist = math.sqrt((self.food.x - self.head.x)**2 + (self.food.y - self.head.y)**2)
        
        # Progress rewards
        if new_dist < old_dist:
            progress_ratio = (old_dist - new_dist) / max(old_dist, 1)
            progress_reward = progress_ratio * 6.0  # Good signal
            reward += progress_reward
        else:
            retreat_ratio = (new_dist - old_dist) / max(old_dist, 1)
            retreat_penalty = retreat_ratio * 1.0  # Mild penalty
            reward -= retreat_penalty
        
        # EXPLORATION SYSTEM - encouraging smart paths
        grid_size = 100  
        pos_key = (int(self.head.x / grid_size), int(self.head.y / grid_size))
        
        if not hasattr(self, 'visited'):
            self.visited = set()
            
        if pos_key not in self.visited:
            exploration_bonus = 0.8  # Fixed exploration bonus
            reward += exploration_bonus
            self.visited.add(pos_key)
        
        # ENHANCED CHECKPOINT REWARDS cho 6 enemies với pattern
        checkpoint_bonus = self._get_checkpoint_bonus()
        reward += checkpoint_bonus
        
        # Strong penalty cho staying still
        if self.head == old_head:
            reward -= 4.0
        
        # SIMPLIFIED anti-oscillation
        self._track_movement_pattern()
        if hasattr(self, 'movement_history') and len(self.movement_history) >= 4:
            if self._is_simple_oscillating():
                reward -= 2.5
        
        # ENHANCED enemy proximity awareness cho 6 enemies với pattern
        enemy_bonus = self._get_enemy_awareness_bonus()
        reward += enemy_bonus
        
        self._update_ui()
        return reward, game_over, self.score

    def _get_checkpoint_bonus(self):
        """Reward cho reaching key map areas - với 6 enemies có pattern cụ thể"""
        x, y = self.head.x, self.head.y
        
        # Checkpoint areas dựa trên enemy positions và strategic paths
        if 300 <= x <= 580 and 200 <= y <= 500:  # Approach left enemies zone  
            return 1.0  # Entry bonus
        elif 580 <= x <= 650 and 280 <= y <= 420:  # Navigate through left enemies (608,285 & 608,415)
            return 2.0  # Navigating main danger zone
        elif 650 <= x <= 750 and 280 <= y <= 500:  # Middle passage area
            return 1.5  # Safe middle zone
        elif 750 <= x <= 850 and 280 <= y <= 500:  # Navigate right enemies zone (800,480 & 800,350)
            return 2.5  # Right danger zone navigation
        elif 500 <= x <= 650 and 200 <= y <= 300:  # Approach goal area (food at 590,210)
            return 3.0  # Near goal bonus
        
        return 0.0

    def _get_enemy_awareness_bonus(self):
        """Enhanced bonus cho smart enemy avoidance - với 6 enemies có pattern"""
        if not hasattr(self, 'active_enemies'):
            return 0.0
            
        # Calculate distances to all active enemies
        enemy_distances = []
        for enemy in self.active_enemies:
            enemy_dist = math.sqrt((self.head.x - enemy.rect2.centerx)**2 + 
                                 (self.head.y - enemy.rect2.centery)**2)
            enemy_distances.append(enemy_dist)
        
        if not enemy_distances:
            return 0.0
        
        min_enemy_dist = min(enemy_distances)
        avg_enemy_dist = sum(enemy_distances) / len(enemy_distances)
        
        # ENHANCED DANGER AWARENESS với 6 enemies có pattern cụ thể
        bonus = 0.0
        
        # Reward for maintaining safe minimum distance với 6 enemies
        if min_enemy_dist >= 75:  # Safe from closest enemy
            bonus += 0.35
        elif min_enemy_dist >= 55:  # Reasonable distance 
            bonus += 0.2
        elif min_enemy_dist < 40:  # Too close to any enemy
            bonus -= 1.2  # Penalty tăng cho 6 enemies
        
        # Bonus for good average distance với 6 enemies
        if avg_enemy_dist >= 95:
            bonus += 0.25
        elif avg_enemy_dist >= 75:
            bonus += 0.15
        
        # Special bonus cho việc navigate qua dense enemy areas
        # Check if in main enemy corridor (608,285 to 608,415) or right side (800,350 to 800,480)
        in_left_danger = 580 <= self.head.x <= 630 and 270 <= self.head.y <= 430
        in_right_danger = 780 <= self.head.x <= 820 and 340 <= self.head.y <= 490
        
        if (in_left_danger or in_right_danger) and min_enemy_dist >= 65:
            bonus += 0.4  # High bonus for safely navigating dense areas
        
        return bonus

    def _is_simple_oscillating(self):
        """Simplified oscillation detection"""
        if len(self.movement_history) < 4:
            return False
        
        recent = list(self.movement_history)[-4:]
        
        # Simple back-and-forth pattern
        if (recent[-1] == 'LEFT' and recent[-2] == 'RIGHT') or \
           (recent[-1] == 'RIGHT' and recent[-2] == 'LEFT') or \
           (recent[-1] == 'UP' and recent[-2] == 'DOWN') or \
           (recent[-1] == 'DOWN' and recent[-2] == 'UP'):
            return True
        
        return False

    def is_collision_enemy(self):
        """Kiểm tra va chạm với active enemies"""
        head_rect = pygame.Rect(self.head.x, self.head.y, BLOCK_SIZE, BLOCK_SIZE)
        for enemy in self.active_enemies:
            if enemy.enemy_speed > 0 and head_rect.colliderect(enemy.rect2):
                return True
        return False

    def is_collision_wall(self, pt=None):
        """Kiểm tra va chạm với tường sử dụng TMX colliders"""
        if pt is None:
            pt = self.head
        
        # Kiểm tra va chạm biên map
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        
        # Kiểm tra tự cắn
        if pt in self.snake[1:]:
            return True
        
        # Kiểm tra va chạm với TMX wall colliders
        head_rect = pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)
        for rect in self.collider_rects:
            if head_rect.colliderect(rect):
                return True
        
        return False

    def is_collision(self, pt=None):
        """Compatibility function - use wall collision for general collision"""
        return self.is_collision_wall(pt)

    def is_collision_enemy_at(self, pt):
        """Kiểm tra va chạm enemy tại một điểm cụ thể"""
        test_rect = pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)
        for enemy in self.active_enemies:
            if enemy.enemy_speed > 0 and test_rect.colliderect(enemy.rect2):
                return True
        return False

    def _track_movement_pattern(self):
        """Track movement pattern để detect oscillation"""
        if not hasattr(self, 'movement_history'):
            self.movement_history = deque(maxlen=10)
            self.previous_head = self.head
        
        # Record movement direction
        if hasattr(self, 'previous_head'):
            dx = self.head.x - self.previous_head.x
            dy = self.head.y - self.previous_head.y
            
            if dx > 0:
                direction = 'RIGHT'
            elif dx < 0:
                direction = 'LEFT'
            elif dy > 0:
                direction = 'DOWN'
            elif dy < 0:
                direction = 'UP'
            else:
                direction = 'STAY'
            
            self.movement_history.append(direction)
        
        self.previous_head = self.head

    def _update_ui(self):
        self.screen.fill("white")
        # Draw TMX tiles
        self.sprite_group.update()
        self.sprite_group.draw(self.screen)
        
        # Draw snake
        for pt in self.snake:
            pygame.draw.rect(self.screen, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.screen, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        # Draw food
        pygame.draw.rect(self.screen, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        # Draw and move ALL enemies với strategic patterns
        for enemy in self.enemies:
            enemy.move3(608, 800, 285, 480)  # Adjusted movement boundaries
        for enemy in self.enemies:
            enemy.draw(self.screen)
        
        # Collision detection được handle trong play_step(), không cần ở đây
        # Simple UI 
        text = font.render(f"Score: {self.score} | Steps: {self.steps_in_episode} | {len(self.active_enemies)} Pattern Enemies", True, WHITE)
        self.screen.blit(text, [0, 0])
        
        # Progress indicator
        progress = min(100, (self.frame_iteration / 650) * 100)
        progress_text = font.render(f"Progress: {progress:.1f}%", True, WHITE)
        self.screen.blit(progress_text, [0, 30])
        
        pygame.display.update()
        self.clock.tick(60)

    def _move(self, action):
        # [right, left, up, down] - FIXED to match advanced_trainer.py!
        action_dirs = [Direction.RIGHT, Direction.LEFT, Direction.UP, Direction.DOWN]
        if np.array_equal(action, [1, 0, 0, 0]):
            new_dir = action_dirs[0] # RIGHT
        elif np.array_equal(action, [0, 1, 0, 0]):
            new_dir = action_dirs[1] # LEFT (FIXED!)
        elif np.array_equal(action, [0, 0, 1, 0]):
            new_dir = action_dirs[2] # UP (FIXED!)
        elif np.array_equal(action, [0, 0, 0, 1]):
            new_dir = action_dirs[3] # DOWN (FIXED!)
            
        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        
        # Move according to direction
        if self.direction == Direction.RIGHT:
            x += SPEED
        elif self.direction == Direction.LEFT:
            x -= SPEED
        elif self.direction == Direction.DOWN:
            y += SPEED
        elif self.direction == Direction.UP:
            y -= SPEED
        self.head = Point(x, y)
        self.head_rect = pygame.Rect(x, y, BLOCK_SIZE, BLOCK_SIZE)
    
    def get_frame(self):
        surface = pygame.display.get_surface()
        frame = pygame.surfarray.array3d(surface)  # (width, height, 3)
        frame = np.transpose(frame, (1, 0, 2))     # → (height, width, 3)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Đúng màu cho OpenCV
        return frame

def enumerate_tiles(layer):
    """Helper function to enumerate tiles in a layer"""
    for y_val, row in enumerate(layer.data):
        for x_val, tile_id in enumerate(row):
            yield x_val, y_val