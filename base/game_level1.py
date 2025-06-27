import os
os.environ["SDL_VIDEODRIVER"] = "windib"
import pygame
import math
from enum import Enum
from collections import namedtuple, deque
import numpy as np
from pytmx.util_pygame import load_pygame
from player import Player
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

class Level1AI:

    def __init__(self, w=1320, h=720):  # Updated size to match TMX map (21*64, 11*64)
        self.w = w
        self.h = h
        # Optimized spawn point - gần hơn với main corridor
        self.spawnpoint_x = 128  # Gần entrance hơn
        self.spawnpoint_y = 530  # Aligned với corridor
        # Optimized food position - ngắn path hơn nhưng vẫn challenging
        self.food_x = 340  # Gần hơn nhưng vẫn trong safe zone
        self.food_y = 530   # Aligned với corridor
        
        # init display
        self.screen = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Worlds Hardest Game - Level 1')
        self.clock = pygame.time.Clock()
        self.hascollided = False
        
        # Load TMX map
        self.tmx_data = load_pygame('Tiles/level1.tmx')
        self.sprite_group = pygame.sprite.Group()
        self.tile_rect = []
        self.collider_rects = []  # List of tile colliders from TMX
        self.setup_tiles()
        
        # ALL ENEMIES ACTIVE với speed = 5
        self.enemy = Enemy(640, 285, 14, 14, True, False)
        self.enemy2 = Enemy(640, 415, 14, 14, True, False)
        self.enemy3 = Enemy(640, 350, 14, 14, True, False, False)
        self.enemy4 = Enemy(640, 480, 14, 14, True, False, False)
        
        # Tất cả enemies đều active với speed = 5
        self.active_enemies = [self.enemy, self.enemy2, self.enemy3, self.enemy4]
        
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
        # init game state
        self.enemy = Enemy(640, 285, 14, 14, True, False)
        self.enemy2 = Enemy(640, 415, 14, 14, True, False)
        self.enemy3 = Enemy(640, 350, 14, 14, True, False, False)
        self.enemy4 = Enemy(640, 480, 14, 14, True, False, False)
        self.set_enemy_speeds()
        
        # Tất cả enemies đều active với speed = 5
        self.active_enemies = [self.enemy, self.enemy2, self.enemy3, self.enemy4]
        self.direction = Direction.RIGHT
        self.head = Point(self.spawnpoint_x, self.spawnpoint_y)
        self.snake = [self.head]
        self.score = 0
        self.food_x = 340 #340
        self.food_y = 530
        self.food = None
        self._spawn_new_food()
        self.frame_iteration = 0
        # Reset visited positions
        self.visited = set()
        # Reset step count for this episode
        self.steps_in_episode = 0
    def _spawn_new_food(self):
        """Spawn food theo thứ tự cố định như heat map để AI học đường đi tối ưu"""
        
        # Danh sách food theo thứ tự tối ưu (heat map path)
        food_sequence = [
            (340, 530),   # Checkpoint 1 - Start area
            (340, 400),   # Checkpoint 2 - Move up  
            (704, 320),   # Checkpoint 3 - Enter danger zone
            (896, 320),   # Checkpoint 4 - Navigate through enemies
            (914, 200),   # Checkpoint 5 - Safe zone
            (1106, 200),  # Checkpoint 6 - Goal area
        ]
        
        # Lấy food theo thứ tự (score bắt đầu từ 1 sau khi ăn food đầu tiên)
        if self.score < 6 :
            self.food_x, self.food_y = food_sequence[self.score]
        else:
            # Nếu vượt quá sequence, giữ ở vị trí cuối
            self.food_x, self.food_y = food_sequence[-1]
        
        self.food = Point(self.food_x, self.food_y)
        self.food_rect = pygame.Rect(self.food_x, self.food_y, BLOCK_SIZE, BLOCK_SIZE)

    def set_enemy_speeds(self):
        """Set tất cả enemies speed = 4 - balanced với agent speed 10"""
        self.enemy.enemy_speed = 4
        self.enemy2.enemy_speed = 4
        self.enemy3.enemy_speed = 4
        self.enemy4.enemy_speed = 4

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
        
        # SIMPLE REWARD SYSTEM cho 4 enemies speed = 5
        reward = -0.003  # Base penalty
        game_over = False
        
        # Timeout cho sequential checkpoint system
        base_timeout = 1500  # Base timeout cho 6 checkpoints
        checkpoint_bonus = self.score * 150  # Bonus time cho mỗi checkpoint đạt được
        timeout_limit = int(base_timeout + checkpoint_bonus)
        
        if self.frame_iteration > timeout_limit:
            game_over = True
            reward = -8  # Smaller timeout penalty
            return reward, game_over, self.score
        
        # Enemy collision penalty - khó tránh hơn với 4 enemies
        if self.is_collision_enemy():
            game_over = True
            reward = -12  # Reasonable penalty cho 4 enemies
            return reward, game_over, self.score
        
        # Wall collision penalty
        if self.is_collision_wall():
            game_over = True
            reward = -6  # Smaller wall penalty
            return reward, game_over, self.score
        
        # SEQUENTIAL CHECKPOINT LOGIC - Food xuất hiện theo thứ tự như heat map
        if self.head_rect.colliderect(self.food_rect):
            self.score += 1
            
            # Checkpoint reward cho mỗi lần hoàn thành
            checkpoint_reward = 60  # Reward cho mỗi checkpoint
            reward += checkpoint_reward
            
            # Kiểm tra win condition (hoàn thành 6 checkpoints)
            if self.score >= 6:
                # WIN GAME - Đã hoàn thành tất cả checkpoints
                
                victory_bonus = 350  # Bonus lớn khi hoàn thành path
                time_bonus = max(0, (timeout_limit - self.frame_iteration) * 0.6)
                efficiency_bonus = max(0, (900 - self.steps_in_episode) * 0.25)
                
                total_victory_reward = victory_bonus + time_bonus + efficiency_bonus
                reward += total_victory_reward
                game_over = True
                self.snake.pop()
                return reward, game_over, self.score
            else:
                # Chưa hoàn thành, spawn checkpoint tiếp theo theo thứ tự
                self._spawn_new_food()
                self.snake.pop()
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
        
        # ENHANCED CHECKPOINT REWARDS cho 4 enemies
        checkpoint_bonus = self._get_checkpoint_bonus()
        reward += checkpoint_bonus
        
        # Mild penalty cho staying still
        if self.head == old_head:
            reward -= 4.0
        
        # SIMPLIFIED anti-oscillation
        self._track_movement_pattern()
        if hasattr(self, 'movement_history') and len(self.movement_history) >= 4:
            if self._is_simple_oscillating():
                reward -= 2.5
        
        # ENHANCED enemy proximity awareness cho 4 enemies
        enemy_bonus = self._get_enemy_awareness_bonus()
        reward += enemy_bonus
        
        self._update_ui()
        return reward, game_over, self.score

    def _get_checkpoint_bonus(self):
        """Reward cho reaching key map areas - với 4 enemies active"""
        x, y = self.head.x, self.head.y
        
        # Checkpoint areas trong map - khó hơn vì có 4 enemies
        if 300 <= x <= 500 and 200 <= y <= 500:  # Entered main corridor
            return 1.0  # Increased bonus vì có 4 enemies
        elif 500 <= x <= 750 and 200 <= y <= 500:  # Middle danger zone
            return 2.0  # Bonus lớn vì đây là danger zone 
        elif 750 <= x <= 950 and 200 <= y <= 500:  # Past enemies zone
            return 3.0  # Bonus rất lớn vì đã qua zone enemies
        elif 950 <= x <= 1200 and 150 <= y <= 350:  # Goal area
            return 5.0  # Bonus lớn nhất
        
        return 0.0

    def _get_enemy_awareness_bonus(self):
        """Enhanced bonus cho smart enemy avoidance - với 4 enemies"""
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
        
        # ENHANCED DANGER AWARENESS với 4 enemies
        bonus = 0.0
        
        # Reward for maintaining safe minimum distance
        if min_enemy_dist >= 80:  # Safe from closest enemy
            bonus += 0.4
        elif min_enemy_dist >= 60:  # Reasonable distance
            bonus += 0.2
        elif min_enemy_dist < 40:  # Too close to any enemy
            bonus -= 1.5  # Strong penalty
        
        # Bonus for good average distance from all enemies
        if avg_enemy_dist >= 100:
            bonus += 0.3
        elif avg_enemy_dist >= 80:
            bonus += 0.1
        
        # Extra bonus for navigating through enemy zone safely
        enemy_zone_x = 600 <= self.head.x <= 700  # Main enemy zone
        if enemy_zone_x and min_enemy_dist >= 70:
            bonus += 0.5  # Navigating safely through danger zone
        
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
        # pygame.draw.rect(self.screen, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        # Draw and move ALL 4 ENEMIES với optimized patterns
        self.enemy.move(345, 935)
        self.enemy2.move(345, 935)
        self.enemy3.move(345, 935)
        self.enemy4.move(345, 935)
        self.enemy.draw(self.screen)
        self.enemy2.draw(self.screen)
        self.enemy3.draw(self.screen)
        self.enemy4.draw(self.screen)
       
        # UI cho sequential checkpoint system
        text = font.render(f"Checkpoint: {self.score}/6 | Steps: {self.steps_in_episode}", True, WHITE)
        self.screen.blit(text, [0, 0])
        
        # Progress indicator based on checkpoint sequence
        checkpoint_progress = (self.score / 6) * 100
        progress_text = font.render(f"Path Progress: {checkpoint_progress:.0f}% | Sequential Training", True, WHITE)
        self.screen.blit(progress_text, [0, 30])
        
        # Current objective
        if self.score < 6:
            checkpoint_names = ["Start→Up", "Up→Center", "Center→Danger", "Danger→Through", "Through→Safe", "Safe→Goal"]
            current_objective = checkpoint_names[self.score] if self.score < len(checkpoint_names) else "Complete"
            obj_text = font.render(f"Next: {current_objective}", True, WHITE)
            self.screen.blit(obj_text, [0, 60])
        else:
            complete_text = font.render("Path Complete! Victory!", True, WHITE)
            self.screen.blit(complete_text, [0, 60])
        
        pygame.display.update()
        self.clock.tick(60)

    def _move(self, action):
        # [right, down, left, up]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        if np.array_equal(action, [1, 0, 0, 0]):
            new_dir = clock_wise[0] # right
        elif np.array_equal(action, [0, 1, 0, 0]):
            new_dir = clock_wise[1] # down
        elif np.array_equal(action, [0, 0, 1, 0]):
            new_dir = clock_wise[2] # left
        elif np.array_equal(action, [0, 0, 0, 1]):
            new_dir = clock_wise[3] # up
            
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