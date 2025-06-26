import os
os.environ["SDL_VIDEODRIVER"] = "windib"
import pygame
import math
from enum import Enum
from collections import namedtuple
import numpy as np
from pytmx.util_pygame import load_pygame
from enemy import Enemy
import cv2
from collections import deque

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

class Level2AI:

    def __init__(self, w=1320, h=720):
        self.w = w
        self.h = h
        # Spawn point và food position được tối ưu cho Level 2
<<<<<<< HEAD
        self.spawnpoint_x = 260
        self.spawnpoint_y = 332
        self.food_x = 300
        self.food_y = 404
=======
        self.spawnpoint_x = 140
        self.spawnpoint_y = 275
        self.food_x = 350
        self.food_y = 272
>>>>>>> 650bc2ed31d1fded9cd84a9dc89e68b26516b1d4
        # init display
        # self.display = pygame.display.set_mode((self.w, self.h))
        self.screen = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Worlds Hardest Game - Level 2')
        self.clock = pygame.time.Clock()
        self.hascollided = False
        self.reset()
        # Enemies
        # for tiles
        self.tmx_data = load_pygame('Tiles/Level2.tmx')
        self.sprite_group = pygame.sprite.Group()
        self.tile_rect = []
        self.collider_rects = [] # List of tile colliders
        self.can_move_left = True
        self.can_move_up = True
        self.can_move_down = True
        self.setup_tiles()
        # Values
        
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
        for rect in self.collider_rects:
            pygame.draw.rect(self.screen, (0, 0, 0), rect, 2)
    
    
    def reset(self):
        
        # init game state - khởi tạo lại enemies
        self.direction = Direction.RIGHT
        
        # Tối ưu: Sử dụng list thay vì 12 biến riêng biệt
        self.enemies = [
            Enemy(350, 156, 14, 14, 5, (0, 0, 255)),  # enemy1
            Enemy(414, 484, 14, 14, 5, (0, 0, 255)),  # enemy2
            Enemy(478, 156, 14, 14, 5, (0, 0, 255), False),  # enemy3
            Enemy(542, 484, 14, 14, 5, (0, 0, 255), False),  # enemy4
            Enemy(606, 156, 14, 14, 5, (0, 0, 255), False),  # enemy5
            Enemy(670, 484, 14, 14, 5, (0, 0, 255), False),  # enemy6
            Enemy(734, 156, 14, 14, 5, (0, 0, 255), False),  # enemy7
            Enemy(798, 484, 14, 14, 5, (0, 0, 255), False),  # enemy8
            Enemy(862, 156, 14, 14, 5, (0, 0, 255), False),  # enemy9
            Enemy(926, 484, 14, 14, 5, (0, 0, 255), False),  # enemy10
            Enemy(990, 156, 14, 14, 5, (0, 0, 255), False),  # enemy11
            Enemy(1054, 484, 14, 14, 5, (0, 0, 255), False),  # enemy12
        ]
        
        # Set enemy speeds
<<<<<<< HEAD
        self.set_enemy_speeds(2)
=======
        self.set_enemy_speeds(3)
>>>>>>> 650bc2ed31d1fded9cd84a9dc89e68b26516b1d4
        self.head = Point(self.spawnpoint_x, self.spawnpoint_y)
        self.snake = [self.head]
        self.score = 0
        self.food = None
        self.food_x = 300 #340
        self.food_y = 436
        self._spawn_new_food()
        self.frame_iteration = 0
        self.visited = set()

<<<<<<< HEAD
    def set_enemy_speeds(self, speed=2):
=======
    def set_enemy_speeds(self, speed=3):
>>>>>>> 650bc2ed31d1fded9cd84a9dc89e68b26516b1d4
        """Set speed cho tất cả enemies - balanced với agent speed 10"""
        for enemy in self.enemies:
            enemy.enemy_speed = speed
    
    def get_enemy_count(self):
        """Trả về số lượng enemies hiện tại"""
        return len(self.enemies)

    def _spawn_new_food(self):
        """Spawn food theo thứ tự cố định như heat map cho Level 2 - khó hơn Level 1"""
        
        # Danh sách food theo thứ tự tối ưu cho Level 2 (heat map)
        food_sequence = [
<<<<<<< HEAD
            (340, 404),   # Checkpoint 1 - Goal area (bắt đầu ở goal)
=======
            (340, 436),   # Checkpoint 1 - Goal area (bắt đầu ở goal)
>>>>>>> 650bc2ed31d1fded9cd84a9dc89e68b26516b1d4
            (512, 436),    # Checkpoint 2 - Middle danger zone  
            (772, 372),    # Checkpoint 3 - Navigate through enemies
            (896, 355),    # Checkpoint 4 - High danger area
            (1006, 340),    # Checkpoint 5 - Upper safe zone
            (1138, 285),    # Checkpoint 6 - Return to start area (harder path)
        ]
        
        # Lấy food theo thứ tự (score bắt đầu từ 0)
        if self.score < len(food_sequence):
            self.food_x, self.food_y = food_sequence[self.score]
        else:
            # Nếu vượt quá sequence, giữ ở vị trí cuối
            self.food_x, self.food_y = food_sequence[-1]
        
        self.food = Point(self.food_x, self.food_y)
        self.food_rect = pygame.Rect(self.food_x, self.food_y, BLOCK_SIZE, BLOCK_SIZE)

    def play_step(self, action):
        self.frame_iteration += 1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # Lưu thông tin cũ cho reward calculation
        old_head = self.head
        old_dist = math.sqrt((self.food.x - old_head.x)**2 + (self.food.y - old_head.y)**2)
        
        # Di chuyển
        self._move(action)
        self.head_rect = pygame.Rect(self.head.x, self.head.y, BLOCK_SIZE, BLOCK_SIZE)
        self.snake.insert(0, Point(self.head.x, self.head.y))
        
<<<<<<< HEAD
        # SIMPLIFIED reward system cho Level 2 - dễ học hơn
        reward = -0.001  # Phạt base nhẹ hơn để AI dễ học
        game_over = False
        
        # Timeout cho Level 2 sequential checkpoint system (khó hơn Level 1)
        base_timeout = 3000  # Base timeout cao hơn cho Level 2 với 12 enemies
=======
        # Enhanced reward system cho Level 2 với 12 enemies speed = 5
        reward = -0.05  # Phạt base cao hơn do Level 2 khó hơn với 12 enemies
        game_over = False
        
        # Timeout cho Level 2 sequential checkpoint system (khó hơn Level 1)
        base_timeout = 2000  # Base timeout cao hơn cho Level 2 với 12 enemies
>>>>>>> 650bc2ed31d1fded9cd84a9dc89e68b26516b1d4
        checkpoint_bonus = self.score * 200  # Bonus time lớn cho mỗi checkpoint (Level 2 khó hơn)
        max_timeout = int(base_timeout + checkpoint_bonus)
        
        # Kiểm tra timeout với adaptive threshold
        if self.frame_iteration > max_timeout:
            game_over = True
            reward = -30  # Phạt timeout nặng hơn
            return reward, game_over, self.score
        
        # Kiểm tra va chạm enemy với proximity warning - khó hơn với 12 enemies
        enemy_collision, min_enemy_dist = self._check_enemy_collision_with_distance()
        if enemy_collision:
            game_over = True
<<<<<<< HEAD
            reward = -20  # Phạt va chạm enemy rất nặng cho Level 2 với 12 enemies
=======
            reward = -50  # Phạt va chạm enemy rất nặng cho Level 2 với 12 enemies
>>>>>>> 650bc2ed31d1fded9cd84a9dc89e68b26516b1d4
            return reward, game_over, self.score
        
        # Kiểm tra va chạm tường
        if self.is_collision():
            game_over = True
            reward = -25  # Phạt va chạm tường
            return reward, game_over, self.score
        
        # SEQUENTIAL CHECKPOINT LOGIC cho Level 2 - Food xuất hiện theo thứ tự
        if self.head_rect.colliderect(self.food_rect):
            self.score += 1
            
<<<<<<< HEAD
            # MASSIVE reward cho mỗi checkpoint để AI học nhanh
            checkpoint_reward = 100  # Reward rất cao để AI ưu tiên ăn food
=======
            # Checkpoint reward cho mỗi lần hoàn thành (Level 2 khó hơn)
            checkpoint_reward = 80  # Reward cao hơn Level 1 vì có 12 enemies
>>>>>>> 650bc2ed31d1fded9cd84a9dc89e68b26516b1d4
            reward += checkpoint_reward
            
            # Kiểm tra win condition (hoàn thành 6 checkpoints trong Level 2)
            if self.score >= 6:
                # WIN GAME - Đã hoàn thành tất cả checkpoints Level 2
<<<<<<< HEAD
                self.score = 6
=======
                self.score = 10  # Tự động đặt score = 10 khi hoàn thành Level 2
>>>>>>> 650bc2ed31d1fded9cd84a9dc89e68b26516b1d4
                
                victory_bonus = 500  # Bonus rất lớn vì Level 2 với 12 enemies
                time_bonus = max(0, (max_timeout - self.frame_iteration) * 0.8)
                efficiency_bonus = max(0, (1200 - self.frame_iteration) * 0.3)
                survival_bonus = 150  # Bonus lớn cho việc sống sót với 12 enemies
                
                total_victory_reward = victory_bonus + time_bonus + efficiency_bonus + survival_bonus
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
        
        # Progressive movement rewards
        new_dist = math.sqrt((self.food.x - self.head.x)**2 + (self.food.y - self.head.y)**2)
        
<<<<<<< HEAD
        # SIMPLIFIED distance-based reward
        if new_dist < old_dist:
            progress_reward = min(5.0, (old_dist - new_dist) / 10.0)  # Thưởng tiến bộ cao hơn
            reward += progress_reward
        else:
            retreat_penalty = -min(0.5, (new_dist - old_dist) / 50.0)  # Phạt lùi lại nhẹ hơn
=======
        # Distance-based reward (scaled for Level 2)
        if new_dist < old_dist:
            progress_reward = min(1.5, (old_dist - new_dist) / 20.0)  # Thưởng tiến bộ
            reward += progress_reward
        else:
            retreat_penalty = -min(1.0, (new_dist - old_dist) / 25.0)  # Phạt lùi lại
>>>>>>> 650bc2ed31d1fded9cd84a9dc89e68b26516b1d4
            reward += retreat_penalty
        
        # Enhanced exploration reward cho Level 2
        if not hasattr(self, 'visited'):
            self.visited = set()
        pos_key = (int(self.head.x / 40), int(self.head.y / 40))  # Grid dày hơn cho Level 2
        if pos_key not in self.visited:
            reward += 0.2  # Thưởng khám phá cao hơn
            self.visited.add(pos_key)
        elif len(self.visited) > 20:  # Phạt quay lại vùng cũ sau khi đã khám phá nhiều
            reward -= 0.1
        
        # Enemy danger awareness rewards cho Level 2
        danger_bonus = self._calculate_danger_awareness_bonus(min_enemy_dist)
        reward += danger_bonus
        
        # Anti-oscillation penalty cho Level 2
        self._track_movement_pattern()
        if hasattr(self, 'movement_history') and len(self.movement_history) >= 6:
            if self._is_oscillating():
                reward -= 1.5  # Phạt dao động mạnh hơn Level 1
        
        # Phạt đứng yên
        if self.head == old_head:
            reward -= 2.0  # Phạt nặng hơn cho Level 2
        
        # Strategic positioning bonus (gần safe paths)
        safe_bonus = self._calculate_safe_positioning_bonus()
        reward += safe_bonus
        
        self._update_ui()
        return reward, game_over, self.score

    def _check_enemy_collision_with_distance(self):
        """Kiểm tra va chạm enemy và trả về distance tới enemy gần nhất"""
        head_rect = pygame.Rect(self.head.x, self.head.y, BLOCK_SIZE, BLOCK_SIZE)
        
        min_distance = float('inf')
        collision = False
        
        for enemy in self.enemies:
            if head_rect.colliderect(enemy.rect2):
                collision = True
            
            # Tính distance tới enemy center
            dist = math.sqrt((enemy.rect2.centerx - self.head.x)**2 + 
                           (enemy.rect2.centery - self.head.y)**2)
            min_distance = min(min_distance, dist)
        
        return collision, min_distance

    def _calculate_danger_awareness_bonus(self, min_enemy_dist):
        """Tính thưởng dựa trên việc tránh enemy một cách thông minh"""
        if min_enemy_dist == float('inf'):
            return 0
        
        # Thưởng khi ở khoảng cách an toàn nhưng không quá xa
        if 80 <= min_enemy_dist <= 120:  # Sweet spot cho Level 2
            return 0.3
        elif 50 <= min_enemy_dist < 80:  # Gần nhưng vẫn an toàn
            return 0.1
        elif min_enemy_dist < 40:  # Quá gần, nguy hiểm
            return -0.5
        elif min_enemy_dist > 200:  # Quá xa, không hiệu quả
            return -0.1
        
        return 0

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

    def _is_oscillating(self):
        """Kiểm tra xem agent có đang dao động không"""
        if len(self.movement_history) < 6:
            return False
        
        # Kiểm tra pattern dao động đơn giản
        recent_moves = list(self.movement_history)[-6:]
        
        # Pattern: A-B-A-B-A-B hoặc tương tự
        if (recent_moves[0] == recent_moves[2] == recent_moves[4] and
            recent_moves[1] == recent_moves[3] == recent_moves[5] and
            recent_moves[0] != recent_moves[1]):
            return True
        
        # Pattern: quay lại vị trí cũ
        if (recent_moves[-1] == 'LEFT' and recent_moves[-2] == 'RIGHT') or \
           (recent_moves[-1] == 'RIGHT' and recent_moves[-2] == 'LEFT') or \
           (recent_moves[-1] == 'UP' and recent_moves[-2] == 'DOWN') or \
           (recent_moves[-1] == 'DOWN' and recent_moves[-2] == 'UP'):
            return True
        
        return False

    def _calculate_safe_positioning_bonus(self):
        """Thưởng khi ở vị trí strategically safe cho Level 2"""
        head_x, head_y = self.head.x, self.head.y
        
        # Kiểm tra khoảng cách đến boundaries
        dist_to_left = head_x
        dist_to_right = self.w - head_x
        dist_to_top = head_y  
        dist_to_bottom = self.h - head_y
        
        min_boundary_dist = min(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)
        
        # Thưởng khi không quá gần boundary (cho Level 2 cần space để maneuver)
        if 100 <= min_boundary_dist <= 200:
            return 0.1
        elif min_boundary_dist < 50:
            return -0.2  # Phạt khi quá gần boundary
        
        return 0

    def is_collision_enemy(self):
        """Kiểm tra va chạm với enemy"""
        head_rect = pygame.Rect(self.head.x, self.head.y, BLOCK_SIZE, BLOCK_SIZE)
        
        for enemy in self.enemies:
            if head_rect.colliderect(enemy.rect2):
                return True
        return False
    
    def is_collision_wall(self, pt=None):
        """Kiểm tra va chạm với tường hoặc biên map"""
        if pt is None:
            pt = self.head
        
        # Kiểm tra va chạm biên map
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        
        # Kiểm tra tự cắn
        if pt in self.snake[1:]:
            return True
        
        # Kiểm tra va chạm với tile colliders (nếu có)
        head_rect = pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)
        for rect in self.collider_rects:
            if head_rect.colliderect(rect):
                return True
        
        return False

    def is_collision(self, pt=None):
        """Hàm collision tổng hợp cho tương thích với code cũ"""
        return self.is_collision_wall(pt) or (pt is None and self.is_collision_enemy())

    def _update_ui(self):
        
        self.screen.fill("white")
        self.sprite_group.update()
        self.sprite_group.draw(self.screen)
        for pt in self.snake:
            pygame.draw.rect(self.screen, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.screen, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
        # Tối ưu: Move và draw tất cả enemies bằng loop
        for enemy in self.enemies:
            enemy.move2(155, 484)
        for enemy in self.enemies:
            enemy.draw(self.screen)


        # Enhanced UI cho Level 2 sequential checkpoint system
        text = font.render(f"Level 2 - Checkpoint: {self.score}/6 | 12 Enemies", True, WHITE)
        self.screen.blit(text, [0, 0])
        
        # Progress indicator cho Level 2
        checkpoint_progress = (self.score / 6) * 100
        progress_text = font.render(f"Level 2 Progress: {checkpoint_progress:.0f}% | Advanced Training", True, WHITE)
        self.screen.blit(progress_text, [0, 30])
        
        # Current objective cho Level 2 (khó hơn Level 1)
        if self.score < 6:
            checkpoint_names = [
                "Goal→Danger", "Danger→Navigate", "Navigate→High Risk", 
                "Risk→Upper Safe", "Safe→Return", "Return→Complete"
            ]
            current_objective = checkpoint_names[self.score] if self.score < len(checkpoint_names) else "Complete"
            obj_text = font.render(f"Next: {current_objective} | Enemies: {len(self.enemies)}", True, WHITE)
            self.screen.blit(obj_text, [0, 60])
        else:
            complete_text = font.render("Level 2 Complete! Master Level!", True, WHITE)
            self.screen.blit(complete_text, [0, 60])
            
        # Show frame count và difficulty indicator
        frame_text = font.render(f"Frame: {self.frame_iteration} | Difficulty: HARD", True, WHITE)
        self.screen.blit(frame_text, [0, 90])
        # pygame.draw.rect(self.screen, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
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
        
        # Di chuyển theo hướng đã chọn
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