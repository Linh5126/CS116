import pygame
import math
from enum import Enum
from collections import namedtuple
import numpy as np
from pytmx.util_pygame import load_pygame
from player import Player
from enemy import Enemy

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
SPEED = 15
reward = 0

class Level2AI:

    def __init__(self, w=1320, h=720):
        self.w = w
        self.h = h
        
        self.spawnpoint_x = 140
        self.spawnpoint_y = 275
        self.food_x = 300
        self.food_y = 436
        # init display
        # self.display = pygame.display.set_mode((self.w, self.h))
        self.screen = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Worlds Hardest Game')
        self.clock = pygame.time.Clock()
        self.hascollided = False
        self.reset()
        # Enemies
        self.enemy = Enemy(350, 155, 14, 14, 5, (0, 0, 255))
        self.enemy2 = Enemy(414, 484, 14, 14, 5, (0, 0, 255))
        self.enemy3 = Enemy(478, 155, 14, 14, 5, (0, 0, 255), False)
        self.enemy4 = Enemy(542, 484, 14, 14, 5, (0, 0, 255), False)
        self.enemy5 = Enemy(606, 155, 14, 14, 5, (0, 0, 255), False)
        self.enemy6 = Enemy(670, 484, 14, 14, 5, (0, 0, 255), False)
        self.enemy7 = Enemy(734, 155, 14, 14, 5, (0, 0, 255), False)
        self.enemy8 = Enemy(798, 484, 14, 14, 5, (0, 0, 255), False)
        self.enemy9 = Enemy(862, 155, 14, 14, 5, (0, 0, 255), False)
        self.enemy10 = Enemy(926, 484, 14, 14, 5, (0, 0, 255), False)
        self.enemy11 = Enemy(990, 155, 14, 14, 5, (0, 0, 255), False)
        self.enemy12 = Enemy(1054, 484, 14, 14, 5, (0, 0, 255), False)
        # Colliders
        self.toprect = pygame.Rect(0, 0, 300, 320)
        self.collidedtopleft = True
        self.top1rect = pygame.Rect(300, 128, 768, 40)
        self.collidedtop1= False
        self.bottomleftrect = pygame.Rect(0, 384, 320, 50)
        self.collidedbottomleft = False
        # self.left1rect = pygame.Rect(250, 200, 50, 300)
        self.collidedleft1 = False
        self.left2rect = pygame.Rect(50, 200, 256, 400)
        self.collidedleft2 = False
        # self.left3rect = pygame.Rect(300, 200, 25, 300)
        self.collidedleft3 = False
        self.bot1rect = pygame.Rect(250, 500, 70, 20)
        self.collidedbot1 = False
        self.bot2rect = pygame.Rect(320, 505, 792, 20)
        self.collidedbot2 = False
        self.bot3rect = pygame.Rect(374, 512, 60, 70)
        self.collidedbot3 = False
        self.right1rect = pygame.Rect(1088, 384, 25, 128)
        self.collidedright1 = False
        self.right2rect = pygame.Rect(900, 170, 320, 30)
        self.collidedright2 = False
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
        for layer in self.tmx_data.visible_layers:
            if hasattr(layer, 'data'):
                if layer.name == "Main":
                    for x_val, y_val, surf in layer.tiles():
                        pos = (x_val * 64, y_val * 64)
                        self.Tile(pos=pos, surf=surf, groups=self.sprite_group)
                        self.tile_rect.append(pygame.Rect(x_val * 64, y_val * 64, 64, 64))
                        self.collider_rects.append(pygame.Rect(x_val * 64, y_val * 64, 64, 64))

        for layer in self.tmx_data.visible_layers:
            if hasattr(layer, 'data'):
                for x_val, y_val, surf in layer.tiles():
                    pos = (x_val * 64, y_val * 64)
                    self.Tile(pos=pos, surf=surf, groups=self.sprite_group)

    def drawColliders(self):
        pass
    
    def reset(self):
        
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.spawnpoint_x, self.spawnpoint_y)
        self.snake = [self.head]
        self.enemy = Enemy(350, 155, 14, 14, 5, (0, 0, 255))
        self.enemy2 = Enemy(414, 484, 14, 14, 5, (0, 0, 255))
        self.enemy3 = Enemy(478, 155, 14, 14, 5, (0, 0, 255), False)
        self.enemy4 = Enemy(542, 484, 14, 14, 5, (0, 0, 255), False)
        self.enemy5 = Enemy(606, 155, 14, 14, 5, (0, 0, 255), False)
        self.enemy6 = Enemy(670, 484, 14, 14, 5, (0, 0, 255), False)
        self.enemy7 = Enemy(734, 155, 14, 14, 5, (0, 0, 255), False)
        self.enemy8 = Enemy(798, 484, 14, 14, 5, (0, 0, 255), False)
        self.enemy9 = Enemy(862, 155, 14, 14, 5, (0, 0, 255), False)
        self.enemy10 = Enemy(926, 484, 14, 14, 5, (0, 0, 255), False)
        self.enemy11 = Enemy(990, 155, 14, 14, 5, (0, 0, 255), False)
        self.enemy12 = Enemy(1054, 484, 14, 14, 5, (0, 0, 255), False)
        self.score = 0
        self.food = None
        self.food_x = 340 #340
        self.food_y = 436
        self._place_food()
        self.frame_iteration = 0


    def _place_food(self):
        
        self.food = Point(self.food_x, self.food_y)
        self.food_rect = pygame.Rect(self.food_x, self.food_y, BLOCK_SIZE, BLOCK_SIZE)
        self.food_x = 500
        self.food_y = 436
        if self.food in self.snake:
            self._place_food1()


    def _place_food1(self):
        self.food_x = 512
        self.food_y = 436
        self.food = Point(self.food_x, self.food_y)
        self.food_rect = pygame.Rect(self.food_x, self.food_y, BLOCK_SIZE, BLOCK_SIZE)
        
        if self.food in self.snake:
            self._place_food2()
    
    def _place_food2(self):
        self.food_x = 772
        self.food_y = 372
        self.food = Point(self.food_x, self.food_y)
        self.food_rect = pygame.Rect(self.food_x, self.food_y, BLOCK_SIZE, BLOCK_SIZE)
        
        if self.food in self.snake:
            self._place_food3()

    def _place_food3(self):
        self.food_x = 896
        self.food_y = 320
        self.food = Point(self.food_x, self.food_y)
        self.food_rect = pygame.Rect(self.food_x, self.food_y, BLOCK_SIZE, BLOCK_SIZE)
        
        if self.food in self.snake:
            self._place_food4()
    
    
    def _place_food4(self):
        self.food_x = 914
        self.food_y = 200
        self.food = Point(self.food_x, self.food_y)
        self.food_rect = pygame.Rect(self.food_x, self.food_y, BLOCK_SIZE, BLOCK_SIZE)
        
        if self.food in self.snake:
            self._place_food5()

    def _place_food5(self):
        self.food_x = 1106
        self.food_y = 200
        self.food = Point(self.food_x, self.food_y)
        self.food_rect = pygame.Rect(self.food_x, self.food_y, BLOCK_SIZE, BLOCK_SIZE)
        
        if self.food in self.snake:
            self._place_food5()

    def play_step(self, action):
        self.frame_iteration += 1
        
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 10*(self.score+10):
            self.hascollided = False
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        food_collide = pygame.Rect.colliderect(self.head_rect, self.food_rect)
        if food_collide:
            self.score += 1
            reward = 20
            if self.score==1:
                self._place_food()
            elif self.score == 2:
                self._place_food2()
            elif self.score == 3:
                self._place_food3()
            elif self.score == 4:
                self._place_food4()
            elif self.score == 5:
                self._place_food5()
            self.snake.pop()
        else:
            self.snake.pop()            
            reward = 100/math.sqrt((self.food.x  - self.head.x)**2 + (self.food.y - self.head.y)**2)
        index = 0
        pt = self.head
        self.collidedtop1 = pygame.Rect.colliderect(self.head_rect, self.top1rect)
        # self.collidedleft1 = pygame.Rect.colliderect(self.head_rect, self.left1rect)
        self.collidedleft2 = pygame.Rect.colliderect(self.head_rect, self.left2rect)
        # self.collidedleft3 = pygame.Rect.colliderect(self.head_rect, self.left3rect)
        self.collidedtopleft = pygame.Rect.colliderect(self.head_rect, self.toprect)
        self.collidedbottomleft = pygame.Rect.colliderect(self.head_rect, self.bottomleftrect)
        self.collidedbot1 = pygame.Rect.colliderect(self.head_rect, self.bot1rect)
        self.collidedbot2 = pygame.Rect.colliderect(self.head_rect, self.bot2rect)
        self.collidedright1 = pygame.Rect.colliderect(self.head_rect, self.right1rect)
        # 5. update ui and clock
        self.drawColliders()
        self._update_ui()
        # self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score


    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        if self.hascollided == True:
            return True
        self.head_rect = pygame.Rect(self.head.x, self.head.y, BLOCK_SIZE*2, BLOCK_SIZE*1.2)
        collide = pygame.Rect.colliderect(self.head_rect, self.enemy.rect2)
        if collide:
            return True
        collide2 = pygame.Rect.colliderect(self.head_rect, self.enemy2.rect2)
        if collide2:
            return True
        collide3 = pygame.Rect.colliderect(self.head_rect, self.enemy3.rect2)
        if collide3:
            return True
        collide4 = pygame.Rect.colliderect(self.head_rect, self.enemy4.rect2)
        if collide4:
            return True
        collide5 = pygame.Rect.colliderect(self.head_rect, self.enemy5.rect2)
        if collide5:
            return True
        collide6 = pygame.Rect.colliderect(self.head_rect, self.enemy6.rect2)
        if collide6:
            return True
        collide7 = pygame.Rect.colliderect(self.head_rect, self.enemy7.rect2)
        if collide7:
            return True
        collide8 = pygame.Rect.colliderect(self.head_rect, self.enemy8.rect2)
        if collide8:
            return True
        collide9 = pygame.Rect.colliderect(self.head_rect, self.enemy9.rect2)
        if collide9:
            return True
        collide10 = pygame.Rect.colliderect(self.head_rect, self.enemy10.rect2)
        if collide10:
            return True
        collide11 = pygame.Rect.colliderect(self.head_rect, self.enemy11.rect2)
        if collide11:
            return True
        collide12 = pygame.Rect.colliderect(self.head_rect, self.enemy12.rect2)
        if collide12:
            return True
        return False


    def _update_ui(self):
        
        self.screen.fill("white")
        self.sprite_group.draw(self.screen)
        for pt in self.snake:
            pygame.draw.rect(self.screen, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.screen, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.screen, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        # Drawing Colliders on screen
        # pygame.draw.rect(self.screen, BLACK, pygame.Rect(0, 0, 300, 320), 2) #TopLeft
        # pygame.draw.rect(self.screen, BLACK, pygame.Rect(300, 128, 768, 40), 2) #Top1
        # pygame.draw.rect(self.screen, BLACK, pygame.Rect(0, 384, 320, 50), 2) # BottomLeft
        # pygame.draw.rect(self.screen, BLACK, pygame.Rect(250, 500, 70, 20), 2) # Bot1
        # pygame.draw.rect(self.screen, BLACK, pygame.Rect(320, 505, 792, 20), 2) # Bot2
        # pygame.draw.rect(self.screen, BLACK, pygame.Rect(374, 512, 60, 70), 2) # Bot3
        # pygame.draw.rect(self.screen, BLACK, pygame.Rect(250, 200, 25, 300), 2) #Left1
        # pygame.draw.rect(self.screen, BLACK, pygame.Rect(50, 200, 256, 400), 2) #Left2
        # pygame.draw.rect(self.screen, BLACK, pygame.Rect(300, 200, 25, 300), 2) #Left3
        # pygame.draw.rect(self.screen, BLACK, pygame.Rect(1088, 384, 25, 128), 2) # Right1
        # pygame.draw.rect(self.screen, BLACK, pygame.Rect(900, 170, 320, 30), 2) # Right2
        self.enemy.move2(155, 484)
        self.enemy2.move2(155, 484)
        self.enemy3.move2(155, 484)
        self.enemy4.move2(155, 484)
        self.enemy5.move2(155, 484)
        self.enemy6.move2(155, 484)
        self.enemy7.move2(155, 484)
        self.enemy8.move2(155, 484)
        self.enemy9.move2(155, 484)
        self.enemy10.move2(155, 484)
        self.enemy11.move2(155, 484)
        self.enemy12.move2(155, 484)
        self.enemy.draw(self.screen)
        self.enemy2.draw(self.screen)
        self.enemy3.draw(self.screen)
        self.enemy4.draw(self.screen)
        self.enemy5.draw(self.screen)
        self.enemy6.draw(self.screen)
        self.enemy7.draw(self.screen)
        self.enemy8.draw(self.screen)
        self.enemy9.draw(self.screen)
        self.enemy10.draw(self.screen)
        self.enemy11.draw(self.screen)
        self.enemy12.draw(self.screen)


        text = font.render("Score: " + str(self.score), True, WHITE)
        self.screen.blit(text, [0, 0])
        # pygame.time.delay(50)
        pygame.display.update()
        


    def _move(self, action):
        # [right, down, left, up]
        self.head_rect = pygame.Rect(self.head.x, self.head.y, BLOCK_SIZE*2, BLOCK_SIZE*1.2)  
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
        if self.collidedtopleft or self.collidedtop1:
            y += SPEED*2
            reward = -1
        if self.collidedbottomleft or self.collidedbot2:
            y -= SPEED*2
            reward = -1
        if self.collidedleft1:
            x -= SPEED*2
        if self.collidedright1:
            x -= SPEED*2 
            y -= SPEED*2
        if self.collidedbot1 or self.collidedright2:
            x += SPEED*2
            y += SPEED*2
        if self.collidedbot3:
            x -= SPEED*2
            y += SPEED*2
        if self.collidedleft2 or self.collidedleft3:
            x += SPEED*2
        if self.direction == Direction.RIGHT:
            x += SPEED
        elif self.direction == Direction.LEFT:
            x -= SPEED
        elif self.direction == Direction.DOWN:
            y += SPEED
        elif self.direction == Direction.UP:
            y -= SPEED

        self.head = Point(x, y)