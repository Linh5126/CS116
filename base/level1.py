import sys
import pygame
from player import Player
from enemy import Enemy
from pytmx.util_pygame import load_pygame
from enum import Enum


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


class Level1:
    def __init__(self):
        pygame.init()
        self.screen_w = 1320
        self.screen_h = 720
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        self.tmx_data = load_pygame('Tiles/Level1.tmx')
        self.sprite_group = pygame.sprite.Group()
        self.islevel1 = True
        self.is_running = True
        self.spawnpoint_x = 140
        self.spawnpoint_y = 275
        self.player = Player(self.spawnpoint_x, self.spawnpoint_y, 37, 37, 5)
        self.enemy = Enemy(640, 285, 14, 14, True, False)
        self.enemy2 = Enemy(640, 415, 14, 14, True, False)
        self.enemy3 = Enemy(640, 350, 14, 14, True, False, False)
        self.enemy4 = Enemy(640, 480, 14, 14, True, False, False)
        self.tile_rect = []
        self.enemyrect = pygame.Rect(self.enemy.player_x, self.enemy.player_y, 14, 14)
        self.collider_rects = []  # List of tile colliders
        self.color = (255, 0, 0)
        self.red_rect = pygame.Rect(1050, 200, 50, 50)
        self.red_rect.topleft = (1050, 200)
        self.right_point = pygame.Rect(self.player.player_x + self.player.width, self.player.player_y + self.player.width / 2, 5, 5)
        self.right_point.topleft = (self.player.player_x + self.player.width, self.player.player_y + self.player.width)
        self.left_point = pygame.Rect(self.player.player_x -4,self.player.player_y + self.player.width / 2, 5, 5)
        self.left_point.topleft = (self.player.player_x -4, self.player.player_y + self.player.width)
        self.up_point = pygame.Rect(self.player.player_x + self.player.width / 2, self.player.player_y - 1, 1, 1)
        self.up_point.topleft = (self.player.player_x + self.player.width / 2, self.player.player_y - 1)
        self.down_point = pygame.Rect(self.player.player_x + self.player.width / 2, self.player.player_y + self.player.width, 1, 1)
        self.down_point.topleft = (self.player.player_x + self.player.width / 2, self.player.player_y + self.player.width)

    class Tile(pygame.sprite.Sprite):
        def __init__(self, pos, surf, groups):
            super().__init__(groups)
            self.image = surf
            self.rect = self.image.get_rect(topleft=pos)

    def setup_tiles(self):
        for layer in self.tmx_data.visible_layers:
            if hasattr(layer, 'data'):
                if layer.name == "Main":
                    for x, y, surf in layer.tiles():
                        pos = (x * 64, y * 64)
                        self.Tile(pos=pos, surf=surf, groups=self.sprite_group)
                        self.tile_rect.append(pygame.Rect(x * 64, y * 64, 64, 64))
                        self.collider_rects.append(pygame.Rect(x * 64, y * 64, 64, 64))

        for layer in self.tmx_data.visible_layers:
            if hasattr(layer, 'data'):
                for x, y, surf in layer.tiles():
                    pos = (x * 64, y * 64)
                    self.Tile(pos=pos, surf=surf, groups=self.sprite_group)
                    # tile_rect = pygame.Rect(x * 64, y * 64, 64, 64)

    def update(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_BACKSPACE]:
            from mainscreen import main_screen
            main_screen()
            self.islevel1 = False
            self.is_running = False
            return
        self.player.move(keys)
        self.enemy.move(345, 935)
        self.enemy2.move(345, 935)
        self.enemy3.move(345, 935)
        self.enemy4.move(345, 935)
        self.right_point = pygame.Rect(self.player.player_x + self.player.width, self.player.player_y + self.player.width / 2, 1, 1)
        self.left_point = pygame.Rect(self.player.player_x - 4, self.player.player_y + self.player.width / 2, 5, 5)
        self.up_point = pygame.Rect(self.player.player_x + self.player.width / 2, self.player.player_y -1 , 1, 1)
        self.down_point = pygame.Rect(self.player.player_x + self.player.width / 2, self.player.player_y + self.player.width, 1, 1)
        collide = pygame.Rect.colliderect(self.player.rect1, self.enemy.rect2)
        if collide:
            self.reset()
        collide2 = pygame.Rect.colliderect(self.player.rect1, self.enemy2.rect2)
        if collide2:
            self.reset()
        collide3 = pygame.Rect.colliderect(self.player.rect1, self.enemy3.rect2)
        if collide3:
            self.reset()
        collide4 = pygame.Rect.colliderect(self.player.rect1, self.enemy4.rect2)
        if collide4:
            self.reset()
        collidered = pygame.Rect.colliderect(self.player.rect1, self.red_rect)
        if collidered:
            from mainscreen import main_screen
            self.islevel1 = False
            main_screen()
            sys.exit()
    def reset(self):
        self.player.player_x = self.spawnpoint_x
        self.player.player_y = self.spawnpoint_y
    def drawColliders(self):
        index = 0
        for b in self.tile_rect:
            cleft_point = pygame.Rect.colliderect(self.left_point, self.collider_rects[index])
            if cleft_point:
                self.player.player_x += 5
                self.player.can_move_left = False
            elif not cleft_point:
                self.player.can_move_left = True
            cright_point = pygame.Rect.colliderect(self.right_point, self.collider_rects[index])
            if cright_point:
                self.player.player_x -= 5
                self.player.can_move_right = False
            elif not cright_point:
                self.player.can_move_right = True
            c_up_point = pygame.Rect.colliderect(self.up_point, self.collider_rects[index])
            if c_up_point:
                self.player.player_y += 5
                self.player.can_move_up = False
            elif not c_up_point:
                self.player.can_move_up= True
            c_down_point = pygame.Rect.colliderect(self.down_point, self.collider_rects[index])
            if c_down_point:
                self.player.player_y -= 5
                self.player.can_move_down = False
            elif not c_down_point:
                self.player.can_move_down = True
            index += 1

    def draw(self):
        if self.islevel1:
            self.screen.fill('white')
            self.drawColliders()
            self.sprite_group.draw(self.screen)
            self.player.draw(self.screen)
            self.enemy.draw(self.screen)
            self.enemy2.draw(self.screen)
            self.enemy3.draw(self.screen)
            self.enemy4.draw(self.screen)
            pygame.draw.rect(self.screen, self.color, pygame.Rect(1050, 200, 60, 60), 2)
            pygame.display.update()

    def run(self):
        self.setup_tiles()
        while self.is_running:
            pygame.time.delay(50)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_running = False
            self.update()
            self.draw()
            pygame.display.update()

        pygame.quit()


if __name__ == "__main__":
    game = Level1()
    game.run()
