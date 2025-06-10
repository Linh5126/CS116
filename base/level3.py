import os
import sys
import pygame
from player import Player
from enemy import Enemy
from pytmx.util_pygame import load_pygame
# from mainscreen import main_screen
# from mainscreen import draw_level_selection_menu

# flag = 0

class Level3:
    def __init__(self):
        pygame.init()
        self.islevel3 = True
        self.screen_w = 1320
        self.screen_h = 720
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        self.tmx_data = load_pygame('Tiles/Level3.tmx')
        self.sprite_group = pygame.sprite.Group()
        self.is_running = True
        self.spawnpoint_x = 675
        self.spawnpoint_y = 375

        self.player = Player(self.spawnpoint_x, self.spawnpoint_y, 37, 37, 5)
        self.enemies = [
            Enemy(608, 285, 14, 14, direction=1),
            Enemy(608, 350, 14, 14, direction=4),
            Enemy(608, 415, 14, 14, direction=4),
            Enemy(608, 480, 14, 14, direction=4),
            Enemy(672, 480, 14, 14, direction=3),
            Enemy(736, 480, 14, 14, direction=3),
            Enemy(800, 480, 14, 14, direction=2),
            Enemy(800, 415, 14, 14, direction=2),
            # Đã xóa 2 enemies: Enemy(800, 350) và Enemy(800, 285)
        ]

        self.tile_rect = []
        self.collider_rects = []
        self.color = (255, 0, 0)
        self.red_rect = pygame.Rect(570, 180, 50, 50)
        self.red_rect.topleft = (570, 180)
        self.left_point = pygame.Rect(self.player.player_x + self.player.width, self.player.player_y + self.player.width / 2, 5, 5)
        self.left_point.topleft = (self.player.player_x + self.player.width, self.player.player_y + self.player.width)
        self.right_point = pygame.Rect(self.player.player_x - 1, self.player.player_y + self.player.width / 2, 1, 1)
        self.right_point.topleft = (self.player.player_x - 1, self.player.player_y + self.player.width / 2)
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
                    if surf is not None:
                        pos = (x * 64, y * 64)
                        self.Tile(pos=pos, surf=surf, groups=self.sprite_group)
                        # tile_rect = pygame.Rect(x * 64, y * 64, 64, 64)
    def update(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_BACKSPACE]:
            from mainscreen import main_screen
            main_screen()
            self.islevel3 = False
            self.is_running = False
            return
        self.player.move(keys)
        for enemy in self.enemies:
            enemy.enemy_speed = 5
        for enemy in self.enemies:
            enemy.move3(608, 800, 285, 480)

        self.left_point = pygame.Rect(self.player.player_x + self.player.width, self.player.player_y + self.player.width / 2, 5, 5)
        self.right_point = pygame.Rect(self.player.player_x - 1, self.player.player_y + self.player.width / 2, 1, 1)
        self.up_point = pygame.Rect(self.player.player_x + self.player.width / 2, self.player.player_y - 1, 1, 1)
        self.down_point = pygame.Rect(self.player.player_x + self.player.width / 2, self.player.player_y + self.player.width, 1, 1)

        for enemy in self.enemies:
            if pygame.Rect.colliderect(self.player.rect1, enemy.rect2):
                self.reset()

        if pygame.Rect.colliderect(self.player.rect1, self.red_rect):
            from mainscreen import main_screen
            self.islevel3 = False
            main_screen()
            sys.exit()

    def reset(self):
        self.player.player_x = self.spawnpoint_x
        self.player.player_y = self.spawnpoint_y

    def drawColliders(self):
        # Kiểm tra va chạm với các tile
        for index, b in enumerate(self.tile_rect):
            cleft_point = pygame.Rect.colliderect(self.left_point, self.collider_rects[index])
            if cleft_point:
                self.player.player_x -= 5
                self.player.can_move_left = False
            elif not cleft_point:
                self.player.can_move_left = True

            cright_point = pygame.Rect.colliderect(self.right_point, self.collider_rects[index])
            if cright_point:
                self.player.player_x += 5
                self.player.can_move_right = False
            elif not cright_point:
                self.player.can_move_right = True

            c_up_point = pygame.Rect.colliderect(self.up_point, self.collider_rects[index])
            if c_up_point:
                self.player.player_y += 5
                self.player.can_move_up = False
            elif not c_up_point:
                self.player.can_move_up = True

            c_down_point = pygame.Rect.colliderect(self.down_point, self.collider_rects[index])
            if c_down_point:
                self.player.player_y -= 5
                self.player.can_move_down = False
            elif not c_down_point:
                self.player.can_move_down = True

        # Kiểm tra giới hạn màn hình
        if self.player.player_x < 0:
            self.player.player_x = 0
        if self.player.player_x + self.player.width > self.screen_w:
            self.player.player_x = self.screen_w - self.player.width
        if self.player.player_y < 0:
            self.player.player_y = 0
        if self.player.player_y + self.player.height > self.screen_h:
            self.player.player_y = self.screen_h - self.player.height

    def draw(self):
        if self.islevel3:
            self.screen.fill('white')
            self.drawColliders()
            self.sprite_group.draw(self.screen)
            self.player.draw(self.screen)

            for enemy in self.enemies:
                enemy.draw(self.screen)

            pygame.draw.rect(self.screen, self.color, self.red_rect, 2)
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
    game = Level3()
    game.run()