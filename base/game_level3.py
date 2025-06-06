import pygame
import math
import numpy as np

from base_environment import BaseEnvironment
from colors import Colors
from constants import TILE_SIZE, PLAYER_SIZE, ENEMY_SIZE
from helper1 import load_tilemap, place_object, collides_with_any

class Level3AI(BaseEnvironment):
    def __init__(self):
        super().__init__()

        self.tilemap = load_tilemap("level3.txt")
        self.rows = len(self.tilemap)
        self.cols = len(self.tilemap[0])
        self.width = self.cols * TILE_SIZE
        self.height = self.rows * TILE_SIZE

        self.player_start_pos = None
        self.goal_pos = None
        self.player = pygame.Rect(0, 0, PLAYER_SIZE, PLAYER_SIZE)

        self.foods = []
        self.enemies = []
        self.enemy_directions = []

        self.reset()

    def reset(self):
        self.player_start_pos = None
        self.goal_pos = None
        self.foods.clear()
        self.enemies.clear()
        self.enemy_directions.clear()

        # Parse tilemap
        for y, row in enumerate(self.tilemap):
            for x, tile in enumerate(row):
                if tile == 's':  # start
                    self.player_start_pos = (x * TILE_SIZE + TILE_SIZE // 2 - PLAYER_SIZE // 2,
                                             y * TILE_SIZE + TILE_SIZE // 2 - PLAYER_SIZE // 2)
                elif tile == 'g':  # goal
                    self.goal_pos = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                elif tile == 'f':  # food
                    food = pygame.Rect(x * TILE_SIZE + TILE_SIZE // 2 - 5,
                                       y * TILE_SIZE + TILE_SIZE // 2 - 5, 10, 10)
                    self.foods.append(food)
                elif tile == 'e':  # enemy
                    enemy = pygame.Rect(x * TILE_SIZE + TILE_SIZE // 2 - ENEMY_SIZE // 2,
                                        y * TILE_SIZE + TILE_SIZE // 2 - ENEMY_SIZE // 2,
                                        ENEMY_SIZE, ENEMY_SIZE)
                    self.enemies.append(enemy)
                    self.enemy_directions.append(1)

        # Fallback nếu không có start
        if self.player_start_pos:
            self.player.topleft = self.player_start_pos
        else:
            self.player.topleft = (TILE_SIZE, TILE_SIZE)

        self.steps = 0
        return self.get_state()

    def step(self, action):
        self.steps += 1
        dx, dy = 0, 0
        speed = 4

        if action == 0: dx = -speed
        elif action == 1: dx = speed
        elif action == 2: dy = -speed
        elif action == 3: dy = speed

        # Move player and check wall collision
        next_pos = self.player.move(dx, dy)
        if not self.collides_with_walls(next_pos):
            self.player = next_pos

        # Check food collision
        reward = 0
        for food in self.foods[:]:
            if self.player.colliderect(food):
                self.foods.remove(food)
                reward += 10

        # Enemy movement
        for i, enemy in enumerate(self.enemies):
            direction = self.enemy_directions[i]
            enemy.y += direction * 2

            if self.collides_with_walls(enemy):
                enemy.y -= direction * 2
                self.enemy_directions[i] *= -1

        # Enemy collision
        if collides_with_any(self.player, self.enemies):
            return self.get_state(), -100, True

        # Goal check
        if self.goal_pos and self.player.colliderect(self.goal_pos):
            return self.get_state(), 100, True

        # Timeout
        done = self.steps >= 1000
        return self.get_state(), reward, done

    def get_state(self):
        return np.array([
            self.player.centerx / self.width,
            self.player.centery / self.height,
            *(f.centerx / self.width for f in self.foods[:2]),
            *(f.centery / self.height for f in self.foods[:2]),
            *(e.centerx / self.width for e in self.enemies[:2]),
            *(e.centery / self.height for e in self.enemies[:2]),
        ] + [0] * (8 - 2 * len(self.foods[:2])) + [0] * (8 - 2 * len(self.enemies[:2])))

    def render(self, screen):
        screen.fill(Colors.LIGHT_BLUE)

        for y, row in enumerate(self.tilemap):
            for x, tile in enumerate(row):
                rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                if tile == 'w':
                    pygame.draw.rect(screen, Colors.DARK_BLUE, rect)
                elif tile == 'g':
                    pygame.draw.rect(screen, Colors.GREEN, rect)

        pygame.draw.rect(screen, Colors.RED, self.player)

        for food in self.foods:
            pygame.draw.rect(screen, Colors.YELLOW, food)

        for enemy in self.enemies:
            pygame.draw.rect(screen, Colors.BLACK, enemy)

    def collides_with_walls(self, rect):
        for y, row in enumerate(self.tilemap):
            for x, tile in enumerate(row):
                if tile == 'w':
                    wall_rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                    if rect.colliderect(wall_rect):
                        return True
        return False
