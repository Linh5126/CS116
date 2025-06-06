import pygame

class Player:
    def __init__(self, player_x, player_y, width, height, player_speed, color=(255, 0, 0)):
        self.player_x = player_x
        self.player_y = player_y
        self.width = width
        self.height = height
        self.player_speed = player_speed
        self.color = color
        self.rect1 = pygame.Rect(player_x, player_y, width, height)
        self.rect1.topleft = (player_x, player_y)
        self.can_move_left = True
        self.can_move_up = True
        self.can_move_down = True
    def move(self, keys):
        if keys[pygame.K_LEFT] and self.player_x > 0:
            self.player_x -= self.player_speed
            self.rect1.x = self.player_x
        if keys[pygame.K_RIGHT] and self.can_move_left:
            self.player_x += self.player_speed
            self.rect1.x = self.player_x
        if keys[pygame.K_UP] and self.can_move_up:
            self.player_y -= self.player_speed
            self.rect1.y = self.player_y
        if keys[pygame.K_DOWN] and self.can_move_down:
            self.player_y += self.player_speed
            self.rect1.y = self.player_y
    def draw(self, screen):
        pygame.draw.rect(screen, (255, 0, 0), (self.player_x, self.player_y, self.width, self.height))
