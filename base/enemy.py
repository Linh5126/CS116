import pygame


class Enemy:
    def __init__(self, player_x, player_y, width, height, enemy_speed=25, color=(0, 0, 255), right=True, up=True, direction=1):
        self.player_x = player_x
        self.player_y = player_y
        self.width = width
        self.height = height
        self.enemy_speed = enemy_speed
        self.color = color
        self.right = right
        self.up = up
        self.direction = direction
        self.rect2 = pygame.Rect(player_x-14, player_y-14, 28, 28)
        self.rect2.topleft = (player_x, player_y)

    def draw(self, screen):
        pygame.draw.circle(screen, (0, 0, 255), (self.player_x, self.player_y), self.width, self.height)

    def move(self, bound_x1, bound_x2):
        if self.player_x < bound_x1 or self.player_x > bound_x2:
            self.right = not self.right
        if self.right:
            self.player_x += self.enemy_speed
            self.rect2.x = self.player_x
        else:
            self.player_x -= self.enemy_speed
            self.rect2.x = self.player_x
    def move2(self, bound_y1, bound_y2):
        if self.player_y < bound_y1 or self.player_y > bound_y2:
            self.up = not self.up
        if self.up:
            self.player_y += self.enemy_speed
            self.rect2.y = self.player_y
        else:
            self.player_y -= self.enemy_speed
            self.rect2.y = self.player_y
    def move3(self, bound_x1, bound_x2, bound_y1, bound_y2):
        if self.direction == 1:  # Moving right
            self.player_x += self.enemy_speed
            if self.player_x >= bound_x2:
                self.player_x = bound_x2
                self.direction = 2  # Change direction to down
        elif self.direction == 2:  # Moving down
            self.player_y += self.enemy_speed
            if self.player_y >= bound_y2:
                self.player_y = bound_y2
                self.direction = 3  # Change direction to left
        elif self.direction == 3:  # Moving left
            self.player_x -= self.enemy_speed
            if self.player_x <= bound_x1:
                self.player_x = bound_x1
                self.direction = 4  # Change direction to up
        elif self.direction == 4:  # Moving up
            self.player_y -= self.enemy_speed
            if self.player_y <= bound_y1:
                self.player_y = bound_y1
                self.direction = 1  # Change direction to right
        
        # Cập nhật collision rect sau khi di chuyển
        self.rect2.x = self.player_x
        self.rect2.y = self.player_y