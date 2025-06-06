import pygame
import sys
from agent import train
from agent2 import train2
from level2 import Level2
from level1 import Level1
from game_level1 import Level1AI
from game_level2 import Level2AI
# from game_ai import snake

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600

# Colors
BLACK = (0, 0, 0)
GRAY = (169, 169, 169)
BLUEISH_WHITE = (240, 248, 255)  # A light bluish-white color

# Fonts
pygame.font.init()
TITLE_FONT = pygame.font.Font(None, 74)
MENU_FONT = pygame.font.Font(None, 50)

# Screen setup
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Main Menu")

# Main menu options
main_menu_options = ["Play as Human", "DQN AI", "DOUBLE DQN AI", "Quit"]
level_options = ["Level 1", "Level 2"]
selected_option = 0

# Current screen flag
# current_screen = "main_menu"


def draw_main_menu():
    screen.fill(BLUEISH_WHITE)

    # Draw the game title
    title_surface = TITLE_FONT.render("World's Hardest Game", True, BLACK)
    title_rect = title_surface.get_rect(center=(WIDTH / 2, HEIGHT / 4))
    screen.blit(title_surface, title_rect)

    # Draw the menu options
    for i, option in enumerate(main_menu_options):
        color = BLACK if i == selected_option else GRAY
        text_surface = MENU_FONT.render(option, True, color)
        text_rect = text_surface.get_rect(center=(WIDTH / 2, HEIGHT / 2 + i * 75))
        screen.blit(text_surface, text_rect)

    pygame.display.flip()


def draw_level_selection_menu():
    screen.fill(BLUEISH_WHITE)

    # Draw the level selection title
    title_surface = TITLE_FONT.render("Level Selection", True, BLACK)
    title_rect = title_surface.get_rect(center=(WIDTH / 2, HEIGHT / 4))
    screen.blit(title_surface, title_rect)

    # Draw the level options
    for i, option in enumerate(level_options):
        color = BLACK if i == selected_option else GRAY
        text_surface = MENU_FONT.render(option, True, color)
        text_rect = text_surface.get_rect(center=(WIDTH / 2, HEIGHT / 2 + i * 75))
        screen.blit(text_surface, text_rect)

    pygame.display.flip()


def draw_level_selection_menuDQN():
    screen.fill(BLUEISH_WHITE)

    # Draw the level selection title
    title_surface = TITLE_FONT.render("Level Selection DQN", True, BLACK)
    title_rect = title_surface.get_rect(center=(WIDTH / 2, HEIGHT / 4))
    screen.blit(title_surface, title_rect)

    # Draw the level options
    for i, option in enumerate(level_options):
        color = BLACK if i == selected_option else GRAY
        text_surface = MENU_FONT.render(option, True, color)
        text_rect = text_surface.get_rect(center=(WIDTH / 2, HEIGHT / 2 + i * 75))
        screen.blit(text_surface, text_rect)

    pygame.display.flip()

def draw_level_selection_menuREINFORCE():
    screen.fill(BLUEISH_WHITE)
    title_surface = TITLE_FONT.render("Level Selection DB DQN", True, BLACK)
    title_rect = title_surface.get_rect(center=(WIDTH / 2, HEIGHT / 4))
    screen.blit(title_surface, title_rect)

    for i, option in enumerate(level_options):
        color = BLACK if i == selected_option else GRAY
        text_surface = MENU_FONT.render(option, True, color)
        text_rect = text_surface.get_rect(center=(WIDTH / 2, HEIGHT / 2 + i * 75))
        screen.blit(text_surface, text_rect)

    pygame.display.flip()

def main_screen():
    global selected_option # current_screen
    current_screen = "main_menu"

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DOWN:
                    if current_screen == "main_menu":
                        selected_option = (selected_option + 1) % len(main_menu_options)
                    elif current_screen in ["level_selection", "level_selection DQN", "level_selection DBDQN"]:
                        selected_option = (selected_option + 1) % len(level_options)
                elif event.key == pygame.K_UP:
                    if current_screen == "main_menu":
                        selected_option = (selected_option - 1) % len(main_menu_options)
                    elif current_screen in ["level_selection", "level_selection DQN", "level_selection DBDQN"]:
                        selected_option = (selected_option - 1) % len(level_options)
                elif event.key == pygame.K_RETURN:
                    if current_screen == "main_menu":
                        if selected_option == 0:
                            current_screen = "level_selection"
                            selected_option = 0  # Reset selected option for level selection
                        elif selected_option == 1:
                            current_screen = "level_selection DQN"
                            selected_option = 0
                            # Add your code to start the game as an AI player
                        elif selected_option == 2:
                            current_screen = "level_selection DBDQN"
                            selected_option = 0
                        elif selected_option == 3:
                            pygame.quit()
                            sys.exit()
                    elif current_screen == "level_selection":
                        if selected_option == 0:
                            print("Level 1 selected")
                            game = Level1()
                            game.run()
                        elif selected_option == 1:
                            print("Level 2 selected")
                            game = Level2()
                            game.run()
                        #elif selected_option == 2:
                        #    print("Level 3 selected")
                        #    game = Level3()
                        #    game.run()

                    elif current_screen == "level_selection DQN":
                        if selected_option == 0:
                            print("Level 1 selected")
                            game=Level1AI()
                            train(game)
                        elif selected_option == 1:
                            print("Level 2 selected")
                            game=Level2AI()
                            train(game)
                            # Add your code to start level 2
                        #elif selected_option == 2:
                        #    print("Level 3 selected")
                        #    game = Level2AI()
                        #    train(game)
                            # Add your code to start level 3
                    elif current_screen == "level_selection DBDQN":
                        if selected_option == 0:
                            print("Level 1 selected")
                            game = Level1AI()
                            train2(game)
                        elif selected_option == 1:
                            print("Level 2 selected")
                            game = Level2AI()
                            train2(game)
                        #elif selected_option == 2:
                        #    print("Level 3 - REINFORCE selected")
                        #    game = Level2AI()
                        #    train2(game)
                elif event.key == pygame.K_BACKSPACE:
                    if current_screen == "level_selection":
                        current_screen = "main_menu"
                        selected_option = 0  # Reset selected option for main menu
                    elif current_screen == "level_selection DQN":
                        current_screen = "main_menu"
                        selected_option = 0  # Reset selected option for main menu
                    elif current_screen == "level_selection DBDQN":
                        current_screen = "main_menu"
                        selected_option = 0

        if current_screen == "main_menu":
            draw_main_menu()
        elif current_screen == "level_selection":
            draw_level_selection_menu()
        elif current_screen == "level_selection DQN":
            draw_level_selection_menuDQN()
        elif current_screen == "level_selection DBDQN":
            draw_level_selection_menuREINFORCE()


if __name__ == "__main__":
    main_screen()
