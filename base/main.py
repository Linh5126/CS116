import argparse
from agent import train
from agent2 import train2
from game_level1 import Level1AI
from game_level2 import Level2AI
from game_level3 import Level3AI

LEVEL_MAP = {
    "level1": Level1AI,
    "level2": Level2AI,
    "level3": Level3AI
}

AGENT_MAP = {
    "dqn": train,
    "double_dqn": train2
}

def parse_args():
    parser = argparse.ArgumentParser(description="Train agent with selected level and algorithm.")
    parser.add_argument("level", type=str, choices=LEVEL_MAP.keys(),
                        help="Game level to train: level1, level2, level3")
    parser.add_argument("algo", type=str, choices=AGENT_MAP.keys(),
                        help="Training algorithm: dqn or double_dqn")
    parser.add_argument("num_games", type=int,
                        help="Number of games to train (e.g., 100, 200)")
    return parser.parse_args()


def main():
    args = parse_args()
    level_class = LEVEL_MAP[args.level]
    train_func = AGENT_MAP[args.algo]

    print(f"🚀 Training: {args.algo.upper()} on {args.level.capitalize()} for {args.num_games} games")
    # Giờ hàm train phải nhận thêm tham số num_games
    train_func(game=level_class(), num_games=args.num_games)

if __name__ == '__main__':
    main()
