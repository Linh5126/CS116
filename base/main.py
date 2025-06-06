import argparse
from agent import train
from agent2 import train2
from game_level1 import Level1AI
from game_level2 import Level2AI

LEVEL_MAP = {
    "level1": Level1AI,
    "level2": Level2AI
}

AGENT_MAP = {
    "dqn": train,
    "double_dqn": train2
}

def parse_args():
    parser = argparse.ArgumentParser(description="Train agent with selected level and algorithm.")
    parser.add_argument("level", type=str, choices=LEVEL_MAP.keys(),
                        help="Game level to train: level1, level2")
    parser.add_argument("algo", type=str, choices=AGENT_MAP.keys(),
                        help="Training algorithm: dqn or db_dqn")
    return parser.parse_args()

def main():
    args = parse_args()
    level_class = LEVEL_MAP[args.level]
    train_func = AGENT_MAP[args.algo]

    print(f"🚀 Training: {args.algo.upper()} on {args.level.capitalize()}")
    train_func(game=level_class())

if __name__ == '__main__':
    main()
