"""Train and validate the visual MOC-6 baseline."""
from config import TRAIN_CONFIG


def main():
    from trainer import run

    run(TRAIN_CONFIG)


if __name__ == "__main__":
    main()
