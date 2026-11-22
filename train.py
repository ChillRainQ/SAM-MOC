"""Train and validate the visual MOC-6 baseline."""
from config import cfg


def main():
    from trainer import run

    run(cfg)


if __name__ == "__main__":
    main()
