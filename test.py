"""Evaluate a trained checkpoint on the official NWPU-MOC split."""
from config import TEST_CONFIG


def main():
    from trainer import run

    run(TEST_CONFIG)


if __name__ == "__main__":
    main()
