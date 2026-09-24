"""Turn the raw files in data/ into the daily table."""

from src.config import SEED, set_global_seed


def main() -> None:
    set_global_seed(SEED)
    raise NotImplementedError


if __name__ == "__main__":
    main()
