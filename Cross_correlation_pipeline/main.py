from config import CONFIG
from pipeline import run_pipeline


def main() -> None:
    run_pipeline(CONFIG)


if __name__ == "__main__":
    main()