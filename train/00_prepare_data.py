"""Step 0 - download OffensEval-TR 2020 and create the train / val / test splits."""
import _common  # noqa: F401
from src.data import prepare

if __name__ == "__main__":
    prepare()
