"""Allow etchgen to be run as: python -m Art.etchgen"""

from .cli import main
import sys

if __name__ == "__main__":
    sys.exit(main())
