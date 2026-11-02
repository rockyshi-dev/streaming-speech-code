import sys
from pathlib import Path

# import hbasr from this repository's code/, ahead of any installed copy
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
