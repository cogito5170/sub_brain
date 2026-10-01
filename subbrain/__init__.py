"""subbrain -- LLM 옆에 붙는 보조-뇌(cognitive co-processor).

Blackboard(Hearsay-II) + Truth Maintenance(Doyle) + Impasse/Subgoal(SOAR)
+ STRIPS Planner + Verifier + 장기 Memory.
"""
__version__ = "0.1.0"

from .brain import AuxBrain  # noqa: E402,F401
from .protocol import apply, parse, SYSTEM  # noqa: E402,F401
