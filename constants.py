"""Project-wide defaults for noise modeling and gate timing.
All durations are expressed in microseconds (μs).
"""
from typing import Dict

DEFAULT_T1: float = 100.0
DEFAULT_T2: float = 200.0

DEFAULT_GATE_DURATIONS: Dict[str, float] = {
    "CNOT": 0.2,   # 200 ns = 0.2 μs
    "RY": 0.025,   # 25 ns = 0.025 μs
    "H": 0.025,    # 25 ns = 0.025 μs (useful for example circuits)
}
