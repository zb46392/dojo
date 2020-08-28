from typing import NamedTuple, Tuple


class ObservableEnvironment(NamedTuple):
    state: Tuple[int, ...]
    reward: float
    is_terminal: bool
