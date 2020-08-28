from environment import TicTacToe
from typing import List, Optional, Tuple


def generate_horizontal_win_actions() -> Tuple[Tuple[Tuple[int, int], ...], ...]:
    return (
        ((0, 0), (0, 1), (0, 2)),
        ((1, 0), (1, 1), (1, 2)),
        ((2, 0), (2, 1), (2, 2))
    )


def generate_vertical_win_actions() -> Tuple[Tuple[Tuple[int, int], ...], ...]:
    return (
        ((0, 0), (1, 0), (2, 0)),
        ((0, 1), (1, 1), (2, 1)),
        ((0, 2), (1, 2), (2, 2))
    )


def generate_diagonal_win_actions() -> Tuple[Tuple[Tuple[int, int], ...], ...]:
    return ((0, 0), (1, 1), (2, 2)), ((0, 2), (1, 1), (2, 0))


def find_loser_action(tic_tac_toe: TicTacToe, winning_actions: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    for action in tic_tac_toe.possible_actions:
        if action not in winning_actions:
            return action

    return None
