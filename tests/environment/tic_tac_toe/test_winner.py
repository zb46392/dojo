from environment import TicTacToe
import unittest
from unittest import TestCase
from .utils import generate_horizontal_win_actions, generate_vertical_win_actions, generate_diagonal_win_actions, \
    find_loser_action
from typing import Tuple


class TestWinner(TestCase):
    def setUp(self) -> None:
        super().setUp()

        self.ttt = TicTacToe()

    def test_win_after_reset(self) -> None:
        self.ttt.execute_action((0, 0))
        self.ttt.execute_action((1, 0))

        self.ttt.execute_action((0, 1))
        self.ttt.execute_action((1, 1))

        self.ttt.execute_action((0, 2))

        self.assertEqual('x', self.ttt.winner)
        self.assertEqual(((0, 0), (0, 1), (0, 2)), self.ttt.win_position)

        self.ttt.reset()

        expected_winner = None
        expected_win_pos = None

        actual_winner = self.ttt.winner
        actual_win_pos = self.ttt.win_position

        self.assertEqual(expected_winner, actual_winner)
        self.assertEqual(expected_win_pos, actual_win_pos)

    def test_winner_on_horizontal(self) -> None:
        actions = generate_horizontal_win_actions()
        self._run_plays(2, actions)

    def test_winner_on_vertical(self) -> None:
        actions = generate_vertical_win_actions()
        self._run_plays(2, actions)

    def test_winner_on_diagonal(self) -> None:
        actions = generate_diagonal_win_actions()
        self._run_plays(2, actions)

    def _run_plays(self, plays: int, w_actions: Tuple[Tuple[Tuple[int, int], ...], ...]) -> None:
        for run in range(plays):
            if run % 2 == 0:
                first = 'x'
            else:
                first = 'o'

            self.ttt = TicTacToe(first_turn=first)
            self._run_play(first, w_actions)

    def _run_play(self, first: str, w_actions: Tuple[Tuple[Tuple[int, int], ...], ...]):
        for i in range(len(w_actions)):
            self.ttt.reset()

            self.assertEqual(None, self.ttt.winner)
            self.assertEqual(None, self.ttt.win_position)

            for j in range(len(w_actions[i])):
                self.ttt.execute_action(w_actions[i][j])
                if j < len(w_actions[i]) - 1:
                    l_action = find_loser_action(self.ttt, w_actions[i])
                    self.ttt.execute_action(l_action)

                    self.assertEqual(None, self.ttt.winner)
                    self.assertEqual(None, self.ttt.win_position)

            self.assertEqual(first, self.ttt.winner)
            self.assertEqual(tuple(w_actions[i]), self.ttt.win_position)


if __name__ == '__main__':
    unittest.main()
