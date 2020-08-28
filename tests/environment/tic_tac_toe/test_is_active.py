import unittest
from unittest import TestCase
from environment import TicTacToe
from .utils import generate_horizontal_win_actions, generate_vertical_win_actions, generate_diagonal_win_actions, \
    find_loser_action


class TestIsActive(TestCase):
    def setUp(self) -> None:
        super().setUp()

        self.ttt = TicTacToe()

    def test_is_active(self) -> None:
        expected = True
        actual = self.ttt.is_active

        self.assertEqual(expected, actual)

    def test_is_active_on_game_over(self) -> None:
        for i in range(len(self.ttt.state)):
            for j in range(len(self.ttt.state[i])):
                if i % 2 == 0:
                    a = self.ttt.possible_actions[0]
                else:
                    a = self.ttt.possible_actions[-1]
                self.assertTrue(self.ttt.is_active)
                self.ttt.execute_action(a)

        self.assertFalse(self.ttt.is_active)

    def test_is_active_after_win(self) -> None:
        action_space = [generate_horizontal_win_actions(), generate_vertical_win_actions(),
                        generate_diagonal_win_actions()]

        for win_actions in action_space:
            for i in range(len(win_actions)):
                self.ttt.reset()
                for j in range(len(win_actions[i])):
                    self.ttt.execute_action(win_actions[i][j])
                    if j < len(win_actions[i]) - 1:
                        self.ttt.execute_action(find_loser_action(self.ttt, win_actions[i]))
                        self.assertTrue(self.ttt.is_active)
                self.assertFalse(self.ttt.is_active)


if __name__ == '__main__':
    unittest.main()
