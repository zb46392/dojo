import unittest
from unittest import TestCase
from environment import TicTacToe
from random import randint


class TestActions(TestCase):
    def setUp(self) -> None:
        super().setUp()

        self.ttt = TicTacToe()

    def test_possible_actions_on_empty_state(self) -> None:
        expected = ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2))
        actual = self.ttt.possible_actions

        self.assertEqual(expected, actual)

    def test_possible_actions_after_first_action_execution(self) -> None:
        actions = self.ttt.possible_actions
        self.ttt.execute_action(actions[0])
        expected = ((0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2))
        actual = self.ttt.possible_actions

        self.assertEqual(expected, actual)

    def test_possible_actions_after_random_action_execution(self) -> None:
        old_actions = self.ttt.possible_actions
        r_i = randint(0, len(old_actions) - 1)
        action = old_actions[r_i]
        self.ttt.execute_action(action)
        new_actions = self.ttt.possible_actions

        self.assertTrue(action not in new_actions and len(new_actions) == (len(old_actions) - 1))

    def test_on_invalid_action(self) -> None:
        action = (-1, 99)
        self.ttt.execute_action(action)

        expected_state = [['', '', ''], ['', '', ''], ['', '', '']]
        actual_state = self.ttt.state

        expected_turn = 'x'
        actual_turn = self.ttt.turn

        self.assertEqual(expected_state, actual_state)
        self.assertEqual(expected_turn, actual_turn)


if __name__ == '__main__':
    unittest.main()
