import unittest
from unittest import TestCase
from environment import TicTacToe
from random import randint


class TestState(TestCase):
    def setUp(self) -> None:
        super().setUp()

        self.ttt = TicTacToe()

    def test_empty_state(self) -> None:
        expected = [['', '', ''], ['', '', ''], ['', '', '']]
        actual = self.ttt.state

        self.assertEqual(expected, actual)

    def test_state_immutability_from_outside(self) -> None:
        state = self.ttt.state
        state[0] = ['x', 'x', 'x']
        expected = [['', '', ''], ['', '', ''], ['', '', '']]
        actual = self.ttt.state

        self.assertEqual(expected, actual)

    def test_empty_state_as_string(self) -> None:
        expected = '___|___|___\n___|___|___\n   |   |   \n'
        actual = self.ttt.get_state_as_string()

        self.assertEqual(expected, actual)

    def test_default_turn_selection(self) -> None:
        expected = 'x'
        actual = self.ttt.turn

        self.assertEqual(expected, actual)

    def test_state_after_action_execution(self) -> None:
        actions = self.ttt.possible_actions
        self.ttt.execute_action(actions[0])
        expected = [['x', '', ''], ['', '', ''], ['', '', '']]
        actual = self.ttt.state

        self.assertEqual(expected, actual)

    def test_state_after_second_action_execution(self) -> None:
        a1 = self.ttt.possible_actions[0]
        self.ttt.execute_action(a1)
        a2 = self.ttt.possible_actions[0]
        self.ttt.execute_action(a2)

        expected = [['x', 'o', ''], ['', '', ''], ['', '', '']]
        actual = self.ttt.state

        self.assertEqual(expected, actual)

    def test_state_after_random_action_execution(self) -> None:
        actions = self.ttt.possible_actions
        r_i = randint(0, len(actions) - 1)
        action = actions[r_i]
        turn = self.ttt.turn

        self.ttt.execute_action(action)
        element = self.ttt.state[action[0]][action[1]]
        self.assertTrue(turn == element)

    def test_state_after_reset(self) -> None:
        expected = self.ttt.state

        a = self.ttt.possible_actions[0]
        self.ttt.execute_action(a)
        self.ttt.reset()

        actual = self.ttt.state

        self.assertEqual(expected, actual)

    def test_state_as_string_after_first_move(self) -> None:
        self.ttt.execute_action((1, 1))
        expected = '___|___|___\n___|_X_|___\n   |   |   \n'
        actual = self.ttt.get_state_as_string()

        self.assertEqual(expected, actual)

    def test_state_as_string_after_second_move(self) -> None:
        self.ttt.execute_action((1, 1))
        self.ttt.execute_action((2, 2))

        expected = '___|___|___\n___|_X_|___\n   |   | O \n'
        actual = self.ttt.get_state_as_string()

        self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
