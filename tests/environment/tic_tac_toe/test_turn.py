import unittest
from unittest import TestCase
from environment import TicTacToe


class TestTurn(TestCase):
    def setUp(self) -> None:
        super().setUp()

        self.ttt = TicTacToe()

    def test_first_turn_selection(self) -> None:
        t = TicTacToe(first_turn='x')

        expected = 'x'
        actual = t.turn

        self.assertEqual(expected, actual)

        t = TicTacToe(first_turn='o')

        expected = 'o'
        actual = t.turn

        self.assertEqual(expected, actual)

    def test_upper_first_turn_selection(self) -> None:
        t = TicTacToe(first_turn='O')
        expected = 'o'
        actual = t.turn

        self.assertEqual(expected, actual)

    def test_invalid_first_turn_selection(self) -> None:
        t = TicTacToe(first_turn='A')
        expected = 'x'
        actual = t.turn

        self.assertEqual(expected, actual)

    def test_invalid_first_turn_type(self) -> None:
        t = TicTacToe(first_turn=-1)
        expected = 'x'
        actual = t.turn

        self.assertEqual(expected, actual)

    def test_turn_after_first_action_execution(self) -> None:
        a = self.ttt.possible_actions[0]
        self.ttt.execute_action(a)

        expected = 'o'
        actual = self.ttt.turn

        self.assertEqual(expected, actual)

    def test_turn_after_second_action_execution(self) -> None:
        a = self.ttt.possible_actions[0]
        self.ttt.execute_action(a)

        a = self.ttt.possible_actions[0]
        self.ttt.execute_action(a)

        expected = 'x'
        actual = self.ttt.turn

        self.assertEqual(expected, actual)

    def test_turn_after_reset(self) -> None:
        t = TicTacToe(first_turn='o')
        expected = t.turn

        a = t.possible_actions[0]
        t.execute_action(a)
        t.reset()

        actual = t.turn

        self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
