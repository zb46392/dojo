from environment import TicTacToe
from environment_interpreter.tic_tac_toe import QLearning

import unittest
from unittest import TestCase


class TestTicTacToeToQlearningInterpreter(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._env = TicTacToe()
        self._inter = QLearning(self._env)

    def get_state(self) -> str:
        return self._inter.observable_environment.state

    def get_reward(self) -> float:
        return self._inter.observable_environment.reward

    def get_is_terminal(self) -> bool:
        return self._inter.observable_environment.is_terminal

    def test_empty_state(self) -> None:
        expected = (0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0)
        actual = self.get_state()

        self.assertEqual(expected, actual)

    def test_empty_reward(self) -> None:
        expected = 0.0
        actual = self.get_reward()

        self.assertEqual(expected, actual)

    def test_empty_is_terminal(self) -> None:
        expected = False
        actual = self.get_is_terminal()

        self.assertEqual(expected, actual)

    def test_second_move_state(self) -> None:
        self._env.execute_action((1, 1))
        self._env.execute_action((2, 2))
        expected = (
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 1, 0
        )
        actual = self.get_state()

        self.assertEqual(expected, actual)

    def test_inverse_state(self) -> None:
        self._env.execute_action((0, 0))
        self._env.execute_action((2, 1))
        self._env.execute_action((0, 2))

        expected = (
            1, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0
        )
        actual = self.get_state()

        self.assertEqual(expected, actual)

    def test_is_terminal_on_draw(self) -> None:
        self._env.execute_action((0, 0))
        self.assertEqual(False, self.get_is_terminal())
        self._env.execute_action((0, 1))
        self.assertEqual(False, self.get_is_terminal())
        self._env.execute_action((0, 2))
        self.assertEqual(False, self.get_is_terminal())
        self._env.execute_action((1, 0))
        self.assertEqual(False, self.get_is_terminal())
        self._env.execute_action((1, 1))
        self.assertEqual(False, self.get_is_terminal())
        self._env.execute_action((1, 2))
        self.assertEqual(False, self.get_is_terminal())
        self._env.execute_action((2, 1))
        self.assertEqual(False, self.get_is_terminal())
        self._env.execute_action((2, 0))
        self.assertEqual(False, self.get_is_terminal())
        self._env.execute_action((2, 2))
        self.assertEqual(True, self.get_is_terminal())

    def test_terminal_on_win(self) -> None:
        self._env.execute_action((0, 0))
        self.assertEqual(False, self.get_is_terminal())
        self._env.execute_action((0, 1))
        self.assertEqual(False, self.get_is_terminal())
        self._env.execute_action((1, 0))
        self.assertEqual(False, self.get_is_terminal())
        self._env.execute_action((1, 1))
        self.assertEqual(False, self.get_is_terminal())
        self._env.execute_action((2, 0))
        self.assertEqual(True, self.get_is_terminal())

    def test_reward_on_win(self) -> None:
        self._env.execute_action((0, 0))
        self._env.execute_action((1, 0))
        self.assertEqual(0.0, self.get_reward())

        self._env.execute_action((1, 1))
        self._env.execute_action((0, 2))
        self.assertEqual(0.0, self.get_reward())

        self._env.execute_action((2, 2))
        self.assertEqual(1.0, self.get_reward())

    def test_reward_on_lose(self) -> None:
        self._env.execute_action((1, 2))
        self._env.execute_action((1, 1))
        self.assertEqual(0.0, self.get_reward())

        self._env.execute_action((2, 0))
        self._env.execute_action((2, 1))
        self.assertEqual(0.0, self.get_reward())

        self._env.execute_action((0, 0))
        self._env.execute_action((0, 1))

        self.assertEqual(-1.0, self.get_reward())

    def test_reward_on_win_inverted_state(self) -> None:
        self._env.execute_action((2, 0))
        self.assertEqual(0.0, self.get_reward())

        self._env.execute_action((1, 2))
        self._env.execute_action((2, 2))

        self.assertEqual(0.0, self.get_reward())

        self._env.execute_action((1, 1))
        self._env.execute_action((0, 1))

        self.assertEqual(0.0, self.get_reward())

        self._env.execute_action((1, 0))
        self.assertEqual(1.0, self.get_reward())
        self.assertEqual(1.0, self.get_reward())

    def test_reward_on_lose_inverted_state(self) -> None:
        self._env.execute_action((0, 2))
        self.assertEqual(0.0, self.get_reward())

        self._env.execute_action((2, 1))
        self._env.execute_action((0, 1))
        self.assertEqual(0.0, self.get_reward())

        self._env.execute_action((1, 0))
        self._env.execute_action((0, 0))
        self.assertEqual(-1.0, self.get_reward())
        self.assertEqual(-1.0, self.get_reward())


if __name__ == '__main__':
    unittest.main()
