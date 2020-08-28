from agent import Random as RandomAgent
import unittest
from unittest import TestCase


class TestRandom(TestCase):
    def test_choose_action(self) -> None:
        agent = RandomAgent()
        actions = ((10, 2), (-1, 8), ('A', 'x'), (('a', 1), ('b', -1)), ([1, 2]))
        action = agent.choose_action(actions)

        self.assertTrue(action in actions)

    def test_choose_from_empty_actions(self) -> None:
        agent = RandomAgent()
        actions = ()
        expected = None
        actual = agent.choose_action(actions)

        self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
