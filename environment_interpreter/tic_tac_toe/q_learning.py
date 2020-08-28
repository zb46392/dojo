from .. import Base
from agent.q_learning import ObservableEnvironment
from environment import TicTacToe
from typing import List, Tuple


class QLearning(Base):
    def __init__(self, environment: TicTacToe) -> None:
        super().__init__()
        self._environment = environment
        self._mark = None

    def interpret(self) -> ObservableEnvironment:
        return ObservableEnvironment(
            state=self._interpret_state(),
            reward=self._interpret_reward(),
            is_terminal=self._interpret_is_terminal()
        )

    def _interpret_state(self) -> Tuple[int, ...]:
        state = self._environment.state

        if self._environment.turn == 'o':
            state = self._inverse_state()

        return self._generate_observable_state(state)

    def _interpret_reward(self) -> float:
        if self._environment.is_active:
            self._mark = self._environment.turn

        return self._calculate_reward()

    def _interpret_is_terminal(self) -> bool:
        return not self._environment.is_active

    def _inverse_state(self) -> List[List[str]]:
        i_state = []

        for row in self._environment.state:
            i_row = self._inverse_row(row)
            i_state.append(i_row)

        return i_state

    @staticmethod
    def _generate_observable_state(state: List[List[str]]) -> Tuple[int, ...]:
        observable_state = []
        for row in state:
            for cell in row:
                if cell == 'x':
                    observable_state.append(0)
                    observable_state.append(1)
                elif cell == 'o':
                    observable_state.append(1)
                    observable_state.append(0)
                else:
                    observable_state.append(0)
                    observable_state.append(0)

        return tuple(observable_state)

    def _calculate_reward(self) -> float:
        winner = self._environment.winner

        if winner is not None and winner == self._mark:
            return 1.0
        elif winner is not None and winner != self._mark:
            return -1.0
        else:
            return 0.0

    def _inverse_row(self, row: List[str]) -> List[str]:
        i_row = []
        for cell in row:
            i_row.append(self._inverse_cell(cell))

        return i_row

    @staticmethod
    def _inverse_cell(cell: str) -> str:
        if cell == 'o':
            return 'x'
        elif cell == 'x':
            return 'o'
        else:
            return cell
