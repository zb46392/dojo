from .. import Base as Interpreter
from agent import Base as Agent
from environment import Base as Environment, TicTacToe
from .tic_tac_toe_interpreter_factory import TicTacToeInterpreterFactory


class EnvironmentInterpreterFactory:
    @staticmethod
    def create(environment: Environment, agent: Agent) -> Interpreter:
        if isinstance(environment, TicTacToe):
            return TicTacToeInterpreterFactory.create(environment, agent)
        else:
            raise Exception(f'No interpreter for environment: {environment.__class__}')
