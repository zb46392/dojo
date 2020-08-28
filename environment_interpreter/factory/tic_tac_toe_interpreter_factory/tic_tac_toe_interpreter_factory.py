from environment_interpreter import Base as Interpreter
from environment_interpreter import Basic as BasicInterpreter
from environment_interpreter.tic_tac_toe import QLearning as QLearningInterpreter
from environment import TicTacToe
from agent import Base as Agent, Random as RandomAgent
from agent import QLearning as QLearningAgent, SimpleDqn as SimpleDqnAgent, DQN as DqnAgent


class TicTacToeInterpreterFactory:
    @staticmethod
    def create(environment: TicTacToe, agent: Agent) -> Interpreter:
        if isinstance(agent, RandomAgent):
            return BasicInterpreter(environment)
        elif isinstance(agent, QLearningAgent) or isinstance(agent, DqnAgent) or isinstance(agent, SimpleDqnAgent):
            return QLearningInterpreter(environment)
        else:
            print(f'No interpreter for {type(agent)}, using default...')
            return BasicInterpreter(environment)
