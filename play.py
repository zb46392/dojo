#!/usr/bin/env python3

from environment import TicTacToe
from agent import Random as RandomAgent
from agent import QLearning as QLearningAgent
from agent import Base as BaseAgent
from agent import DQN as DqnAgent
from agent import SimpleDqn as SimpleDqnAgent
from time import time
from typing import Tuple
from environment_interpreter import EnvironmentInterpreterFactory


def main() -> None:
    training_amount = 100000
    env = TicTacToe()
    state_size = 18

    rnd = RandomAgent()
    # q_learn = QLearningAgent(training_amount)
    simple_dqn = SimpleDqnAgent(state_size=state_size, all_possible_actions=env.possible_actions,
                                training_amount=training_amount)
    # dqn = DqnAgent(state_size=state_size, all_possible_actions=env.possible_actions, training_amount=training_amount,
    #                replay_memory_size=64000, batch_size=512)

    a1 = rnd
    a1_train = False
    a2 = simple_dqn
    a2_train = True

    start = time()
    play(env, a1, a2, ep=10000, nbr_of_games_to_print=2, should_agent1_train=a1_train, should_agent2_train=a2_train,
         nbr_of_training=training_amount)
    end = time()
    print(f'\nElapsed time: {end - start}')


def play(t: TicTacToe, a1: BaseAgent, a2: BaseAgent, ep: int = 10000, nbr_of_games_to_print: int = 2,
         should_print_info: bool = True, should_agent1_train: bool = False, should_agent2_train: bool = False,
         nbr_of_training: int = 10000) -> Tuple[int, int]:
    t.reset()

    if should_agent1_train:
        if should_print_info:
            print('Agent 1 Training...')
        play(t=t, a1=a1, a2=RandomAgent(), ep=nbr_of_training, nbr_of_games_to_print=0, should_print_info=False)
        if should_print_info:
            print('Agent 1 Finished training...')

    if should_agent2_train:
        if should_print_info:
            print('Agent 2 Training...')
        play(t=t, a1=a2, a2=RandomAgent(), ep=nbr_of_training, nbr_of_games_to_print=0, should_print_info=False)
        if should_print_info:
            print('Agent 2 finished training...')

    wins = {a1: 0, a2: 0}
    turn = {a1: '', a2: ''}

    interpreter1 = EnvironmentInterpreterFactory.create(t, a1)
    interpreter2 = EnvironmentInterpreterFactory.create(t, a2)

    a = a1
    inter = interpreter1

    for i in range(ep):
        print(f'\rEpisode: {i}/{ep}\t{round((100 / ep) * i, 2)}%', end='')
        t.reset()
        a1.prepare_for_episode()
        a2.prepare_for_episode()
        turn[a1] = turn[a2] = ''

        while t.is_active:
            if turn[a] == '':
                turn[a] = t.turn

            a.observe_environment(inter.observable_environment)
            t.execute_action(a.choose_action(t.possible_actions))
            if i > (ep - (nbr_of_games_to_print + 1)):
                print(f'\n{t.get_state_as_string()}\n')
                if not t.is_active:
                    print(f'a1: {turn[a1].upper()}, a2: {turn[a2].upper()}')

            if a == a1:
                a = a2
                inter = interpreter2
            else:
                a = a1
                inter = interpreter1

        a1.observe_environment(interpreter1.observable_environment)
        a2.observe_environment(interpreter2.observable_environment)

        if turn[a1] == t.winner:
            wins[a1] += 1
        elif turn[a2] == t.winner:
            wins[a2] += 1

    print()
    if should_print_info:
        print(f'P1: {wins[a1]} :: P2 {wins[a2]}')

    return wins[a1], wins[a2]


if __name__ == '__main__':
    main()
