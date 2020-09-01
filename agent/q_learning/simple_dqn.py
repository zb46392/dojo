from .observable_environment import ObservableEnvironment
from agent.base import Base
from agent.utils import ReplayMemory
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Tuple, Optional
from random import random, randint
from datetime import datetime


class SimpleDqn(Base):
    def __init__(self, state_size: int, all_possible_actions: Tuple[Any, ...], training_amount: int = 100000,
                 alpha: float = 0.001, gamma: int = 0.999) -> None:
        super().__init__()
        torch.set_grad_enabled(True)

        self._previous_state = None
        self._previous_action = None
        self._current_state = None
        self._current_action = None

        self._all_actions = all_possible_actions
        self._actions_indices = {self._all_actions[i]: i for i in range(len(self._all_actions))}
        self._previous_possible_actions = None

        self._alpha = alpha  # Learning rate
        self._gamma = gamma  # Future reward discount
        self._epsilon = 1.0  # Exploration rate
        self._epsilon_decay = self._epsilon / training_amount

        self._nbr_of_inputs = state_size
        self._nbr_of_outputs = len(self._all_actions)
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self._policy_net = self._create_neural_network()
        self._loss = nn.MSELoss()
        self._optim = optim.Adam(self._policy_net.parameters(), self._alpha)

        self._step_cnt = 0
        self._episode_cnt = 0
        self._rewards_sum = 0

        self._memory = ReplayMemory()

        self._summary_writer = None
        self._is_training = True

    def prepare_for_episode(self) -> None:
        self._previous_action = None
        self._previous_state = None

    def observe_environment(self, env: ObservableEnvironment) -> None:
        self._update_states(env)
        self._update_actions()
        self._store_experience(env)
        self._update_reward_sum(env)

        if self._is_training and self._summary_writer is None:
            self._summary_writer = SummaryWriter(
                comment=f' - {type(self).__name__} : ' +
                        f'alpha={self._alpha}, gamma={self._gamma}, train={int(self._epsilon / self._epsilon_decay)}'
            )
            self._summary_writer.add_graph(self._policy_net, self._current_state)

        if env.is_terminal:
            self._update_network(env)
            self._update_epsilon()
            self._episode_cnt += 1

            if self._is_training:
                self._write_to_tensorboard()

        if self._epsilon <= 0 and self._summary_writer is not None:
            self._summary_writer.close()
            self._summary_writer = None
            self._is_training = False

        self._step_cnt += 1

    def choose_action(self, actions: Tuple[Any, ...]) -> Optional[Any]:
        self._determine_current_action(actions)
        self._previous_possible_actions = actions

        return self._current_action

    def save_model(self) -> None:
        now = self._generate_time_string()
        file_name = f'dqn_model_{now}.weights'

        torch.save(self._policy_net.state_dict(), file_name)

    def load_model(self, file_path: str) -> None:
        state_dict = torch.load(file_path)
        self._policy_net.load_state_dict(state_dict)

    def _create_neural_network(self) -> nn.Sequential:
        network = nn.Sequential()
        network.add_module('linear_0', nn.Linear(in_features=self._nbr_of_inputs, out_features=1000))
        network.add_module('relu_0', nn.ReLU())
        network.add_module('linear_1', nn.Linear(in_features=1000, out_features=1000))
        network.add_module('relu_1', nn.ReLU())
        network.add_module('linear_2', nn.Linear(in_features=1000, out_features=self._nbr_of_outputs))
        network.to(self._device)
        return network

    def _update_states(self, env: ObservableEnvironment) -> None:
        if self._current_state is not None:
            self._previous_state = self._current_state

        self._current_state = torch.as_tensor([env.state], dtype=torch.float32).to(self._device)

    def _update_actions(self) -> None:
        self._previous_action = self._current_action
        self._current_action = None

    def _store_experience(self, env: ObservableEnvironment) -> None:
        if self._previous_action is not None and self._previous_state is not None:
            self._memory.insert(self._create_experience(env))

    def _create_experience(self, env: ObservableEnvironment) -> Tuple[Any, ...]:
        return (self._previous_state, self._previous_action, self._previous_possible_actions,
                env.reward, self._current_state, env.is_terminal)

    def _update_network(self, env: ObservableEnvironment) -> None:
        if self._previous_action is not None and self._previous_state is not None:
            self._policy_net.train()

            experiences = self._memory.flush()

            batch = zip(*experiences)

            previous_states, previous_actions, previous_possible_actions, rewards, next_states, is_terminals = batch

            future_discounted_reward = []
            for i in range(len(rewards) - 1, -1, -1):
                sum_reward = 0
                for j, reward in enumerate(rewards[i:]):
                    sum_reward += self._gamma ** j * reward
                future_discounted_reward.insert(0, sum_reward)

            previous_preds = self._policy_net(torch.cat(previous_states))
            next_preds = self._policy_net(torch.cat(next_states))

            next_qs = next_preds.max(dim=1).values.detach()

            target_preds = torch.zeros(previous_preds.shape).to(self._device)

            for i, ppa in enumerate(previous_possible_actions):
                indices = [self._actions_indices[a] for a in ppa]
                previous_action_i = self._actions_indices.get(previous_actions[i])
                target_preds[i][indices] = previous_preds[i][indices]
                if is_terminals[i]:
                    target_preds[i][previous_action_i] = future_discounted_reward[i]
                else:
                    target_preds[i][previous_action_i] = (next_qs[i] * self._gamma) + future_discounted_reward[i]

            loss = self._loss(previous_preds, target_preds)
            self._optim.zero_grad()
            loss.backward()
            self._optim.step()

            # self.print_special_case_info()

    def _update_reward_sum(self, env: ObservableEnvironment) -> None:
        self._rewards_sum += env.reward

    def _update_epsilon(self) -> None:
        if self._epsilon > 0:
            self._epsilon -= self._epsilon_decay

    def _write_to_tensorboard(self) -> None:
        self._summary_writer.add_scalar('Reward', self._rewards_sum, self._episode_cnt)

        if self._episode_cnt % 1000 == 0:
            for name, param in self._policy_net.named_parameters():
                self._summary_writer.add_histogram(name, param, self._episode_cnt)
                self._summary_writer.add_histogram(f'{name}_grad', param.grad, self._episode_cnt)

    def _determine_current_action(self, actions: Tuple[Any, ...]) -> None:
        if random() < self._epsilon:  # EXPLORE
            action_i = self._choose_random_action_index(actions)
        else:  # EXPLOIT
            action_i = self._choose_best_action_index(actions)

        self._current_action = self._all_actions[action_i]

    def _choose_random_action_index(self, actions: Tuple[Any, ...]) -> int:
        rnd_i = randint(0, len(actions) - 1)

        for i in range(len(self._all_actions)):
            if self._all_actions[i] == actions[rnd_i]:
                return i

    def _choose_best_action_index(self, actions: Tuple[Any, ...]) -> int:
        pred = self._inference(self._current_state)

        actions_indices_sorted = [i.item() for i in pred.sort(descending=True).indices.squeeze()]

        for i in actions_indices_sorted:
            if self._all_actions[i] in actions:
                return i

    def _inference(self, state: torch.Tensor) -> torch.Tensor:
        self._policy_net.eval()

        with torch.no_grad():
            pred = self._policy_net(state)

        return pred

    @staticmethod
    def _generate_time_string() -> str:
        return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    def print_special_case_info(self) -> None:
        #  _X_|_O_|_X_
        #  _X_|_O_|_O_
        #     |   |
        s = (
            0, 1, 1, 0, 0, 1,
            0, 1, 1, 0, 1, 0,
            0, 0, 0, 0, 0, 0
        )
        state = torch.as_tensor(s, dtype=torch.float32).to(self._device)

        pred = self._inference(state)
        v = [round(i.item(), 2) for i in pred.squeeze()]

        if self._step_cnt % 100 == 0:
            print(f'\r\t\t\t\t\t {v}                                 ', end='')
