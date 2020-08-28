from agent.q_learning import SimpleDqn
from agent.utils import ReplayMemory
import torch
from typing import Tuple, Any
from .observable_environment import ObservableEnvironment


class DQN(SimpleDqn):
    def __init__(self, state_size: int, all_possible_actions: Tuple[Any, ...], training_amount: int = 100000,
                 replay_memory_size: int = 64000, batch_size: int = 512, alpha: float = 0.001,
                 gamma: int = 0.999) -> None:
        super().__init__(state_size=state_size, all_possible_actions=all_possible_actions,
                         training_amount=training_amount, alpha=alpha, gamma=gamma)

        self._replay_memory = ReplayMemory(memory_size=replay_memory_size, batch_size=batch_size)
        self._batch_size = batch_size

        self._target_net = self._create_neural_network()
        self._target_net.load_state_dict(self._policy_net.state_dict())  # Load weights from policy to target net
        self._target_net.eval()  # Target net in evaluation mode

    def observe_environment(self, env: ObservableEnvironment) -> None:
        self._update_replay_memory(env)
        super().observe_environment(env)

    def _update_network(self, env: ObservableEnvironment) -> None:
        if self._replay_memory.can_sample():
            self._policy_net.train()

            batch = self._replay_memory.get_sample()
            experiences = zip(*batch)

            previous_states, previous_actions, rewards, next_states, is_terminals = map(torch.cat, experiences)
            not_is_terminals = is_terminals.logical_not()

            target_q = torch.zeros(self._batch_size, device=self._device)

            pa = previous_actions.unsqueeze(-1)
            previous_q = self._policy_net(previous_states).gather(dim=1, index=pa)
            target_q[is_terminals] = rewards[is_terminals]

            if torch.sum(not_is_terminals).item() > 0:
                next_q = self._target_net(next_states[not_is_terminals]).max(dim=1).values.detach()
                target_q[not_is_terminals] = next_q * self._gamma + rewards[not_is_terminals]

            # target_q.clamp_(min=-1.0, max=1.0)

            loss = self._loss(previous_q, target_q.unsqueeze(dim=-1))
            self._optim.zero_grad()
            loss.backward()
            self._optim.step()

            if self._step_cnt % 10 == 0:
                self._target_net.load_state_dict(self._policy_net.state_dict())

            # self.print_special_case_info()

    def _update_replay_memory(self, env: ObservableEnvironment) -> None:
        if self._previous_action is not None and self._previous_state is not None:
            experience = (self._previous_state,
                          torch.as_tensor([self._actions_indices[self._previous_action]], dtype=torch.int64).to(
                              self._device),
                          torch.as_tensor([env.reward], dtype=torch.float32).to(self._device),
                          torch.as_tensor([env.state], dtype=torch.float32).to(self._device),
                          torch.as_tensor([env.is_terminal], dtype=torch.bool).to(self._device)
                          )
            self._replay_memory.insert(experience)

    def _write_to_tensorboard(self) -> None:
        if self._replay_memory.can_sample():
            super()._write_to_tensorboard()
