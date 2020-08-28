from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Tuple


class Base(ABC):
    def __init__(self) -> None:
        self._state = self._create_init_state()
        self._is_active = True
        self._possible_actions = self._create_init_actions()

    @property
    def state(self) -> Any:
        return deepcopy(self._state)

    @property
    def is_active(self) -> bool:
        return self._is_active

    @property
    def possible_actions(self) -> Tuple[Any, ...]:
        return deepcopy(self._possible_actions)

    def reset(self) -> None:
        self._state = self._create_init_state()
        self._is_active = True
        self._possible_actions = self._create_init_actions()

    @abstractmethod
    def execute_action(self, action: Any) -> None:
        pass

    @abstractmethod
    def _create_init_state(self) -> Any:
        pass

    @abstractmethod
    def _create_init_actions(self) -> Tuple[Any, ...]:
        pass
