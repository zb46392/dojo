from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple


class Base(ABC):
    @abstractmethod
    def observe_environment(self, env: Any) -> None:
        pass

    @abstractmethod
    def choose_action(self, actions: Tuple[Any, ...]) -> Optional[Any]:
        pass

    @abstractmethod
    def prepare_for_episode(self) -> None:
        pass
