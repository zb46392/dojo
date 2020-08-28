from abc import ABC, abstractmethod
from typing import Any


class Base(ABC):

    @property
    def observable_environment(self) -> Any:
        return self.interpret()

    @abstractmethod
    def interpret(self) -> Any:
        pass
