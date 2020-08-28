from . import Base
from environment import Base as Environment
from typing import Any


class Basic(Base):
    def __init__(self, environment: Environment) -> None:
        self._environment = environment

    def interpret(self) -> Any:
        return self._environment.state
