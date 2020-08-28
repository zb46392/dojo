from .. import Base
from typing import Tuple, Any, Optional
from random import randint


class Random(Base):
    def observe_environment(self, env: Any) -> None:
        pass

    def choose_action(self, actions: Tuple[Any, ...]) -> Optional[Any]:
        total_actions = len(actions)

        if total_actions > 0:
            rnd = randint(0, total_actions - 1)
            return actions[rnd]
        else:
            return None

    def prepare_for_episode(self) -> None:
        pass
