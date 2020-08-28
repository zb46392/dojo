from .. import Base
from typing import List, Optional, Tuple


class TicTacToe(Base):
    def __init__(self, first_turn: Optional[str] = 'x') -> None:
        super().__init__()
        self._first_turn = first_turn
        self._rectify_first_turn()
        self._turn = self._first_turn
        self._win_position = None
        self._winner = None

    @property
    def turn(self) -> str:
        return self._turn

    @property
    def win_position(self) -> Optional[Tuple[Tuple[int, int], ...]]:
        return self._win_position

    @property
    def winner(self) -> Optional[str]:
        return self._winner

    def execute_action(self, action: Tuple[int, int]) -> None:
        if action in self._possible_actions:
            self._update_state(action)
            self._update_win_position()
            self._update_winner()
            self._remove_possible_action(action)
            self._toggle_turn()
            self._update_is_active()

    def reset(self) -> None:
        super(TicTacToe, self).reset()
        self._turn = self._first_turn
        self._win_position = None
        self._winner = None

    def get_state_as_string(self) -> str:
        total_rows = len(self._state)
        total_cells = len(self._state[total_rows - 1]) if total_rows > 0 else 0
        as_str = ''

        for i in range(total_rows):
            for j in range(total_cells):
                cell_value = self._state[i][j]

                if i < total_rows - 1:
                    if cell_value != '':
                        str_value = f'_{cell_value.upper()}_'
                    else:
                        str_value = '___'
                else:
                    if cell_value != '':
                        str_value = f' {cell_value.upper()} '
                    else:
                        str_value = '   '

                if j < total_cells - 1:
                    str_value += '|'

                as_str += str_value
            as_str += '\n'

        return as_str

    def _rectify_first_turn(self) -> None:
        if isinstance(self._first_turn, str):
            self._first_turn_as_lower_xo()
        else:
            self._first_turn = 'x'

    def _create_init_state(self) -> List[List[str]]:
        return [['', '', ''], ['', '', ''], ['', '', '']]

    def _create_init_actions(self) -> Tuple[Tuple[int, int], ...]:
        return (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)

    def _first_turn_as_lower_xo(self) -> None:
        turn = self._first_turn.lower()
        if turn != 'x' and turn != 'o':
            self._first_turn = 'x'
        else:
            self._first_turn = turn

    def _update_state(self, action: Tuple[int, int]) -> None:
        i, j = action
        self._state[i][j] = self._turn

    def _update_win_position(self) -> None:
        self._win_position = self._find_win_position()

    def _update_winner(self) -> None:
        if self._win_position is not None:
            self._winner = self._turn

    def _remove_possible_action(self, action: Tuple[int, int]) -> None:
        possible_actions = []
        for pa in self._possible_actions:
            if not pa == action:
                possible_actions.append(pa)

        self._possible_actions = tuple(possible_actions)

    def _update_is_active(self) -> None:
        self._is_active = len(self._possible_actions) > 0
        if self._winner is not None:
            self._is_active = False

    def _toggle_turn(self) -> None:
        if self._turn == 'x':
            self._turn = 'o'
        else:
            self._turn = 'x'

    def _find_win_position(self) -> Optional[Tuple[Tuple[int, int], ...]]:
        win_position = self._find_win_position_in_horizontal_positions()
        if win_position is not None:
            return win_position

        win_position = self._find_win_position_in_vertical_positions()
        if win_position is not None:
            return win_position

        win_position = self._find_win_position_in_diagonal_positions()
        return win_position

    def _find_win_position_in_horizontal_positions(self) -> Optional[Tuple[Tuple[int, int], ...]]:
        h_pos = self._generate_horizontal_win_positions()
        return self._find_winning_position_from_positions(h_pos)

    def _find_win_position_in_vertical_positions(self) -> Optional[Tuple[Tuple[int, int], ...]]:
        v_pos = self._generate_vertical_win_positions()
        return self._find_winning_position_from_positions(v_pos)

    def _find_win_position_in_diagonal_positions(self) -> Optional[Tuple[Tuple[int, int], ...]]:
        d_pos = self._generate_diagonal_win_positions()
        return self._find_winning_position_from_positions(d_pos)

    def _generate_horizontal_win_positions(self) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
        positions = []

        for i in range(len(self._state)):
            h = []
            for j in range(len(self._state[i])):
                h.append((i, j))
            positions.append(tuple(h))

        return tuple(positions)

    def _generate_vertical_win_positions(self) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
        positions = []
        rows = len(self._state)
        columns = len(self._state[rows - 1])

        for j in range(rows):
            v = []
            for i in range(columns):
                v.append((i, j))
            positions.append(tuple(v))

        return tuple(positions)

    def _generate_diagonal_win_positions(self) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
        return ((0, 0), (1, 1), (2, 2)), ((0, 2), (1, 1), (2, 0))

    def _find_winning_position_from_positions(self, positions: Tuple[Tuple[Tuple[int, int], ...], ...]) \
            -> Optional[Tuple[Tuple[int, int], ...]]:

        for position in positions:
            winner = self._find_winner_in_position(position)

            if winner is not None:
                return position

        return None

    def _find_winner_in_position(self, position: Tuple[Tuple[int, int], ...]) -> Optional[str]:
        x_cnt = 0
        o_cnt = 0

        for i, j in position:
            if self._state[i][j] == 'x':
                x_cnt += 1
            elif self._state[i][j] == 'o':
                o_cnt += 1

        if x_cnt == 3:
            return 'x'
        elif o_cnt == 3:
            return 'o'
        else:
            return None
