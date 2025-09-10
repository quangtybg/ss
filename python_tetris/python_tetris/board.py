
from typing import List, Optional, Tuple
from settings import COLS, ROWS

Cell = Optional[str]  # None or a single-letter piece name

class Board:
    def __init__(self):
        self.grid: List[List[Cell]] = [[None for _ in range(COLS)] for _ in range(ROWS)]

    def inside(self, r: int, c: int) -> bool:
        return 0 <= r < ROWS and 0 <= c < COLS

    def empty(self, r: int, c: int) -> bool:
        return self.inside(r, c) and self.grid[r][c] is None

    def place(self, cells: List[Tuple[int,int]], id: str):
        for r, c in cells:
            if self.inside(r, c):
                self.grid[r][c] = id

    def valid(self, cells: List[Tuple[int,int]]) -> bool:
        for r, c in cells:
            if not self.inside(r, c) or self.grid[r][c] is not None:
                return False
        return True

    def clear_lines(self) -> int:
        new_rows = [row for row in self.grid if any(cell is None for cell in row)]
        cleared = ROWS - len(new_rows)
        if cleared:
            for _ in range(cleared):
                new_rows.insert(0, [None for _ in range(COLS)])
            self.grid = new_rows
        return cleared

    def top_out(self) -> bool:
        # Game over if any blocks are in row 0 after placement
        return any(cell is not None for cell in self.grid[0])
