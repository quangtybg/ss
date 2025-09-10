
import random, time
from typing import List, Tuple, Optional
import pygame as pg

from settings import (COLS, ROWS, BLOCK, BORDER, MARGIN, SIDE_PANEL, FPS, LEVEL_SPEEDS,
                      SCORES, SOFT_DROP_POINTS, HARD_DROP_POINTS_PER_CELL, DAS_MS, ARR_MS, COLORS)
from tetrominoes import TETROMINOES
from board import Board

Vec = Tuple[int,int]

def rotate_matrix_cw(m: List[List[int]]) -> List[List[int]]:
    return [list(row) for row in zip(*m[::-1])]

def rotate_matrix_ccw(m: List[List[int]]) -> List[List[int]]:
    return [list(row) for row in zip(*m)][::-1]

def shape_cells(matrix: List[List[int]], origin: Vec) -> List[Vec]:
    (r0, c0) = origin
    res = []
    for r in range(4):
        for c in range(4):
            if matrix[r][c]:
                res.append((r0 + r, c0 + c))
    return res

class Piece:
    def __init__(self, id: str):
        self.id = id
        self.matrix = [row[:] for row in TETROMINOES[id]]
        self.origin = (0, COLS // 2 - 2)  # spawn near top-center
        self.lock_delay = 0.5  # seconds to lock after landing
        self.drop_time_accum = 0.0
        self.on_ground_time = 0.0

    def clone(self) -> "Piece":
        p = Piece(self.id)
        p.matrix = [row[:] for row in self.matrix]
        p.origin = self.origin
        p.lock_delay = self.lock_delay
        p.drop_time_accum = self.drop_time_accum
        p.on_ground_time = self.on_ground_time
        return p

class Game:
    def __init__(self):
        self.board = Board()
        self.bag: List[str] = []
        self.queue: List[Piece] = []
        self.hold: Optional[Piece] = None
        self.hold_used = False
        self.active: Piece = self._next_piece()
        self.level = 0
        self.score = 0
        self.lines = 0
        self.gravity_s = LEVEL_SPEEDS[self.level]
        self.drop_timer = 0.0
        self.paused = False
        self.game_over = False
        self.last_move_dir = 0  # -1 left, +1 right, 0 none
        self.move_held = False
        self.das_timer = 0.0
        self.arr_timer = 0.0

    def _refill_bag(self):
        self.bag = list(TETROMINOES.keys())
        random.shuffle(self.bag)

    def _next_piece(self) -> Piece:
        if not self.bag:
            self._refill_bag()
        pid = self.bag.pop()
        p = Piece(pid)
        return p

    def _ensure_queue(self):
        while len(self.queue) < 5:
            self.queue.append(self._next_piece())

    def spawn(self):
        self._ensure_queue()
        self.active = self.queue.pop(0)
        self.hold_used = False
        # Adjust spawn y if blocked
        for _ in range(2):
            if not self._valid(self.active):
                (r, c) = self.active.origin
                self.active.origin = (r+1, c)

        if not self._valid(self.active):
            self.game_over = True

    def _valid(self, piece: Piece) -> bool:
        return self.board.valid(shape_cells(piece.matrix, piece.origin))

    def _kick(self, piece: Piece, rotated: List[List[int]]) -> bool:
        # Simple wall kicks: try a few offsets
        (r, c) = piece.origin
        for off in [(0,0), (0,-1), (0,1), (-1,0), (1,0), (0,-2), (0,2)]:
            test = piece.clone()
            test.matrix = rotated
            test.origin = (r+off[0], c+off[1])
            if self._valid(test):
                piece.matrix = rotated
                piece.origin = test.origin
                return True
        return False

    def rotate(self, cw=True):
        if self.game_over or self.paused: return
        if self.active.id == "O":
            return  # O doesn't change
        mat = self.active.matrix
        rotated = rotate_matrix_cw(mat) if cw else rotate_matrix_ccw(mat)
        self._kick(self.active, rotated)

    def move(self, dx: int):
        if self.game_over or self.paused: return
        (r, c) = self.active.origin
        test = self.active.clone()
        test.origin = (r, c+dx)
        if self._valid(test):
            self.active.origin = test.origin

    def soft_drop(self) -> bool:
        if self.game_over or self.paused: return False
        (r, c) = self.active.origin
        test = self.active.clone()
        test.origin = (r+1, c)
        if self._valid(test):
            self.active.origin = test.origin
            self.score += SOFT_DROP_POINTS
            return True
        return False

    def hard_drop(self):
        if self.game_over or self.paused: return
        dist = 0
        while self.soft_drop():
            dist += 1
        self.score += dist * HARD_DROP_POINTS_PER_CELL
        self.lock_piece()

    def can_fall(self) -> bool:
        (r, c) = self.active.origin
        test = self.active.clone()
        test.origin = (r+1, c)
        return self._valid(test)

    def tick_gravity(self, dt: float):
        if self.game_over or self.paused: return
        self.active.drop_time_accum += dt
        if self.can_fall():
            if self.active.drop_time_accum >= self.gravity_s:
                self.active.drop_time_accum = 0.0
                self.active.origin = (self.active.origin[0]+1, self.active.origin[1])
        else:
            self.active.on_ground_time += dt
            if self.active.on_ground_time >= self.active.lock_delay:
                self.lock_piece()

    def lock_piece(self):
        cells = shape_cells(self.active.matrix, self.active.origin)
        self.board.place(cells, self.active.id)
        cleared = self.board.clear_lines()
        if cleared:
            self.lines += cleared
            self.score += SCORES.get(cleared, 0) * (self.level + 1)
            # Level progression: every 10 lines
            new_level = self.lines // 10
            if new_level > self.level:
                self.level = new_level
                self.gravity_s = LEVEL_SPEEDS[min(self.level, len(LEVEL_SPEEDS)-1)]
        self.spawn()

    def hold_piece(self):
        if self.game_over or self.paused: return
        if self.hold_used: return
        if self.hold is None:
            self.hold = self.active
            self.spawn()
        else:
            self.hold, self.active = self.active, self.hold
            # reset origin for the swapped-in piece
            self.active.origin = (0, COLS // 2 - 2)
            self.active.drop_time_accum = 0.0
            self.active.on_ground_time = 0.0
            if not self._valid(self.active):
                # nudge down to try spawn
                self.active.origin = (1, self.active.origin[1])
                if not self._valid(self.active):
                    self.game_over = True
        self.hold_used = True

    # ---------------- Rendering ----------------
    def draw(self, screen: pg.Surface, font: pg.font.Font):
        screen.fill(COLORS["bg"])
        # playfield rect
        left = MARGIN
        top = MARGIN
        w = COLS * BLOCK
        h = ROWS * BLOCK
        # grid
        for r in range(ROWS):
            for c in range(COLS):
                rect = pg.Rect(left + c*BLOCK, top + r*BLOCK, BLOCK, BLOCK)
                pg.draw.rect(screen, COLORS["grid"], rect, BORDER)

        # ghost
        ghost_origin = self._ghost_origin()
        ghost_cells = shape_cells(self.active.matrix, ghost_origin)
        for (r, c) in ghost_cells:
            rect = pg.Rect(left + c*BLOCK, top + r*BLOCK, BLOCK, BLOCK)
            pg.draw.rect(screen, COLORS["ghost"], rect, 2)

        # locked blocks
        for r in range(ROWS):
            for c in range(COLS):
                pid = self.board.grid[r][c]
                if pid:
                    rect = pg.Rect(left + c*BLOCK, top + r*BLOCK, BLOCK, BLOCK)
                    pg.draw.rect(screen, COLORS[pid], rect)
                    pg.draw.rect(screen, COLORS["bg"], rect, 2)

        # active piece
        for (r, c) in shape_cells(self.active.matrix, self.active.origin):
            rect = pg.Rect(left + c*BLOCK, top + r*BLOCK, BLOCK, BLOCK)
            pg.draw.rect(screen, COLORS[self.active.id], rect)
            pg.draw.rect(screen, COLORS["bg"], rect, 2)

        # side panel
        sx = left + w + MARGIN
        sy = top
        self._draw_text(screen, font, f"Score: {self.score}", (sx, sy))
        self._draw_text(screen, font, f"Level: {self.level}", (sx, sy+30))
        self._draw_text(screen, font, f"Lines: {self.lines}", (sx, sy+60))

        # Hold
        self._draw_text(screen, font, "Hold:", (sx, sy+110))
        if self.hold:
            self._draw_mini(screen, self.hold.matrix, self.hold.id, (sx, sy+140))

        # Next queue
        self._ensure_queue()
        self._draw_text(screen, font, "Next:", (sx, sy+220))
        qy = sy + 250
        for i in range(5):
            m = self.queue[i].matrix
            pid = self.queue[i].id
            self._draw_mini(screen, m, pid, (sx, qy))
            qy += 70

        if self.paused:
            self._draw_center_text(screen, font, "Paused")
        if self.game_over:
            self._draw_center_text(screen, font, "Game Over (R to restart)")

    def _draw_text(self, screen: pg.Surface, font: pg.font.Font, text: str, pos: Tuple[int,int]):
        surf = font.render(text, True, COLORS["text"])
        screen.blit(surf, pos)

    def _draw_center_text(self, screen: pg.Surface, font: pg.font.Font, text: str):
        surf = font.render(text, True, COLORS["text"])
        rect = surf.get_rect(center=screen.get_rect().center)
        screen.blit(surf, rect.topleft)

    def _draw_mini(self, screen: pg.Surface, matrix: List[List[int]], pid: str, pos: Tuple[int,int]):
        # Draw a tiny preview (scale down to 20px blocks)
        mini = 20
        (sx, sy) = pos
        # compute bounds to center the 4x4 shape
        cells = [(r,c) for r in range(4) for c in range(4) if matrix[r][c]]
        if not cells:
            return
        min_r = min(r for r,c in cells)
        max_r = max(r for r,c in cells)
        min_c = min(c for r,c in cells)
        max_c = max(c for r,c in cells)
        w = (max_c - min_c + 1) * mini
        h = (max_r - min_r + 1) * mini
        ox = sx + (SIDE_PANEL - 2*mini - w) // 2
        oy = sy + (60 - h) // 2
        for (r,c) in cells:
            rect = pg.Rect(ox + (c-min_c)*mini, oy + (r-min_r)*mini, mini, mini)
            pg.draw.rect(screen, COLORS[pid], rect)
            pg.draw.rect(screen, (10,10,10), rect, 1)

    def _ghost_origin(self) -> Vec:
        test = self.active.clone()
        while True:
            (r, c) = test.origin
            test.origin = (r+1, c)
            if not self._valid(test):
                return (r, c)

    # --------------- Input handling with DAS/ARR ----------------
    def handle_input(self, dt: float):
        keys = pg.key.get_pressed()
        if keys[pg.K_LEFT] or keys[pg.K_a]:
            self._repeat_move(-1, dt, keys)
        elif keys[pg.K_RIGHT] or keys[pg.K_d]:
            self._repeat_move(1, dt, keys)
        else:
            self.move_held = False
            self.das_timer = 0.0
            self.arr_timer = 0.0

        if keys[pg.K_DOWN] or keys[pg.K_s]:
            self.soft_drop()

    def _repeat_move(self, dir: int, dt: float, keys):
        if not self.move_held or dir != self.last_move_dir:
            self.move(dir)
            self.move_held = True
            self.last_move_dir = dir
            self.das_timer = 0.0
            self.arr_timer = 0.0
            return
        # holding same direction
        self.das_timer += dt * 1000
        if self.das_timer >= DAS_MS:
            self.arr_timer += dt * 1000
            while self.arr_timer >= ARR_MS:
                self.move(dir)
                self.arr_timer -= ARR_MS

