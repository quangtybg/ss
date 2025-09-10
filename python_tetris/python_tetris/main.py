
import pygame as pg
from settings import (COLS, ROWS, BLOCK, MARGIN, SIDE_PANEL, FPS, COLORS)
from game import Game

def main():
    pg.init()
    pg.display.set_caption("Python Tetris")
    width = MARGIN + COLS*BLOCK + MARGIN + SIDE_PANEL + MARGIN
    height = MARGIN + ROWS*BLOCK + MARGIN
    screen = pg.display.set_mode((width, height))
    clock = pg.time.Clock()
    font = pg.font.SysFont("consolas", 22)

    g = Game()
    g.spawn()  # start

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    running = False
                elif event.key in (pg.K_UP, pg.K_x):
                    g.rotate(cw=True)
                elif event.key == pg.K_z:
                    g.rotate(cw=False)
                elif event.key == pg.K_SPACE:
                    g.hard_drop()
                elif event.key == pg.K_c:
                    g.hold_piece()
                elif event.key == pg.K_p:
                    g.paused = not g.paused
                elif event.key == pg.K_r:
                    g = Game()
                    g.spawn()

        g.handle_input(dt)
        g.tick_gravity(dt)

        g.draw(screen, font)
        pg.display.flip()

    pg.quit()

if __name__ == "__main__":
    main()
