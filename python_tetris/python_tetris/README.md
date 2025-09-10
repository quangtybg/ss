
# Python Tetris (Pygame)

A clean, beginner-friendly yet fully featured Tetris clone written in Python using Pygame.

## Features
- 10×20 playfield, 7 classic tetrominoes (I, O, T, S, Z, J, L)
- Smooth movement with DAS/ARR feel (key repeat), soft/hard drop
- Hold piece (press **C**), 5-piece preview queue
- Line clear, scoring, levels, and speed ramp
- Pause (**P**), Restart (**R**), Quit (**Esc**)
- Configurable in `settings.py`

## Requirements
- Python 3.9+
- Pygame 2.5+

Install dependencies:
```bash
pip install -r requirements.txt
```

Run:
```bash
python main.py
```

## Controls
- **Left/Right**: Move piece
- **Down**: Soft drop
- **Up / X**: Rotate clockwise
- **Z**: Rotate counter-clockwise
- **Space**: Hard drop
- **C**: Hold piece
- **P**: Pause
- **R**: Restart
- **Esc**: Quit

## Project Structure
```
python_tetris/
├── assets/
├── board.py
├── game.py
├── main.py
├── settings.py
├── tetrominoes.py
├── requirements.txt
└── README.md
```

## Notes
- Rotation is a simple SRS-like system with basic wall kicks for common cases.
- Sound effects are optional (kept simple without external files).
- Code is well-commented for learning and extension.
