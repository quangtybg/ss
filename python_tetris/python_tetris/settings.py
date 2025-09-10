
# Gameplay & UI settings

# Board
COLS = 10
ROWS = 20
BLOCK = 32  # pixel size of a cell
BORDER = 2  # grid line width

# Window
MARGIN = 20
SIDE_PANEL = 200
FPS = 60

# Gravity (seconds per row) by level (min at higher level)
LEVEL_SPEEDS = [
    0.8, 0.72, 0.63, 0.55, 0.47, 0.40, 0.32, 0.27, 0.22, 0.18,
    0.16, 0.14, 0.12, 0.11, 0.10, 0.09, 0.085, 0.08, 0.075, 0.07
]

# Scoring
SCORES = {1: 100, 2: 300, 3: 500, 4: 800}
SOFT_DROP_POINTS = 1
HARD_DROP_POINTS_PER_CELL = 2

# Movement repeat (DAS/ARR-like)
DAS_MS = 160  # delayed auto shift
ARR_MS = 35   # auto repeat rate

# Colors (R, G, B)
COLORS = {
    "bg": (15, 15, 22),
    "grid": (35, 35, 50),
    "text": (230, 230, 235),
    "ghost": (120, 120, 140),

    "I": (80, 200, 255),
    "O": (240, 230, 90),
    "T": (200, 120, 250),
    "S": (120, 220, 120),
    "Z": (240, 120, 120),
    "J": (120, 140, 250),
    "L": (250, 170, 70),
}
