import random
import time
import hashlib
import json
from multiprocessing import Manager

BOARD_SIZE = 15
CACHE_FILE = "gomoku_cache.json"
CACHE = {}

# -------------------- Utils --------------------

def empty_board():
    return [[" " for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

def board_to_hash(board, player):
    """Sinh hash duy nhất cho trạng thái bàn cờ + lượt đi"""
    flat = "".join("".join(row) for row in board) + player
    return hashlib.md5(flat.encode()).hexdigest()

def save_cache():
    with open(CACHE_FILE, "w") as f:
        json.dump(CACHE, f)

def load_cache():
    global CACHE
    try:
        with open(CACHE_FILE, "r") as f:
            CACHE = json.load(f)
    except FileNotFoundError:
        CACHE = {}

def check_win(board, player):
    """Kiểm tra thắng (5 liên tiếp)"""
    directions = [(1,0),(0,1),(1,1),(1,-1)]
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != player:
                continue
            for dr,dc in directions:
                cnt = 0
                nr, nc = r, c
                while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr][nc] == player:
                    cnt += 1
                    if cnt >= 5:
                        return True
                    nr += dr
                    nc += dc
    return False

# -------------------- Evaluation --------------------

PATTERNS = {
    "XXXXX": 1000000,  # thắng ngay
    "XXXX ": 50000, " XXXXT": 50000, "TXXXXT": 100000,  # 4
    "XXX ": 5000, " XXXT": 5000, "TXXXT": 8000,  # 3
    "XX ": 500, "XXT": 500,  # 2
}

def evaluate(board, player):
    """Heuristic đơn giản dựa vào pattern"""
    score = 0
    opp = "O" if player == "X" else "X"

    lines = []
    # hàng
    for row in board:
        lines.append("".join(row))
    # cột
    for c in range(BOARD_SIZE):
        lines.append("".join(board[r][c] for r in range(BOARD_SIZE)))
    # chéo xuống
    for d in range(-BOARD_SIZE+1, BOARD_SIZE):
        lines.append("".join(board[r][r-d] for r in range(BOARD_SIZE) if 0 <= r-d < BOARD_SIZE))
    # chéo lên
    for d in range(0, 2*BOARD_SIZE-1):
        lines.append("".join(board[r][d-r] for r in range(BOARD_SIZE) if 0 <= d-r < BOARD_SIZE))

    for line in lines:
        for pat, val in PATTERNS.items():
            if pat.replace("T", "") in line:
                score += val
            if pat.replace("T", "").replace("X","O") in line:
                score -= val * 0.9  # ưu tiên chặn đối thủ

    return score

# -------------------- Search --------------------

def search_best(board, player, time_limit=0.3, max_depth=2):
    start = time.time()
    best_move, best_score = None, -float("inf")
    opp = "O" if player == "X" else "X"

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != " ":
                continue
            board[r][c] = player
            score = evaluate(board, player)
            board[r][c] = " "

            if score > best_score:
                best_score = score
                best_move = (r, c)

            if time.time() - start > time_limit:
                return best_move
    return best_move

def get_move(board, player, time_limit=0.5, max_depth=2):
    """Hàm chính chọn nước đi"""
    key = board_to_hash(board, player)
    if key in CACHE:
        return tuple(CACHE[key])

    move = search_best(board, player, time_limit, max_depth)
    if move:
        CACHE[key] = move
    return move
