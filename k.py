#!/usr/bin/env python3
"""
Gomoku Bot:
 - Iterative deepening + minimax alpha-beta
 - Pattern-based evaluation (attack + defense)
 - Move generation radius-based
 - Move ordering (attack + defensive priority)
 - Cache best moves (resume search next time)
"""

import os
import json
import time
from typing import List, Tuple, Optional

BOARD_SIZE = 15
CACHE_FILE = "gomoku_cache.json"

# =========================
# Utility
# =========================

def switch_player(p: str) -> str:
    return "O" if p == "X" else "X"

def check_win(board: List[List[str]]) -> bool:
    """Kiểm tra có 5 quân liên tiếp không"""
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] == " ":
                continue
            player = board[r][c]
            for dr, dc in [(1,0),(0,1),(1,1),(1,-1)]:
                count = 0
                rr, cc = r, c
                while 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and board[rr][cc] == player:
                    count += 1
                    if count >= 5:
                        return True
                    rr += dr
                    cc += dc
    return False

# =========================
# Evaluation
# =========================

def evaluate(board: List[List[str]], player: str) -> int:
    """Đánh giá bàn cờ theo pattern-based scoring"""
    opponent = switch_player(player)
    score = 0

    patterns = {
        "11111": 1000000,   # 5 liên tiếp
        "011110": 100000,   # 4 mở
        "011112": 5000,     # 4 bị chặn 1 đầu
        "211110": 5000,
        "01110": 1000,      # 3 mở
        "010110": 800,
        "001112": 500,
        "211100": 500,
        "00110": 200,       # 2 mở
        "01010": 150,
        "0110": 100,
        "1": 10             # quân lẻ
    }

    def line_score(line: str, target: str) -> int:
        s = 0
        for pat, val in patterns.items():
            if target == 'me':
                s += line.count(pat.replace("1", "X").replace("2", "O")) * val
            else:
                s += line.count(pat.replace("1", "O").replace("2", "X")) * val
        return s

    lines = []
    for r in range(BOARD_SIZE):
        lines.append("".join(board[r][c] for c in range(BOARD_SIZE)))
    for c in range(BOARD_SIZE):
        lines.append("".join(board[r][c] for r in range(BOARD_SIZE)))
    for d in range(-BOARD_SIZE+1, BOARD_SIZE):
        lines.append("".join(board[r][r-d] for r in range(BOARD_SIZE) if 0 <= r-d < BOARD_SIZE))
        lines.append("".join(board[r][d+BOARD_SIZE-1-r] for r in range(BOARD_SIZE) if 0 <= d+BOARD_SIZE-1-r < BOARD_SIZE))

    for line in lines:
        line = line.replace(" ", "0").replace(player, "X").replace(opponent, "O")
        score += line_score(line, 'me')
        score -= line_score(line, 'opp')

    return score

# =========================
# Move Generation + Ordering
# =========================

def generate_moves(board: List[List[str]], radius: int = 2) -> List[Tuple[int, int]]:
    moves = set()
    has_piece = False
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != " ":
                has_piece = True
                for dr in range(-radius, radius+1):
                    for dc in range(-radius, radius+1):
                        rr, cc = r+dr, c+dc
                        if 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and board[rr][cc] == " ":
                            moves.add((rr, cc))
    if not has_piece:
        return [(BOARD_SIZE//2, BOARD_SIZE//2)]
    return list(moves)

def score_move(board: List[List[str]], move: Tuple[int,int], player: str) -> int:
    r, c = move
    opponent = switch_player(player)

    # thử đặt quân player
    board[r][c] = player
    score_self = evaluate(board, player)
    win_self = check_win(board)
    board[r][c] = " "

    # thử đặt quân đối thủ
    board[r][c] = opponent
    score_opp = evaluate(board, opponent)
    win_opp = check_win(board)
    board[r][c] = " "

    if win_self:
        return 10**8
    if win_opp:
        return 10**7

    defense_bonus = 0
    if score_opp > 5000:  # gần 4
        defense_bonus += 20000
    elif score_opp > 1000:  # gần 3
        defense_bonus += 5000

    return score_self - score_opp + defense_bonus

# =========================
# Minimax + AlphaBeta
# =========================

def minimax(board, depth, alpha, beta, maximizing, player, start_time, time_limit):
    if time.time() - start_time > time_limit:
        return evaluate(board, player), None
    if depth == 0 or check_win(board):
        return evaluate(board, player), None

    moves = generate_moves(board, radius=2)
    moves.sort(key=lambda m: score_move(board, m, player), reverse=True)

    best_move = None
    if maximizing:
        max_eval = -float("inf")
        for m in moves:
            r, c = m
            board[r][c] = player
            eval_score, _ = minimax(board, depth-1, alpha, beta, False, player, start_time, time_limit)
            board[r][c] = " "
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = m
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float("inf")
        opp = switch_player(player)
        for m in moves:
            r, c = m
            board[r][c] = opp
            eval_score, _ = minimax(board, depth-1, alpha, beta, True, player, start_time, time_limit)
            board[r][c] = " "
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = m
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval, best_move

# =========================
# Cache
# =========================

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)

def board_to_key(board: List[List[str]], player: str) -> str:
    return "".join("".join(row) for row in board) + "_" + player

# =========================
# Main API
# =========================

def get_move(board: List[List[str]], current_player: str, time_limit: float = 0.9, max_depth: int = 4) -> Tuple[int,int]:
    start_time = time.time()
    cache = load_cache()
    key = board_to_key(board, current_player)

    best_move: Optional[Tuple[int,int]] = None

    if key in cache:
        cached = cache[key]
        best_move = tuple(cached["move"])
        # nếu còn thời gian thì tìm tiếp
        if time.time() - start_time < time_limit:
            for depth in range(cached["depth"]+1, max_depth+1):
                eval_score, move = minimax(board, depth, -float("inf"), float("inf"), True, current_player, start_time, time_limit)
                if time.time() - start_time >= time_limit:
                    break
                if move is not None:
                    best_move = move
                    cache[key] = {"move": best_move, "depth": depth}
                    save_cache(cache)
    else:
        for depth in range(1, max_depth+1):
            eval_score, move = minimax(board, depth, -float("inf"), float("inf"), True, current_player, start_time, time_limit)
            if time.time() - start_time >= time_limit:
                break
            if move is not None:
                best_move = move
                cache[key] = {"move": best_move, "depth": depth}
                save_cache(cache)

    if best_move is None:
        moves = generate_moves(board)
        best_move = moves[0]

    return best_move
