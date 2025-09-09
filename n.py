#!/usr/bin/env python3
import time
import json
import os
import random
from typing import List, Tuple, Optional

SIZE = 15
INF = 10**9
CACHE_FILE = "gomoku_cache.json"

# Load cache
if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, "r") as f:
            TT = json.load(f)
    except:
        TT = {}
else:
    TT = {}

# Zobrist hashing
import random
ZOBRIST = [[[random.getrandbits(64) for _ in range(3)] for _ in range(SIZE)] for _ in range(SIZE)]
SYMBOL_TO_INDEX = {" ": 0, "X": 1, "O": 2}

def zobrist_hash(board: List[List[str]]) -> str:
    h = 0
    for r in range(SIZE):
        for c in range(SIZE):
            h ^= ZOBRIST[r][c][SYMBOL_TO_INDEX[board[r][c]]]
    return str(h)

# Evaluation (basic patterns)
def evaluate(board: List[List[str]], player: str) -> int:
    opp = "O" if player == "X" else "X"
    lines = []

    for r in range(SIZE):
        for c in range(SIZE):
            if c+4 < SIZE: lines.append([board[r][c+i] for i in range(5)])
            if r+4 < SIZE: lines.append([board[r+i][c] for i in range(5)])
            if r+4 < SIZE and c+4 < SIZE: lines.append([board[r+i][c+i] for i in range(5)])
            if r+4 < SIZE and c-4 >= 0: lines.append([board[r+i][c-i] for i in range(5)])

    score = 0
    for line in lines:
        s = "".join(line)
        if player*5 in s: score += 100000
        if player*4 in s and " " in s: score += 10000
        if player*3 in s and s.count(" ") >= 2: score += 1000
        if player*2 in s and s.count(" ") >= 3: score += 100
        if opp*5 in s: score -= 100000
        if opp*4 in s and " " in s: score -= 10000
        if opp*3 in s and s.count(" ") >= 2: score -= 1000
        if opp*2 in s and s.count(" ") >= 3: score -= 100
    return score

def generate_moves(board: List[List[str]]) -> List[Tuple[int, int]]:
    moves = set()
    radius = 2
    for r in range(SIZE):
        for c in range(SIZE):
            if board[r][c] != " ":
                for dr in range(-radius, radius+1):
                    for dc in range(-radius, radius+1):
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < SIZE and 0 <= nc < SIZE and board[nr][nc] == " ":
                            moves.add((nr, nc))
    if not moves:
        moves = {(SIZE//2, SIZE//2)}
    return list(moves)

# Minimax with alpha-beta
def minimax(board, depth, alpha, beta, maximizing, player, start, limit) -> int:
    if time.time() - start > limit:
        return 0

    key = zobrist_hash(board)
    if key in TT:
        entry = TT[key]
        if entry.get("depth", 0) >= depth and not entry.get("bad", False):
            return entry["score"]

    if depth == 0:
        return evaluate(board, player)

    moves = generate_moves(board)
    if not moves:
        return 0

    opp = "O" if player == "X" else "X"
    best = -INF if maximizing else INF

    for r, c in moves:
        board[r][c] = player if maximizing else opp
        val = minimax(board, depth-1, alpha, beta, not maximizing, player, start, limit)
        board[r][c] = " "

        if maximizing:
            best = max(best, val)
            alpha = max(alpha, val)
        else:
            best = min(best, val)
            beta = min(beta, val)

        if beta <= alpha:
            break

    if time.time() - start < limit:
        TT[key] = {"score": best, "depth": depth, "bad": False}
    return best

def get_move(board: List[List[str]], current_player: str, time_limit=0.9, max_depth_cap=4) -> Tuple[int, int]:
    start = time.time()
    key = zobrist_hash(board)

    best_move: Optional[Tuple[int, int]] = None
    best_score = -INF

    # Try cache
    if key in TT and not TT[key].get("bad", False):
        entry = TT[key]
        if entry["score"] > -5000:  # score quá xấu thì bỏ qua
            best_move = tuple(entry["move"])
            best_score = entry["score"]
            start_depth = entry["depth"] + 1
        else:
            start_depth = 1
    else:
        start_depth = 1

    # Iterative deepening
    for depth in range(start_depth, max_depth_cap+1):
        if time.time() - start > time_limit:
            break

        moves = generate_moves(board)
        random.shuffle(moves)

        for r, c in moves:
            if time.time() - start > time_limit:
                break
            board[r][c] = current_player
            val = minimax(board, depth-1, -INF, INF, False, current_player, start, time_limit)
            board[r][c] = " "
            if val > best_score:
                best_score = val
                best_move = (r, c)

        if best_move:
            TT[key] = {"move": best_move, "score": best_score, "depth": depth, "bad": False}

    # Nếu không có move nào
    if not best_move:
        best_move = (SIZE//2, SIZE//2)

    # Save cache
    with open(CACHE_FILE, "w") as f:
        json.dump(TT, f)

    return best_move

# Mark a bad state (learning)
def mark_bad_state(board: List[List[str]]):
    key = zobrist_hash(board)
    if key in TT:
        TT[key]["bad"] = True
    else:
        TT[key] = {"move": None, "score": -INF, "depth": 0, "bad": True}
    with open(CACHE_FILE, "w") as f:
        json.dump(TT, f)
