#!/usr/bin/env python3
"""
Gomoku bot with:
 - Zobrist hashing + persistent cache (move, score, depth)
 - Iterative deepening + negamax (alpha-beta)
 - Pattern-based evaluation (attack/defense)
 - Candidate move generation (neighborhood)
 - get_move(board, current_player, time_limit, max_depth)
"""

import os
import json
import time
import random
from typing import List, Tuple, Dict, Optional

SIZE = 15
WIN_LEN = 5
CACHE_FILE = "gomoku_cache_v2.json"  # new structured cache file

# Pattern scores (you can tune)
PATTERN_SCORES = {
    "five": 1_000_000,
    "open_four": 100_000,
    "closed_four": 10_000,
    "open_three": 1_000,
    "closed_three": 100,
    "open_two": 50,
    "closed_two": 10,
}

def opponent(p: str) -> str:
    return "O" if p == "X" else "X"

# -----------------------------
# Zobrist hashing utilities
# -----------------------------
def init_zobrist(seed: Optional[int] = None):
    """Return a new zobrist table: dict with 2 arrays (X and O) SIZE*SIZE each and side random."""
    rnd = random.Random(seed)
    table = {
        "X": [rnd.getrandbits(64) for _ in range(SIZE*SIZE)],
        "O": [rnd.getrandbits(64) for _ in range(SIZE*SIZE)],
        "side": rnd.getrandbits(64)  # flip for side to move
    }
    return table

def board_zobrist_hash(board: List[List[str]], zobrist_table: Dict) -> int:
    h = 0
    for r in range(SIZE):
        for c in range(SIZE):
            v = board[r][c]
            if v == " ":
                continue
            idx = r*SIZE + c
            h ^= zobrist_table[v][idx]
    return h

# -----------------------------
# Cache: structured JSON
# -----------------------------
def load_cache_file():
    if not os.path.exists(CACHE_FILE):
        # create skeleton
        return {"zobrist_seed": None, "entries": {}}
    try:
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"zobrist_seed": None, "entries": {}}

def save_cache_file(cache_struct):
    tmp = CACHE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache_struct, f)
    os.replace(tmp, CACHE_FILE)

# -----------------------------
# Board helpers
# -----------------------------
def empty_board():
    return [[" " for _ in range(SIZE)] for __ in range(SIZE)]

def in_bounds(r, c):
    return 0 <= r < SIZE and 0 <= c < SIZE

def is_winner(board: List[List[str]], player: str) -> bool:
    # check 5 in a row efficiently
    for r in range(SIZE):
        for c in range(SIZE):
            if board[r][c] != player:
                continue
            # 4 directions
            if c + WIN_LEN <= SIZE and all(board[r][c+i] == player for i in range(WIN_LEN)):
                return True
            if r + WIN_LEN <= SIZE and all(board[r+i][c] == player for i in range(WIN_LEN)):
                return True
            if r + WIN_LEN <= SIZE and c + WIN_LEN <= SIZE and all(board[r+i][c+i] == player for i in range(WIN_LEN)):
                return True
            if r - WIN_LEN + 1 >= 0 and c + WIN_LEN <= SIZE and all(board[r-i][c+i] == player for i in range(WIN_LEN)):
                return True
    return False

# -----------------------------
# Candidate generation
# -----------------------------
def get_candidates(board: List[List[str]], radius: int = 2) -> List[Tuple[int,int]]:
    """Return list of empty cells near existing stones. If empty board, center."""
    cells = set()
    any_stone = False
    for r in range(SIZE):
        for c in range(SIZE):
            if board[r][c] != " ":
                any_stone = True
                for dr in range(-radius, radius+1):
                    for dc in range(-radius, radius+1):
                        nr, nc = r+dr, c+dc
                        if in_bounds(nr, nc) and board[nr][nc] == " ":
                            cells.add((nr, nc))
    if not any_stone:
        return [(SIZE//2, SIZE//2)]
    # convert to list and sort by simple heuristic: closer to center first (optional)
    return list(cells)

# -----------------------------
# Evaluation (pattern-based)
# -----------------------------
def evaluate_position(board: List[List[str]], player: str) -> int:
    """
    Evaluate entire board for 'player'. Combine attack + defense.
    We'll scan on all lines and detect patterns using string windows.
    """
    opp = opponent(player)
    total = 0

    # helper to generate lines
    lines = []

    # rows
    for r in range(SIZE):
        lines.append("".join(board[r]))
    # cols
    for c in range(SIZE):
        lines.append("".join(board[r][c] for r in range(SIZE)))
    # diag down-right
    for d in range(-SIZE+1, SIZE):
        s = []
        for r in range(SIZE):
            c = r - d
            if 0 <= c < SIZE:
                s.append(board[r][c])
        if len(s) >= 1:
            lines.append("".join(s))
    # diag up-right
    for d in range(0, 2*SIZE-1):
        s = []
        for r in range(SIZE):
            c = d - r
            if 0 <= c < SIZE:
                s.append(board[r][c])
        if len(s) >= 1:
            lines.append("".join(s))

    # check patterns in each line; we pad with spaces to detect open patterns
    for line in lines:
        padded = " " + line + " "
        # check for both player and opponent patterns, but attack patterns get positive,
        # opponent patterns add to score too (defense weight)
        # for player (attack)
        if player*5 in padded:
            total += PATTERN_SCORES["five"]
        if (" " + player*4 + " ") in padded:
            total += PATTERN_SCORES["open_four"]
        if (player*4 + " ") in padded or (" " + player*4) in padded:
            total += PATTERN_SCORES["closed_four"]
        if (" " + player*3 + " ") in padded:
            total += PATTERN_SCORES["open_three"]
        if (player*3 + " ") in padded or (" " + player*3) in padded:
            total += PATTERN_SCORES["closed_three"]
        if (" " + player*2 + " ") in padded:
            total += PATTERN_SCORES["open_two"]
        if (player*2 + " ") in padded or (" " + player*2) in padded:
            total += PATTERN_SCORES["closed_two"]

        # defense: weight opponent patterns higher to prioritize blocking
        if opp*5 in padded:
            total += PATTERN_SCORES["five"]
        if (" " + opp*4 + " ") in padded:
            total += PATTERN_SCORES["open_four"] * 0.9
        if (opp*4 + " ") in padded or (" " + opp*4) in padded:
            total += PATTERN_SCORES["closed_four"] * 0.7
        if (" " + opp*3 + " ") in padded:
            total += PATTERN_SCORES["open_three"] * 0.8
        if (opp*3 + " ") in padded or (" " + opp*3) in padded:
            total += PATTERN_SCORES["closed_three"] * 0.5

    return int(total)

# -----------------------------
# Negamax with alpha-beta, transposition via cache entries
# -----------------------------
class SearchContext:
    def __init__(self, board, player, zobrist_table, cache_entries, time_limit):
        self.board = board
        self.player = player
        self.zobrist = zobrist_table
        self.cache = cache_entries  # dict mapping key->entry
        self.start = time.time()
        self.time_limit = time_limit
        self.node_count = 0

    def time_exceeded(self):
        return (time.time() - self.start) >= self.time_limit

def negamax(ctx: SearchContext, depth: int, alpha: int, beta: int, color: int, zobrist_hash: int) -> int:
    """
    color = +1 if side to move is original player, -1 otherwise
    returns score from original player's perspective
    """
    ctx.node_count += 1
    # timeout check
    if ctx.time_exceeded():
        raise TimeoutError()

    # lookup cache (transposition)
    key = f"{zobrist_hash}:{'X' if ctx.player == 'X' else 'O'}"
    if key in ctx.cache:
        e = ctx.cache[key]
        # if entry depth >= current depth we can use its score (approx)
        if e.get("depth", 0) >= depth:
            return e.get("score", 0)

    # terminal
    if is_winner(ctx.board, ctx.player):
        return color * PATTERN_SCORES["five"]
    if is_winner(ctx.board, opponent(ctx.player)):
        return -color * PATTERN_SCORES["five"]
    if depth == 0:
        # evaluate from original player's perspective:
        val = evaluate_position(ctx.board, ctx.player)
        return color * val

    best = -10**12
    candidates = get_candidates(ctx.board, radius=2)
    # move ordering heuristic: try cached move for this state first if present
    # get cached move
    cached_move = None
    if key in ctx.cache:
        cached_move = tuple(ctx.cache[key]["move"])

    # sort candidates to try cached move first, then center closeness
    def cand_key(m):
        if cached_move and m == cached_move:
            return -99999
        # prefer cells closer to center
        return (abs(m[0] - SIZE//2) + abs(m[1] - SIZE//2))
    candidates.sort(key=cand_key)

    for (r, c) in candidates:
        if ctx.board[r][c] != " ":
            continue
        # make move for side-to-move (ctx.player) if color==+1 else opponent? We're using a simpler model:
        # We need to alternate the piece placed. We'll simulate using 'side' variable derived from color sign.
        # To keep negamax simple, we'll place ctx.player when color==1, opponent(ctx.player) when color==-1
        piece = ctx.player if color == 1 else opponent(ctx.player)
        ctx.board[r][c] = piece
        # update zobrist hash
        idx = r*SIZE + c
        added = ctx.zobrist[piece][idx]
        new_hash = zobrist_hash ^ added
        try:
            val = -negamax(ctx, depth-1, -beta, -alpha, -color, new_hash)
        except TimeoutError:
            ctx.board[r][c] = " "
            raise
        ctx.board[r][c] = " "
        if val > best:
            best = val
        if val > alpha:
            alpha = val
        if alpha >= beta:
            break
    return best

# -----------------------------
# Top-level choose move with iterative deepening
# -----------------------------
def choose_move_with_search(board: List[List[str]],
                            current_player: str,
                            zobrist_table: Dict,
                            cache_entries: Dict,
                            time_limit: float = 0.9,
                            max_depth: int = 4) -> Tuple[int,int,int]:
    """
    returns (best_r, best_c, best_score)
    Iterative deepening from depth=1..max_depth until time_limit.
    Uses cache_entries to read/write persistent transposition info.
    """
    ctx = SearchContext(board, current_player, zobrist_table, cache_entries, time_limit)
    base_hash = board_zobrist_hash(board, zobrist_table)
    # We'll store best found move and score
    best_move = None
    best_score = -10**12
    # iterative deepening
    for depth in range(1, max_depth+1):
        if ctx.time_exceeded():
            break
        try:
            # try to find best move at this depth
            candidates = get_candidates(board, radius=2)
            # move ordering: prefer cache
            key = f"{base_hash}:{current_player}"
            cached_move = None
            if key in cache_entries:
                cached_move = tuple(cache_entries[key]["move"])
                if cached_move in candidates:
                    # try cached first by moving it to front
                    candidates.remove(cached_move)
                    candidates.insert(0, cached_move)
            # search each candidate with negamax
            local_best = None
            local_best_score = -10**12
            alpha_init = -10**12
            beta_init = 10**12
            for (r, c) in candidates:
                if ctx.time_exceeded():
                    raise TimeoutError()
                if board[r][c] != " ":
                    continue
                # make move (current player)
                board[r][c] = current_player
                idx = r*SIZE + c
                new_hash = base_hash ^ zobrist_table[current_player][idx]
                try:
                    val = -negamax(ctx, depth-1, -beta_init, -alpha_init, -1, new_hash)
                except TimeoutError:
                    board[r][c] = " "
                    raise
                board[r][c] = " "
                if val > local_best_score:
                    local_best_score = val
                    local_best = (r, c)
                    alpha_init = max(alpha_init, val)
            # if we found a local best at this depth, update global best
            if local_best is not None:
                best_move = local_best
                best_score = local_best_score
                # save in cache (refine): store score and depth
                key = f"{base_hash}:{current_player}"
                cache_entries[key] = {
                    "move": list(best_move),
                    "score": int(best_score),
                    "depth": depth
                }
        except TimeoutError:
            # time up in deeper search: stop iterative deepening
            break
    # fallback: if still no best_move (rare), pick center or random candidate
    if best_move is None:
        cands = get_candidates(board, radius=2)
        best_move = cands[0] if cands else (SIZE//2, SIZE//2)
        best_score = cache_entries.get(f"{base_hash}:{current_player}", {}).get("score", 0)
    return best_move[0], best_move[1], int(best_score)

# -----------------------------
# Public API: get_move
# -----------------------------
def get_move(board: List[List[str]],
             current_player: str,
             time_limit: float = 0.9,
             max_depth: int = 4) -> Tuple[int,int]:
    """
    Main function you call.
    board: 15x15 list of 'X'/'O'/' '.
    current_player: 'X' or 'O'
    time_limit: seconds allowed for this move (inclusive)
    max_depth: maximum search depth for iterative deepening
    """
    # load cache structure
    cache_struct = load_cache_file()
    seed = cache_struct.get("zobrist_seed", None)
    # Keep zobrist stable across runs by storing the seed; if none, create new
    if seed is None:
        seed = random.randrange(1 << 30)
        cache_struct["zobrist_seed"] = seed
        cache_struct["entries"] = cache_struct.get("entries", {})
        save_cache_file(cache_struct)
    zobrist_table = init_zobrist(seed)
    cache_entries = cache_struct.get("entries", {})

    # quick check: immediate win or immediate block (cheap)
    cands = get_candidates(board, radius=2)
    for (r, c) in cands:
        # try winning move
        if board[r][c] != " ":
            continue
        board[r][c] = current_player
        if is_winner(board, current_player):
            board[r][c] = " "
            # store to cache as depth 1
            h = board_zobrist_hash(board, zobrist_table)
            key = f"{h}:{current_player}"
            cache_entries[key] = {"move":[r,c],"score":PATTERN_SCORES["five"], "depth":1}
            cache_struct["entries"] = cache_entries
            save_cache_file(cache_struct)
            return r, c
        board[r][c] = " "
    # block opponent immediate win
    opp = opponent(current_player)
    for (r, c) in cands:
        if board[r][c] != " ":
            continue
        board[r][c] = opp
        if is_winner(board, opp):
            board[r][c] = " "
            # record blocking move
            h = board_zobrist_hash(board, zobrist_table)
            key = f"{h}:{current_player}"
            cache_entries[key] = {"move":[r,c],"score":PATTERN_SCORES["five"], "depth":1}
            cache_struct["entries"] = cache_entries
            save_cache_file(cache_struct)
            return r, c
        board[r][c] = " "

    # run iterative deepening negamax with transposition cache
    try:
        r, c, score = choose_move_with_search(board, current_player, zobrist_table, cache_entries, time_limit, max_depth)
    except Exception:
        # fallback safe pick if something unexpected
        cands2 = get_candidates(board, radius=2)
        if not cands2:
            return SIZE//2, SIZE//2
        return cands2[0]

    # persist cache
    cache_struct["entries"] = cache_entries
    save_cache_file(cache_struct)
    return r, c

# -----------------------------
# Simple self-play trainer (synchronous)
# -----------------------------
def train_self_play(n_games: int = 50, time_limit_fast: float = 0.1, time_limit_slow: float = 0.6, max_depth_fast: int = 2, max_depth_slow: int = 4):
    """
    Quick training schedule:
     - first run many fast games with short time_limit (collect many states)
     - then run fewer slower games to refine
    """
    print("Training start (fast phase)...")
    for g in range(n_games):
        board = empty_board()
        player = "X"
        moves = 0
        while moves < SIZE*SIZE:
            r, c = get_move(board, player, time_limit=time
