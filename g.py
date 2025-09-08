#!/usr/bin/env python3
"""
Gomoku engine (full):
 - Zobrist hashing saved to file (pickle) for stable fast init
 - Persistent JSON cache entries (move, score, depth)
 - Iterative deepening + negamax (alpha-beta) with transposition cache
 - Candidate generation (neighborhood)
 - Pattern-based evaluation improved with double-threat detection
 - Multiprocessing self-play trainer (safe cache writes via Lock)
 - Configurable pattern weights (tune hyperparams)
"""

import os
import json
import time
import random
import pickle
import multiprocessing as mp
from typing import List, Tuple, Dict, Optional

# -----------------------------
# Config / hyperparams
# -----------------------------
SIZE = 15
WIN_LEN = 5

CACHE_FILE = "gomoku_cache_v3.json"
ZOBRIST_FILE = "zobrist_table.pkl"
WEIGHTS_FILE = "pattern_weights.json"

# Multiprocessing safe temp file name
CACHE_TMP = CACHE_FILE + ".tmp"

# Default pattern weights (you can tune or save to WEIGHTS_FILE)
PATTERN_SCORES = {
    "five": 1_000_000,
    "open_four": 120_000,
    "closed_four": 12_000,
    "open_three": 2_000,
    "closed_three": 200,
    "open_two": 80,
    "closed_two": 20,
    # weights for double threats detection
    "double_threat": 250_000,   # creating two immediate winning moves
    "opponent_double_threat_block": 200_000,  # blocking opponent double threat
    # penalty when move creates an immediate dangerous trap for us (allowing opponent double)
    "create_vulnerability_penalty": -150_000,
}

# training defaults
DEFAULT_FAST_TIME = 0.08
DEFAULT_SLOW_TIME = 0.4
DEFAULT_FAST_DEPTH = 1
DEFAULT_SLOW_DEPTH = 3

# -----------------------------
# Utility functions
# -----------------------------
def opponent(p: str) -> str:
    return "O" if p == "X" else "X"

def ensure_dirs():
    # nothing for now, placeholder if you want to create subfolders
    pass

# -----------------------------
# Zobrist hashing: persistent
# -----------------------------
def init_zobrist_from_seed(seed: int):
    rnd = random.Random(seed)
    table = {
        "X": [rnd.getrandbits(64) for _ in range(SIZE*SIZE)],
        "O": [rnd.getrandbits(64) for _ in range(SIZE*SIZE)],
        "side": rnd.getrandbits(64)
    }
    return table

def load_or_create_zobrist():
    if os.path.exists(ZOBRIST_FILE):
        try:
            with open(ZOBRIST_FILE, "rb") as f:
                data = pickle.load(f)
                # data should be dict {"seed": seed, "table": table}
                if isinstance(data, dict) and "table" in data and "seed" in data:
                    return data["seed"], data["table"]
        except Exception:
            pass
    # create new
    seed = random.randrange(1 << 30)
    table = init_zobrist_from_seed(seed)
    # save
    try:
        with open(ZOBRIST_FILE, "wb") as f:
            pickle.dump({"seed": seed, "table": table}, f)
    except Exception:
        pass
    return seed, table

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
# Cache file utilities (JSON)
# -----------------------------
def load_cache_file_struct():
    """Return dict with 'entries' mapping key->entry. If missing file, create structure."""
    if not os.path.exists(CACHE_FILE):
        return {"entries": {}}
    try:
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"entries": {}}

def save_cache_file_struct_atomic(struct: dict, lock: Optional[mp.Lock] = None):
    """Write cache struct atomically. If lock provided, use it (for multiprocessing)."""
    if lock is not None:
        lock.acquire()
    try:
        tmp = CACHE_TMP
        with open(tmp, "w") as f:
            json.dump(struct, f)
        os.replace(tmp, CACHE_FILE)
    finally:
        if lock is not None:
            lock.release()

# -----------------------------
# Board helpers
# -----------------------------
def empty_board():
    return [[" " for _ in range(SIZE)] for __ in range(SIZE)]

def in_bounds(r, c):
    return 0 <= r < SIZE and 0 <= c < SIZE

def is_winner(board: List[List[str]], player: str) -> bool:
    for r in range(SIZE):
        for c in range(SIZE):
            if board[r][c] != player:
                continue
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
    # convert to list, sort by closeness to center as simple heuristic
    lst = list(cells)
    lst.sort(key=lambda m: (abs(m[0] - SIZE//2) + abs(m[1] - SIZE//2)))
    return lst

# -----------------------------
# Pattern-based evaluation + double-threat detection
# -----------------------------
def evaluate_position(board: List[List[str]], player: str, weights: Dict = PATTERN_SCORES) -> int:
    """
    Evaluate board from 'player' perspective.
    Additionally, detect double threats: count immediate winning moves for each side.
    The evaluation combines pattern scanning and adjusts for double threats.
    """
    opp = opponent(player)
    total = 0

    # Build lines (rows, cols, two diagonals)
    lines = []
    for r in range(SIZE):
        lines.append("".join(board[r]))
    for c in range(SIZE):
        lines.append("".join(board[r][c] for r in range(SIZE)))
    # diag down-right
    for d in range(-SIZE+1, SIZE):
        s = []
        for r in range(SIZE):
            c = r - d
            if 0 <= c < SIZE:
                s.append(board[r][c])
        if s:
            lines.append("".join(s))
    # diag up-right
    for d in range(0, 2*SIZE-1):
        s = []
        for r in range(SIZE):
            c = d - r
            if 0 <= c < SIZE:
                s.append(board[r][c])
        if s:
            lines.append("".join(s))

    # pattern scoring
    for line in lines:
        padded = " " + line + " "
        # player patterns
        if player*5 in padded:
            total += weights.get("five", 1_000_000)
        if (" " + player*4 + " ") in padded:
            total += weights.get("open_four", 100_000)
        if (player*4 + " ") in padded or (" " + player*4) in padded:
            total += weights.get("closed_four", 10_000)
        if (" " + player*3 + " ") in padded:
            total += weights.get("open_three", 1_000)
        if (player*3 + " ") in padded or (" " + player*3) in padded:
            total += weights.get("closed_three", 100)
        if (" " + player*2 + " ") in padded:
            total += weights.get("open_two", 50)
        if (player*2 + " ") in padded or (" " + player*2) in padded:
            total += weights.get("closed_two", 10)

        # opponent patterns weighted for defense
        if opp*5 in padded:
            total += weights.get("five", 1_000_000)
        if (" " + opp*4 + " ") in padded:
            total += int(weights.get("open_four", 100_000) * 0.9)
        if (opp*4 + " ") in padded or (" " + opp*4) in padded:
            total += int(weights.get("closed_four", 10_000) * 0.7)
        if (" " + opp*3 + " ") in padded:
            total += int(weights.get("open_three", 1000) * 0.8)
        if (opp*3 + " ") in padded or (" " + opp*3) in padded:
            total += int(weights.get("closed_three", 100) * 0.5)

    # Detect double threats: immediate winning moves count for each side
    # count positions where placing that piece yields immediate win next move
    player_next_wins = count_next_winning_moves(board, player)
    opp_next_wins = count_next_winning_moves(board, opp)

    # if player has >=2 immediate winning moves => huge positive
    if player_next_wins >= 2:
        total += weights.get("double_threat", 250_000)
    elif player_next_wins == 1:
        total += int(weights.get("double_threat", 250_000) * 0.6)

    # if opponent has >=2 immediate winning moves => huge negative (we need to block)
    if opp_next_wins >= 2:
        total -= weights.get("double_threat", 250_000)
    elif opp_next_wins == 1:
        total -= int(weights.get("double_threat", 250_000) * 0.6)

    return int(total)

def count_next_winning_moves(board: List[List[str]], player: str) -> int:
    """Return number of moves that if player plays there, they immediately win."""
    cnt = 0
    for r in range(SIZE):
        for c in range(SIZE):
            if board[r][c] != " ":
                continue
            board[r][c] = player
            if is_winner(board, player):
                cnt += 1
            board[r][c] = " "
    return cnt

# -----------------------------
# Search: negamax with alpha-beta, transposition via cache
# -----------------------------
class SearchContext:
    def __init__(self, board, root_player, zobrist_table, cache_entries, time_limit, weights):
        self.board = board
        self.root_player = root_player  # original player for evaluation perspective
        self.zobrist = zobrist_table
        self.cache = cache_entries
        self.start = time.time()
        self.time_limit = time_limit
        self.node_count = 0
        self.weights = weights

    def time_exceeded(self) -> bool:
        return (time.time() - self.start) >= self.time_limit

def negamax(ctx: SearchContext, depth: int, alpha: int, beta: int, color: int, zobrist_hash: int, side_to_move: str) -> int:
    """
    color: +1 if side_to_move == ctx.root_player, -1 otherwise
    side_to_move: "X" or "O"
    Returns score from root_player perspective.
    """
    ctx.node_count += 1
    if ctx.time_exceeded():
        raise TimeoutError()

    key = f"{zobrist_hash}:{side_to_move}"
    # transposition: if entry depth >= depth, use stored score
    if key in ctx.cache:
        e = ctx.cache[key]
        if e.get("depth", 0) >= depth:
            return e.get("score", 0)

    # terminal checks
    if is_winner(ctx.board, ctx.root_player):
        return color * ctx.weights.get("five", PATTERN_SCORES["five"])
    if is_winner(ctx.board, opponent(ctx.root_player)):
        return -color * ctx.weights.get("five", PATTERN_SCORES["five"])
    if depth == 0:
        val = evaluate_position(ctx.board, ctx.root_player, ctx.weights)
        # color adjustment: if color == 1, result as is; else flip sign because perspective
        return val if color == 1 else -val

    best = -10**12
    candidates = get_candidates(ctx.board, radius=2)
    # ordering: try cached move first if exists
    cached_move = None
    if key in ctx.cache:
        cached_move = tuple(ctx.cache[key]["move"])

    def ord_key(m):
        if cached_move and m == cached_move:
            return -99999
        return (abs(m[0] - SIZE//2) + abs(m[1] - SIZE//2))
    candidates.sort(key=ord_key)

    for (r, c) in candidates:
        if ctx.board[r][c] != " ":
            continue
        # place side_to_move stone
        ctx.board[r][c] = side_to_move
        idx = r*SIZE + c
        new_hash = zobrist_hash ^ ctx.zobrist[side_to_move][idx]
        try:
            val = -negamax(ctx, depth-1, -beta, -alpha, -color, new_hash, opponent(side_to_move))
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
# Iterative deepening wrapper
# -----------------------------
def choose_move_with_search(board: List[List[str]],
                            current_player: str,
                            zobrist_table: Dict,
                            cache_entries: Dict,
                            time_limit: float = 0.9,
                            max_depth: int = 4,
                            weights: Dict = PATTERN_SCORES) -> Tuple[int,int,int]:
    ctx = SearchContext(board, current_player, zobrist_table, cache_entries, time_limit, weights)
    base_hash = board_zobrist_hash(board, zobrist_table)
    best_move = None
    best_score = -10**12

    # iterative deepening
    for depth in range(1, max_depth+1):
        if ctx.time_exceeded():
            break
        try:
            candidates = get_candidates(board, radius=2)
            key_root = f"{base_hash}:{current_player}"
            if key_root in cache_entries:
                cached_move = tuple(cache_entries[key_root]["move"])
                if cached_move in candidates:
                    candidates.remove(cached_move)
                    candidates.insert(0, cached_move)

            local_best = None
            local_best_score = -10**12
            alpha_init = -10**12
            beta_init = 10**12

            for (r, c) in candidates:
                if ctx.time_exceeded():
                    raise TimeoutError()
                if board[r][c] != " ":
                    continue
                # make move
                board[r][c] = current_player
                idx = r*SIZE + c
                new_hash = base_hash ^ zobrist_table[current_player][idx]
                try:
                    val = -negamax(ctx, depth-1, -beta_init, -alpha_init, -1, new_hash, opponent(current_player))
                except TimeoutError:
                    board[r][c] = " "
                    raise
                board[r][c] = " "
                # penalize moves that create immediate vulnerability: if after our move opponent would have >=2 immediate wins
                board[r][c] = current_player
                opp_double_after = count_next_winning_moves(board, opponent(current_player))
                board[r][c] = " "
                if opp_double_after >= 2:
                    val += weights.get("create_vulnerability_penalty", -150_000)
                if val > local_best_score:
                    local_best_score = val
                    local_best = (r, c)
                    alpha_init = max(alpha_init, val)

            if local_best is not None:
                best_move = local_best
                best_score = local_best_score
                # update cache with refined info
                cache_entries[f"{base_hash}:{current_player}"] = {
                    "move": [best_move[0], best_move[1]],
                    "score": int(best_score),
                    "depth": depth
                }
        except TimeoutError:
            break

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
             max_depth: int = 4,
             weights: Optional[Dict] = None) -> Tuple[int,int]:
    """
    Main function to call.
    board: 15x15 list of 'X'/'O'/' '
    current_player: 'X' or 'O'
    """
    ensure_dirs()
    if weights is None:
        weights = PATTERN_SCORES

    # load or create zobrist table
    seed, zobrist_table = load_or_create_zobrist()

    # load cache entries
    struct = load_cache_file_struct()
    cache_entries = struct.get("entries", {})

    # quick immediate win/block cheap scan
    cands = get_candidates(board, radius=2)
    # immediate win
    for (r, c) in cands:
        if board[r][c] != " ":
            continue
        board[r][c] = current_player
        if is_winner(board, current_player):
            board[r][c] = " "
            h = board_zobrist_hash(board, zobrist_table)
            cache_entries[f"{h}:{current_player}"] = {"move":[r,c], "score":weights.get("five", 1_000_000), "depth":1}
            struct["entries"] = cache_entries
            save_cache_file_struct_atomic(struct)
            return r, c
        board[r][c] = " "
    # immediate block
    opp = opponent(current_player)
    for (r, c) in cands:
        if board[r][c] != " ":
            continue
        board[r][c] = opp
        if is_winner(board, opp):
            board[r][c] = " "
            h = board_zobrist_hash(board, zobrist_table)
            cache_entries[f"{h}:{current_player}"] = {"move":[r,c], "score":weights.get("five", 1_000_000), "depth":1}
            struct["entries"] = cache_entries
            save_cache_file_struct_atomic(struct)
            return r, c
        board[r][c] = " "

    # run search
    try:
        r, c, score = choose_move_with_search(board, current_player, zobrist_table, cache_entries, time_limit, max_depth, weights)
    except Exception:
        # fallback
        cs = get_candidates(board, radius=2)
        if not cs:
            return SIZE//2, SIZE//2
        return cs[0]

    # save cache
    struct["entries"] = cache_entries
    save_cache_file_struct_atomic(struct)
    return r, c

# -----------------------------
# Multiprocessing training
# -----------------------------
def _selfplay_worker(worker_id: int, n_games: int, fast_time: float, slow_time: float,
                     fast_depth: int, slow_depth: int, zobrist_seed: int,
                     shared_cache: dict, lock: mp.Lock, weights: Dict):
    """
    Worker that runs n_games self-play. Uses shared_cache (Manager.dict).
    Writes to file occasionally using lock when finishing.
    """
    # init zobrist table from seed
    zob = init_zobrist_from_seed(zobrist_seed)
    random.seed(worker_id + zobrist_seed)
    for g in range(n_games):
        board = empty_board()
        player = "X"
        moves = 0
        # alternate fast/slow policy probabilistically (mostly fast)
        game_time_limit = fast_time
        game_depth = fast_depth
        if random.random() < 0.15:
            game_time_limit = slow_time
            game_depth = slow_depth
        while moves < SIZE*SIZE:
            # choose move using local view of shared cache (it's a manager dict)
            # convert manager dict to normal dict for search function (lighter)
            local_cache = dict(shared_cache)
            try:
                r, c = choose_move_with_search(board, player, zob, local_cache, game_time_limit, game_depth, weights)
            except Exception:
                # fallback quick pick
                cs = get_candidates(board, radius=2)
                if not cs:
                    break
                r, c = cs[0]
            # apply move
            if board[r][c] != " ":
                break
            board[r][c] = player
            # update shared_cache with new info from local_cache diffs
            # For simplicity: we will push root entry for the position we saw
            base_hash = board_zobrist_hash(board, zob)
            key_root = f"{base_hash}:{player}"
            shared_cache[key_root] = local_cache.get(key_root, {"move":[r,c],"score":0,"depth":game_depth})
            # switch
            if is_winner(board, player):
                break
            player = opponent(player)
            moves += 1
        # occasionally flush to disk (synchronized)
        if (g + 1) % 10 == 0:
            # acquire lock and dump shared_cache to file
            if lock is not None:
                lock.acquire()
            try:
                struct = {"entries": dict(shared_cache)}
                # keep zobrist seed in file struct for consistency
                struct["zobrist_seed"] = zobrist_seed
                with open(CACHE_TMP, "w") as f:
                    json.dump(struct, f)
                os.replace(CACHE_TMP, CACHE_FILE)
            finally:
                if lock is not None:
                    lock.release()

def train_multiprocess(n_games: int = 200, n_processes: int = 4,
                       fast_time: float = DEFAULT_FAST_TIME, slow_time: float = DEFAULT_SLOW_TIME,
                       fast_depth: int = DEFAULT_FAST_DEPTH, slow_depth: int = DEFAULT_SLOW_DEPTH,
                       weights: Dict = PATTERN_SCORES):
    """
    Multiprocess self-play trainer.
    Splits n_games across processes; each process updates a Manager.dict() shared cache.
    After finishing, main process writes cache to disk.
    """
    seed, zobrist_table = load_or_create_zobrist()
    manager = mp.Manager()
    shared_cache = manager.dict(load_cache_file_struct().get("entries", {}))
    lock = manager.Lock()

    games_per_worker = max(1, n_games // n_processes)
    procs = []
    for i in range(n_processes):
        gw = games_per_worker if i < n_processes-1 else (n_games - games_per_worker*(n_processes-1))
        p = mp.Process(target=_selfplay_worker, args=(i, gw, fast_time, slow_time, fast_depth, slow_depth, seed, shared_cache, lock, weights))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    # final save (main process)
    struct = {"zobrist_seed": seed, "entries": dict(shared_cache)}
    save_cache_file_struct_atomic(struct)
    print("Multiprocess training complete. Cache saved to", CACHE_FILE)

# -----------------------------
# Utilities to load/save weights
# -----------------------------
def save_weights_file(weights: Dict):
    try:
        with open(WEIGHTS_FILE, "w") as f:
            json.dump(weights, f)
    except Exception:
        pass

def load_weights_file() -> Dict:
    if os.path.exists(WEIGHTS_FILE):
        try:
            with open(WEIGHTS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return PATTERN_SCORES.copy()

# -----------------------------
# Demo / Example usage
# -----------------------------
if __name__ == "__main__":
    # Example sequence:
    # 1) Train with multiprocessing (be careful with CPU/time)
    print("Loading pattern weights...")
    w = load_weights_file()
    print("Starting multiprocessing training (example: 80 games across 4 processes)...")
    train_multiprocess(n_games=80, n_processes=4,
                       fast_time=DEFAULT_FAST_TIME, slow_time=DEFAULT_SLOW_TIME,
                       fast_depth=DEFAULT_FAST_DEPTH, slow_depth=DEFAULT_SLOW_DEPTH,
                       weights=w)

    # 2) Test get_move on empty board with reasonable time allowed
    b = empty_board()
    mv = get_move(b, "X", time_limit=0.6, max_depth=3, weights=w)
    print("Selected move on empty board:", mv)
