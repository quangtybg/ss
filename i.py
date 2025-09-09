#!/usr/bin/env python3
"""
improved_gomoku_bot.py

Gomoku (15x15) bot with:
 - Persistent Zobrist hashing (pickle) for stable hashing
 - Structured cache (JSON) storing entries: { key: {move, score, depth} }
 - Iterative deepening + negamax with alpha-beta
 - Move ordering: PV (cache), killer moves, history heuristic, quick eval ordering
 - Pattern-based evaluation + immediate win/block detection + double-threat detection
 - Candidate generation (neighborhood + heuristics)
 - Multiprocessing self-play trainer to populate cache
 - Public API: get_move(board, current_player, time_limit=0.9, max_depth=4)
"""
import os
import time
import json
import random
import pickle
import math
import multiprocessing as mp
from typing import List, Tuple, Dict, Optional

# ---------------- CONFIG ----------------
SIZE = 15
WIN_LEN = 5
CACHE_FILE = "gomoku_cache_v4.json"
ZOBRIST_FILE = "zobrist_v4.pkl"
PATTERN_WEIGHTS_FILE = "pattern_weights_v4.json"

# default pattern scores (tuneable)
PATTERN_SCORES = {
    "five": 1_000_000,
    "open_four": 120_000,
    "closed_four": 12_000,
    "open_three": 4_000,
    "closed_three": 400,
    "open_two": 80,
    "closed_two": 20,
    "double_threat": 300_000,
    "block_double_threat": 250_000,
    "create_vulnerability_penalty": -200_000,
}

# training defaults
DEFAULT_FAST_TIME = 0.08
DEFAULT_SLOW_TIME = 0.4
DEFAULT_FAST_DEPTH = 1
DEFAULT_SLOW_DEPTH = 3

# ---------------- Utilities ----------------
def opponent(p: str) -> str:
    return "O" if p == "X" else "X"

def empty_board() -> List[List[str]]:
    return [[" " for _ in range(SIZE)] for __ in range(SIZE)]

def in_bounds(r: int, c: int) -> bool:
    return 0 <= r < SIZE and 0 <= c < SIZE

# ---------------- Zobrist ----------------
def init_zobrist(seed: Optional[int] = None):
    rnd = random.Random(seed)
    table = {
        "X": [rnd.getrandbits(64) for _ in range(SIZE*SIZE)],
        "O": [rnd.getrandbits(64) for _ in range(SIZE*SIZE)],
        "side": rnd.getrandbits(64)
    }
    return table

def load_or_create_zobrist() -> Tuple[int, dict]:
    if os.path.exists(ZOBRIST_FILE):
        try:
            with open(ZOBRIST_FILE, "rb") as f:
                obj = pickle.load(f)
                if isinstance(obj, dict) and "seed" in obj and "table" in obj:
                    return obj["seed"], obj["table"]
        except Exception:
            pass
    seed = random.randrange(1 << 30)
    table = init_zobrist(seed)
    try:
        with open(ZOBRIST_FILE, "wb") as f:
            pickle.dump({"seed": seed, "table": table}, f)
    except Exception:
        pass
    return seed, table

def board_zobrist_hash(board: List[List[str]], zobrist_table: Dict, side_to_move: Optional[str] = None) -> int:
    h = 0
    for r in range(SIZE):
        for c in range(SIZE):
            v = board[r][c]
            if v == " ":
                continue
            idx = r*SIZE + c
            h ^= zobrist_table[v][idx]
    if side_to_move is not None:
        # include side info for PV differentiation
        if side_to_move == "X":
            h ^= zobrist_table["side"]
    return h

# ---------------- Cache file (atomic) ----------------
def load_cache_struct() -> Dict:
    if not os.path.exists(CACHE_FILE):
        return {"zobrist_seed": None, "entries": {}}
    try:
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"zobrist_seed": None, "entries": {}}

def save_cache_struct_atomic(struct: Dict):
    tmp = CACHE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(struct, f)
    os.replace(tmp, CACHE_FILE)

# ---------------- Board utilities ----------------
def is_winner(board: List[List[str]], player: str) -> bool:
    # check horizontally, vertically, two diagonals
    for r in range(SIZE):
        for c in range(SIZE):
            if board[r][c] != player:
                continue
            # right
            if c + WIN_LEN <= SIZE and all(board[r][c+i] == player for i in range(WIN_LEN)):
                return True
            # down
            if r + WIN_LEN <= SIZE and all(board[r+i][c] == player for i in range(WIN_LEN)):
                return True
            # diag down-right
            if r + WIN_LEN <= SIZE and c + WIN_LEN <= SIZE and all(board[r+i][c+i] == player for i in range(WIN_LEN)):
                return True
            # diag up-right
            if r - WIN_LEN + 1 >= 0 and c + WIN_LEN <= SIZE and all(board[r-i][c+i] == player for i in range(WIN_LEN)):
                return True
    return False

# ---------------- Candidate generation ----------------
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
    # convert and optionally sort by adjacency score (cells near longer chains get higher priority)
    def adjacency_score(cell):
        r, c = cell
        score = 0
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r+dr, c+dc
                if in_bounds(nr, nc) and board[nr][nc] != " ":
                    score += 1
        # prefer closer to center as tie-break
        score -= (abs(r - SIZE//2) + abs(c - SIZE//2)) * 0.01
        return -score  # negative so smaller sort key => higher priority
    lst = list(cells)
    lst.sort(key=adjacency_score)
    return lst

# ---------------- Evaluation & threats ----------------
def evaluate_position(board: List[List[str]], player: str, weights: Dict = PATTERN_SCORES) -> int:
    """Pattern-based evaluation + account for double threats."""
    opp = opponent(player)
    total = 0

    lines = []

    # rows
    for r in range(SIZE):
        lines.append("".join(board[r]))
    # columns
    for c in range(SIZE):
        lines.append("".join(board[r][c] for r in range(SIZE)))
    # diag down-right
    for d in range(-SIZE+1, SIZE):
        arr = []
        for r in range(SIZE):
            c = r - d
            if 0 <= c < SIZE:
                arr.append(board[r][c])
        if len(arr) >= 1:
            lines.append("".join(arr))
    # diag up-right
    for d in range(0, 2*SIZE-1):
        arr = []
        for r in range(SIZE):
            c = d - r
            if 0 <= c < SIZE:
                arr.append(board[r][c])
        if len(arr) >= 1:
            lines.append("".join(arr))

    # pattern detection via simple substrings with padding
    for line in lines:
        padded = " " + line + " "
        # attack
        if player*5 in padded:
            total += weights["five"]
        if (" " + player*4 + " ") in padded:
            total += weights["open_four"]
        if (player*4 + " ") in padded or (" " + player*4) in padded:
            total += weights["closed_four"]
        if (" " + player*3 + " ") in padded:
            total += weights["open_three"]
        if (player*3 + " ") in padded or (" " + player*3) in padded:
            total += weights["closed_three"]
        # two
        if (" " + player*2 + " ") in padded:
            total += weights["open_two"]
        if (player*2 + " ") in padded or (" " + player*2) in padded:
            total += weights["closed_two"]

        # defense: opponent patterns reduce score (but use scaled weights)
        if opp*5 in padded:
            total -= weights["five"] * 1.1
        if (" " + opp*4 + " ") in padded:
            total -= int(weights["open_four"] * 0.95)
        if (opp*4 + " ") in padded or (" " + opp*4) in padded:
            total -= int(weights["closed_four"] * 0.8)
        if (" " + opp*3 + " ") in padded:
            total -= int(weights["open_three"] * 0.9)
        if (opp*3 + " ") in padded or (" " + opp*3) in padded:
            total -= int(weights["closed_three"] * 0.6)

    # detect double threats quickly: count immediate winning moves for both
    my_wins = count_immediate_wins(board, player)
    opp_wins = count_immediate_wins(board, opp)
    if my_wins >= 2:
        total += weights["double_threat"]
    elif my_wins == 1:
        total += int(weights["double_threat"] * 0.6)
    if opp_wins >= 2:
        total -= weights["double_threat"]
    elif opp_wins == 1:
        total -= int(weights["double_threat"] * 0.6)

    return int(total)

def count_immediate_wins(board: List[List[str]], player: str) -> int:
    """Count number of empty positions that if player plays there would be an immediate win."""
    cnt = 0
    for r in range(SIZE):
        for c in range(SIZE):
            if board[r][c] != " ":
                continue
            board[r][c] = player
            if is_winner(board, player):
                cnt += 1
            board[r][c] = " "
            # optimization: early break if >=2 (we only care about double threat)
            if cnt >= 2:
                return cnt
    return cnt

# ---------------- Search (negamax with enhancements) ----------------
class SearchContext:
    def __init__(self, board, root_player, zobrist_table, cache_entries, time_limit, weights):
        self.board = board
        self.root_player = root_player
        self.zobrist = zobrist_table
        self.cache = cache_entries  # dict
        self.start_time = time.time()
        self.time_limit = time_limit
        self.node_count = 0
        # heuristics state
        self.killer = {}  # depth -> set of killer moves
        self.history = {}  # move -> score
        self.weights = weights

    def time_exceeded(self) -> bool:
        return (time.time() - self.start_time) >= self.time_limit

def negamax(ctx: SearchContext, depth: int, alpha: int, beta: int, side: str, zobrist_hash: int) -> int:
    """
    Negamax returns score from ctx.root_player perspective.
    side: which side to place now ('X' or 'O')
    """
    ctx.node_count += 1
    if ctx.time_exceeded():
        raise TimeoutError()

    key = f"{zobrist_hash}:{side}"
    # transposition table usage
    if key in ctx.cache:
        entry = ctx.cache[key]
        if entry.get("depth", 0) >= depth:
            return entry.get("score", 0)

    # terminal checks
    if is_winner(ctx.board, ctx.root_player):
        return 10**9
    if is_winner(ctx.board, opponent(ctx.root_player)):
        return -10**9
    if depth == 0:
        val = evaluate_position(ctx.board, ctx.root_player, ctx.weights)
        return val

    best = -10**12
    # generate and order candidates
    cands = get_candidates(ctx.board, radius=2)
    # ordering: PV from cache, killer moves at this depth, history heuristic, quick eval
    pv = ctx.cache.get(f"{zobrist_hash}:{side}", {}).get("move")
    def cand_key(m):
        score = 0
        if pv and tuple(m) == tuple(pv):
            score += 1000000
        if m in ctx.killer.get(depth, set()):
            score += 500000
        score += ctx.history.get(tuple(m), 0)
        # lightweight: simulate placing and evaluate quickly for ordering
        r, c = m
        ctx.board[r][c] = side
        quick = evaluate_position(ctx.board, ctx.root_player, ctx.weights)
        ctx.board[r][c] = " "
        score += quick
        # prefer center tie-break
        score -= (abs(r - SIZE//2) + abs(c - SIZE//2)) * 0.001
        return -score  # negative to sort descending
    cands.sort(key=cand_key)

    for (r, c) in cands:
        if ctx.board[r][c] != " ":
            continue
        # play move
        ctx.board[r][c] = side
        idx = r*SIZE + c
        new_hash = zobrist_hash ^ ctx.zobrist[side][idx]
        try:
            val = -negamax(ctx, depth-1, -beta, -alpha, opponent(side), new_hash)
        except TimeoutError:
            ctx.board[r][c] = " "
            raise
        ctx.board[r][c] = " "
        if val > best:
            best = val
            best_move = (r, c)
        if val > alpha:
            alpha = val
            # update history heuristic
            ctx.history.setdefault((r,c), 0)
            ctx.history[(r,c)] += (1 << depth)
        if alpha >= beta:
            # record killer
            ctx.killer.setdefault(depth, set()).add((r,c))
            break

    # store in cache coarse info
    ctx.cache[key] = {
        "move": list(best_move) if best < 10**11 else list(best_move),
        "score": int(best),
        "depth": depth
    }
    return best

# ---------------- Iterative deepening wrapper ----------------
def choose_move_with_search(board: List[List[str]],
                            current_player: str,
                            zobrist_table: Dict,
                            cache_entries: Dict,
                            time_limit: float = 0.9,
                            max_depth: int = 4,
                            weights: Dict = PATTERN_SCORES) -> Tuple[int,int,int]:
    ctx = SearchContext(board, current_player, zobrist_table, cache_entries, time_limit, weights)
    base_hash = board_zobrist_hash(board, zobrist_table, side_to_move=current_player)
    best_move = None
    best_score = -10**12

    # quick heuristic moves set (immediate win/block/double threat detection)
    cands = get_candidates(board, radius=2)

    # immediate win
    for (r,c) in cands:
        if board[r][c] != " ":
            continue
        board[r][c] = current_player
        if is_winner(board, current_player):
            board[r][c] = " "
            # write to cache and return immediately
            key = f"{base_hash}:{current_player}"
            cache_entries[key] = {"move":[r,c], "score":weights["five"], "depth":1}
            return r, c, weights["five"]
        board[r][c] = " "

    # immediate block
    opp = opponent(current_player)
    for (r,c) in cands:
        if board[r][c] != " ":
            continue
        board[r][c] = opp
        if is_winner(board, opp):
            board[r][c] = " "
            key = f"{base_hash}:{current_player}"
            cache_entries[key] = {"move":[r,c], "score":weights["five"], "depth":1}
            return r, c, weights["five"]
        board[r][c] = " "

    # iterative deepening
    for depth in range(1, max_depth+1):
        if ctx.time_exceeded():
            break
        try:
            # use cached PV first
            key_root = f"{base_hash}:{current_player}"
            pv = cache_entries.get(key_root, {}).get("move")
            candidates = cands.copy()
            if pv and tuple(pv) in candidates:
                candidates.remove(tuple(pv))
                candidates.insert(0, tuple(pv))
            local_best = None
            local_best_score = -10**12
            alpha_init = -10**12
            beta_init = 10**12
            for (r,c) in candidates:
                if ctx.time_exceeded():
                    raise TimeoutError()
                if board[r][c] != " ":
                    continue
                # make move
                board[r][c] = current_player
                idx = r*SIZE + c
                new_hash = base_hash ^ zobrist_table[current_player][idx]
                try:
                    val = -negamax(ctx, depth-1, -beta_init, -alpha_init, opponent(current_player), new_hash)
                except TimeoutError:
                    board[r][c] = " "
                    raise
                board[r][c] = " "

                # penalize moves that immediately create opponent double threat
                board[r][c] = current_player
                opp_double_after = count_immediate_wins(board, opp)
                board[r][c] = " "
                if opp_double_after >= 2:
                    val += weights.get("create_vulnerability_penalty", -200_000)

                if val > local_best_score:
                    local_best_score = val
                    local_best = (r, c)
                    alpha_init = max(alpha_init, val)
            if local_best is not None:
                best_move = local_best
                best_score = local_best_score
                cache_entries[key_root] = {"move":[best_move[0], best_move[1]], "score":int(best_score), "depth":depth}
        except TimeoutError:
            break

    if best_move is None:
        fallback = get_candidates(board, radius=2)
        best_move = fallback[0] if fallback else (SIZE//2, SIZE//2)
        best_score = cache_entries.get(f"{base_hash}:{current_player}", {}).get("score", 0)

    return best_move[0], best_move[1], int(best_score)

# ---------------- Public API ----------------
def get_move(board: List[List[str]],
             current_player: str,
             time_limit: float = 0.9,
             max_depth: int = 4,
             weights: Optional[Dict] = None) -> Tuple[int,int]:
    """
    Choose a move for current_player.
    board: 15x15 list of 'X'/'O'/' '.
    Returns (r, c).
    """
    if weights is None:
        weights = PATTERN_SCORES

    # load cache (and zobrist seed) once
    cache_struct = load_cache_struct()
    seed = cache_struct.get("zobrist_seed")
    if seed is None:
        seed = random.randrange(1 << 30)
        cache_struct["zobrist_seed"] = seed
        cache_struct["entries"] = cache_struct.get("entries", {})
        save_cache_struct_atomic(cache_struct)
    _, zobrist_table = seed, init_zobrist(seed)  # use init_zobrist to maintain same generation function
    cache_entries = cache_struct.get("entries", {})

    # run search
    try:
        r, c, _ = choose_move_with_search(board, current_player, zobrist_table, cache_entries, time_limit, max_depth, weights)
    except Exception:
        # fallback
        cs = get_candidates(board, radius=2)
        if not cs:
            return SIZE//2, SIZE//2
        return cs[0]

    # persist cache
    cache_struct["entries"] = cache_entries
    save_cache_struct_atomic(cache_struct)
    return r, c

# ---------------- Multiprocess self-play trainer ----------------
def _selfplay_worker(worker_id: int, n_games: int, seed: int, shared_entries, lock: mp.Lock, weights: Dict):
    # initialize local zobrist
    zob = init_zobrist(seed)
    random.seed(worker_id + seed)
    for g in range(n_games):
        board = empty_board()
        player = "X"
        moves = 0
        # alternate speeds (some slow games)
        time_limit = DEFAULT_FAST_TIME if random.random() > 0.12 else DEFAULT_SLOW_TIME
        depth = DEFAULT_FAST_DEPTH if time_limit == DEFAULT_FAST_TIME else DEFAULT_SLOW_DEPTH

        while moves < SIZE*SIZE:
            # convert shared_entries (Manager.dict) to normal dict for local search read
            local_cache = dict(shared_entries)
            try:
                r, c, score = choose_move_with_search(board, player, zob, local_cache, time_limit, depth, weights)
            except Exception:
                # fallback random candidate
                cands = get_candidates(board, radius=2)
                if not cands:
                    break
                r, c = cands[0]
            if board[r][c] != " ":
                break
            board[r][c] = player
            # update shared cache root entry
            h = board_zobrist_hash(board, zob, side_to_move=player)
            key = f"{h}:{player}"
            # set move in shared_entries (atomic through manager)
            shared_entries[key] = local_cache.get(key, {"move":[r,c],"score":int(score if 'score' in locals() else 0),"depth":depth})
            if is_winner(board, player):
                break
            player = opponent(player)
            moves += 1
        # occasional flush handled by manager; we also occasionally write to disk (main process)
    # worker ends

def train_multiprocess(n_games: int = 200, n_processes: int = 4, weights: Optional[Dict] = None):
    if weights is None:
        weights = PATTERN_SCORES
    seed, _ = load_or_create_zobrist()
    manager = mp.Manager()
    struct = load_cache_struct()
    shared_entries = manager.dict(struct.get("entries", {}))
    lock = manager.Lock()

    per = max(1, n_games // n_processes)
    procs = []
    for i in range(n_processes):
        ng = per if i < n_processes-1 else (n_games - per*(n_processes-1))
        p = mp.Process(target=_selfplay_worker, args=(i, ng, seed, shared_entries, lock, weights))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    # final save
    struct["zobrist_seed"] = seed
    struct["entries"] = dict(shared_entries)
    save_cache_struct_atomic(struct)
    print("Training complete; cache saved to", CACHE_FILE)

def load_or_create_zobrist():
    """Load existing zobrist seed/table if exists, else create and save new."""
    if os.path.exists(ZOBRIST_FILE):
        try:
            with open(ZOBRIST_FILE, "rb") as f:
                obj = pickle.load(f)
                return obj["seed"], obj["table"]
        except Exception:
            pass
    seed = random.randrange(1 << 30)
    table = init_zobrist(seed)
    with open(ZOBRIST_FILE, "wb") as f:
        pickle.dump({"seed": seed, "table": table}, f)
    return seed, table

# ---------------- Quick demo ----------------
if __name__ == "__main__":
    # small demo to build some cache quickly
    print("Demo training (multiprocess small)...")
    train_multiprocess(n_games=60, n_processes=max(1, mp.cpu_count()//2))
    # test get_move on empty board
    b = empty_board()
    mv = get_move(b, "X", time_limit=0.6, max_depth=3)
    print("Selected move on empty board:", mv)
