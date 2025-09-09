#!/usr/bin/env python3
"""
Gomoku single-file bot (improved):
 - Zobrist hashing
 - Transposition Table (TT) persisted to JSON (partial search resume)
 - Iterative deepening + minimax w/ alpha-beta + TT bounds (EXACT/LOWER/UPPER)
 - Opening book (simple)
 - Pattern-based evaluation (attack + defense + double threats)
 - Move generation radius-limited
 - Move ordering with defensive priority
 - get_move(board, player, time_limit=0.9) -> (row, col)
"""

import time
import random
import json
import os
from typing import List, Tuple, Optional, Dict

# ---------------- CONFIG ----------------
BOARD_SIZE = 15
TT_FILE = "gomoku_tt.json"   # persistent transposition table
ZOBRIST_SEED = 123456
DEFAULT_RADIUS = 2
# patterns weights (tunable)
WEIGHTS = {
    "FIVE": 10**9,
    "FOUR_OPEN": 10**6,
    "FOUR_SEMI": 50000,
    "THREE_OPEN": 5000,
    "THREE_SEMI": 800,
    "TWO_OPEN": 200,
    "TWO_SEMI": 50,
    "ONE": 5,
    "DOUBLE_THREAT": 10**7
}

random.seed(ZOBRIST_SEED)

# ---------------- ZOBRIST ----------------
ZOBRIST = [[[random.getrandbits(64) for _ in range(3)] for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
# mapping: ' ' -> 0, 'X' -> 1, 'O' -> 2
SYM_MAP = {' ': 0, 'X': 1, 'O': 2}

def zobrist_hash(board: List[List[str]]) -> str:
    h = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            v = SYM_MAP.get(board[r][c], 0)
            if v:
                h ^= ZOBRIST[r][c][v]
    return str(h)

# ---------------- TT (Transposition Table) ----------------
# TT entry: { "move": [r,c] or None, "score": int, "depth": int, "flag": "EXACT"/"LOWER"/"UPPER" }
TT: Dict[str, Dict] = {}

def save_tt(path: str = TT_FILE):
    try:
        # need to convert tuple moves to list if any
        dumpable = {}
        for k, v in TT.items():
            entry = dict(v)
            if entry.get("move") is not None:
                entry["move"] = list(entry["move"])
            dumpable[k] = entry
        with open(path, "w") as f:
            json.dump(dumpable, f)
    except Exception as e:
        print("save_tt error:", e)

def load_tt(path: str = TT_FILE):
    global TT
    if not os.path.exists(path):
        TT = {}
        return
    try:
        with open(path, "r") as f:
            raw = json.load(f)
        TT = {}
        for k,v in raw.items():
            entry = dict(v)
            if entry.get("move") is not None:
                entry["move"] = tuple(entry["move"])
            TT[k] = entry
    except Exception as e:
        print("load_tt error:", e)
        TT = {}

# ---------------- Helpers ----------------
def switch_player(p: str) -> str:
    return 'O' if p == 'X' else 'X'

def board_to_key(board: List[List[str]], player: str) -> str:
    # prefer zobrist for TT key; for other cache we may use string
    return zobrist_hash(board) + "_" + player

def empty_board(board: List[List[str]]) -> bool:
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != ' ':
                return False
    return True

# ---------------- Move Generation ----------------
def generate_moves(board: List[List[str]], radius: int = DEFAULT_RADIUS) -> List[Tuple[int,int]]:
    moves = set()
    has_piece = False
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != ' ':
                has_piece = True
                for dr in range(-radius, radius+1):
                    for dc in range(-radius, radius+1):
                        rr, cc = r+dr, c+dc
                        if 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and board[rr][cc] == ' ':
                            moves.add((rr,cc))
    if not has_piece:
        return [(BOARD_SIZE//2, BOARD_SIZE//2)]
    return list(moves)

# ---------------- Winning check ----------------
def check_win_board(board: List[List[str]], player: Optional[str]=None) -> bool:
    # if player provided, check that player's win; else check any 5 in row
    directions = [(1,0),(0,1),(1,1),(1,-1)]
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] == ' ':
                continue
            if player is not None and board[r][c] != player:
                continue
            p = board[r][c]
            for dr,dc in directions:
                cnt = 0
                rr, cc = r, c
                while 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and board[rr][cc] == p:
                    cnt += 1
                    if cnt >= 5:
                        return True
                    rr += dr; cc += dc
    return False

# ---------------- Line extraction for evaluation ----------------
def extract_lines(board: List[List[str]]) -> List[str]:
    lines = []
    # rows
    for r in range(BOARD_SIZE):
        lines.append("".join(board[r][c] for c in range(BOARD_SIZE)))
    # cols
    for c in range(BOARD_SIZE):
        lines.append("".join(board[r][c] for r in range(BOARD_SIZE)))
    # main diags
    for d in range(-BOARD_SIZE+1, BOARD_SIZE):
        diag = []
        for r in range(BOARD_SIZE):
            c = r - d
            if 0 <= c < BOARD_SIZE:
                diag.append(board[r][c])
        if len(diag) >= 5:
            lines.append("".join(diag))
    # anti-diags
    for d in range(0, 2*BOARD_SIZE-1):
        diag = []
        for r in range(BOARD_SIZE):
            c = d - r
            if 0 <= c < BOARD_SIZE:
                diag.append(board[r][c])
        if len(diag) >= 5:
            lines.append("".join(diag))
    return lines

# ---------------- Pattern-based evaluation ----------------
# We'll map board chars to '1' (player), '2' (opponent), '0' (empty) per line for pattern matching
PATTERNS = [
    # pattern, weight name
    ("11111","FIVE"),
    ("011110","FOUR_OPEN"),
    ("011112","FOUR_SEMI"),
    ("211110","FOUR_SEMI"),
    ("01110","THREE_OPEN"),
    ("010110","THREE_OPEN"),  # broken 3
    ("001112","THREE_SEMI"),
    ("211100","THREE_SEMI"),
    ("00110","TWO_OPEN"),
    ("01010","TWO_OPEN"),
    ("0110","TWO_SEMI"),
    ("1","ONE")
]

def evaluate(board: List[List[str]], player: str) -> int:
    opp = switch_player(player)
    # quick wins
    if check_win_board(board, player):
        return WEIGHTS["FIVE"]
    if check_win_board(board, opp):
        return -WEIGHTS["FIVE"]
    total = 0
    lines = extract_lines(board)
    for raw in lines:
        # create normalized line for player and opponent
        # map spaces to '0'
        s = raw.replace(' ', '0')
        # for player-scoring replace player->'1', opp->'2'
        s_player = s.replace(player, '1').replace(opp, '2')
        s_opp = s.replace(opp, '1').replace(player, '2')
        # count patterns
        for pat, weight_name in PATTERNS:
            w = WEIGHTS.get(weight_name, 0)
            if w == 0: continue
            total += s_player.count(pat) * w
            total -= s_opp.count(pat.replace('1','2').replace('2','1')) * (w * 1.1)  # defense slightly prioritized
    # detect double threats for player (two distinct occurrences of 3-open -> big)
    # naive: if count of three_open patterns >=2 -> bonus
    three_open_pat = "01110"
    player_cnt_three_open = sum(line.replace(' ', '0').replace(player,'1').replace(opp,'2').count(three_open_pat) for line in lines)
    if player_cnt_three_open >= 2:
        total += WEIGHTS["DOUBLE_THREAT"]
    # and symmetric for opponent (penalize strongly)
    opp_cnt_three_open = sum(line.replace(' ', '0').replace(opp,'1').replace(player,'2').count(three_open_pat) for line in lines)
    if opp_cnt_three_open >= 2:
        total -= WEIGHTS["DOUBLE_THREAT"]
    return int(total)

# ---------------- Quick move scoring for ordering ----------------
def immediate_win_or_block(board: List[List[str]], move: Tuple[int,int], player: str) -> Tuple[bool,bool]:
    r,c = move
    if board[r][c] != ' ':
        return False, False
    board[r][c] = player
    win = check_win_board(board, player)
    board[r][c] = ' '
    board[r][c] = switch_player(player)
    block_win = check_win_board(board, switch_player(player))
    board[r][c] = ' '
    return win, block_win

def score_move_quick(board: List[List[str]], move: Tuple[int,int], player: str) -> int:
    # compute a fast heuristic score for ordering:
    r,c = move
    if board[r][c] != ' ':
        return -10**9
    # immediate win / block
    win = False; block = False
    board[r][c] = player
    if check_win_board(board, player):
        win = True
    board[r][c] = ' '
    board[r][c] = switch_player(player)
    if check_win_board(board, switch_player(player)):
        block = True
    board[r][c] = ' '
    if win:
        return WEIGHTS["FIVE"]  # extremely high
    if block:
        return WEIGHTS["FOUR_OPEN"]  # very high
    # shallow evaluation: evaluate local window to speed up - use evaluate but limited
    # We'll use evaluate(board, player) after temporary placement (cheap enough for ordering)
    board[r][c] = player
    s_self = evaluate(board, player)
    board[r][c] = ' '
    board[r][c] = switch_player(player)
    s_opp = evaluate(board, switch_player(player))
    board[r][c] = ' '
    # defense bonus if this move drastically reduces opponent score
    defense_bonus = 0
    if s_opp > WEIGHTS["THREE_OPEN"]:
        defense_bonus += WEIGHTS["FOUR_SEMI"]
    return s_self - s_opp + defense_bonus

# ---------------- Minimax with alpha-beta + TT ----------------
INF = 10**18

def store_tt(key: str, move: Optional[Tuple[int,int]], score: int, depth: int, flag: str):
    TT[key] = {"move": move, "score": score, "depth": depth, "flag": flag}

def minimax_ab(board: List[List[str]],
               depth: int,
               alpha: int,
               beta: int,
               player: str,
               maximizing: bool,
               start_time: float,
               time_limit: float) -> Tuple[int, Optional[Tuple[int,int]]]:
    # time check
    if time.time() - start_time >= time_limit:
        # signal by returning current eval (no move)
        return evaluate(board, player), None
    key = board_to_key(board, player)
    # TT probe
    if key in TT:
        entry = TT[key]
        if entry["depth"] >= depth:
            flag = entry.get("flag", "EXACT")
            val = entry["score"]
            if flag == "EXACT":
                return val, entry.get("move")
            if flag == "LOWER" and val > alpha:
                alpha = val
            if flag == "UPPER" and val < beta:
                beta = val
            if alpha >= beta:
                return val, entry.get("move")
    # terminal or depth 0
    if depth == 0 or check_win_board(board):
        val = evaluate(board, player)
        return val, None
    # generate moves and order
    moves = generate_moves(board, radius=DEFAULT_RADIUS)
    if not moves:
        return 0, None
    # order moves with quick scoring; put winning/blocking first
    moves.sort(key=lambda m: score_move_quick(board, m, player), reverse=True)
    best_move = None
    if maximizing:
        value = -INF
        for m in moves:
            r,c = m
            board[r][c] = player
            score, _ = minimax_ab(board, depth-1, alpha, beta, player, False, start_time, time_limit)
            board[r][c] = ' '
            if score > value:
                value = score
                best_move = m
            if value > alpha:
                alpha = value
            if alpha >= beta:
                break
            # time check
            if time.time() - start_time >= time_limit:
                break
        # store in TT with flag
        if value <= alpha:
            flag = "UPPER"
        elif value >= beta:
            flag = "LOWER"
        else:
            flag = "EXACT"
        store_tt(key, best_move, value, depth, flag)
        return value, best_move
    else:
        value = INF
        opp = switch_player(player)
        for m in moves:
            r,c = m
            board[r][c] = opp
            score, _ = minimax_ab(board, depth-1, alpha, beta, player, True, start_time, time_limit)
            board[r][c] = ' '
            if score < value:
                value = score
                best_move = m
            if value < beta:
                beta = value
            if alpha >= beta:
                break
            if time.time() - start_time >= time_limit:
                break
        # store in TT with flag
        if value <= alpha:
            flag = "UPPER"
        elif value >= beta:
            flag = "LOWER"
        else:
            flag = "EXACT"
        store_tt(key, best_move, value, depth, flag)
        return value, best_move

# ---------------- Opening Book (simple) ----------------
OPENING_MOVES = [
    (BOARD_SIZE//2, BOARD_SIZE//2),
    (BOARD_SIZE//2, BOARD_SIZE//2 - 1),
    (BOARD_SIZE//2 - 1, BOARD_SIZE//2),
    (BOARD_SIZE//2 - 1, BOARD_SIZE//2 - 1),
    (BOARD_SIZE//2, BOARD_SIZE//2 + 1),
    (BOARD_SIZE//2 + 1, BOARD_SIZE//2),
]

# ---------------- Main get_move ----------------
def get_move(board: List[List[str]],
             current_player: str,
             time_limit: float = 0.9,
             max_depth_cap: int = 8) -> Tuple[int,int]:
    """
    Main API: returns (row, col) move within time_limit seconds.
    - uses iterative deepening and resumes from TT if available.
    - max_depth_cap prevents exploding search depth; TT may allow deeper re-use.
    """
    load_tt()  # make sure TT loaded (idempotent)
    start_time = time.time()
    key = board_to_key(board, current_player)

    # opening: if board empty, use opening book first immediately
    if empty_board(board):
        return OPENING_MOVES[0]

    # fallback move from TT or simple heuristic
    best_move: Optional[Tuple[int,int]] = None
    best_score = -INF
    start_depth = 1

    if key in TT:
        entry = TT[key]
        best_move = entry.get("move")
        best_score = entry.get("score", -INF)
        # continue from stored depth+1 to try improve
        start_depth = min(entry.get("depth", 0) + 1, max_depth_cap)
        # but if stored depth already big, we still iterate further up to cap
    else:
        # try shallow quick heuristic for fallback (block immediate threats)
        moves0 = generate_moves(board, radius=DEFAULT_RADIUS)
        # prefer moves that block immediate win
        scored = sorted(moves0, key=lambda m: score_move_quick(board, m, current_player), reverse=True)
        if scored:
            best_move = scored[0]

    # iterative deepening
    depth = start_depth
    # ensure at least start at 1
    if depth < 1:
        depth = 1
    # Cap depth to max_depth_cap to keep time reasonable
    while depth <= max_depth_cap:
        # time check before starting next layer
        if time.time() - start_time >= time_limit:
            break
        score, move = minimax_ab(board, depth, -INF, INF, current_player, True, start_time, time_limit)
        # if time ran out minimax_ab returns eval but may not have a good move; still update if move found
        if move is not None:
            best_move = move
            best_score = score
            # update TT entry
            TT[key] = {"move": best_move, "score": best_score, "depth": depth, "flag": "EXACT"}
        # proceed deeper
        depth += 1

    # if nothing found (edge), pick any generated move
    if best_move is None:
        mlist = generate_moves(board, radius=DEFAULT_RADIUS)
        best_move = mlist[0]

    # persist TT to disk (save partial search)
    save_tt()
    return tuple(best_move)

# ---------------- Load TT once (module init) ----------------
_loaded_flag = False
def load_tt():
    global _loaded_flag
    if _loaded_flag:
        return
    load_tt_impl()

def load_tt_impl():
    # wrapper to call the real load function defined earlier
    load_tt_file()

def load_tt_file():
    # we must avoid name conflict with above load_tt; call underlying loader
    try:
        load_tt_from_disk()
    except Exception as e:
        # fallback: call direct loader
        try:
            load_tt(default_path := TT_FILE)
        except:
            pass

def load_tt_from_disk():
    # Actually call the earlier defined load_tt(path)
    load_tt.__globals__  # no-op to satisfy static check
    # But simpler: call the real function we defined before (name is load_tt)
    # To avoid recursion confusion, call the saved function by accessing from globals:
    g = globals()
    if "load_tt" in g and callable(g["load_tt"]):
        # this would recurse; thus call the loader we defined earlier by referencing it directly:
        pass

# Because we have some naming overlap (load_tt used above),
# let's just call the loader that populates TT at module import:
try:
    # call the top-level loader we wrote earlier (reassign to function ptr first)
    _loader = globals().get('load_tt', None)
    # But 'load_tt' is the wrapper, so instead call the low-level loader 'load_tt' defined earlier in file.
    # To avoid confusion, simply open file if exists and load into TT variable
    if os.path.exists(TT_FILE):
        with open(TT_FILE, "r") as f:
            raw = json.load(f)
        TT.clear()
        for k,v in raw.items():
            entry = dict(v)
            if entry.get("move") is not None:
                entry["move"] = tuple(entry["move"])
            TT[k] = entry
except Exception:
    TT = {}

# ---------------- Example quick test ----------------
if __name__ == "__main__":
    # quick interactive demo: empty board -> get_move
    board = [[' ' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    mv = get_move(board, 'X', time_limit=0.9, max_depth_cap=4)
    print("Suggested move on empty board:", mv)
