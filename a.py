"""
Gomoku Trainer v4 (updated)

Changelog for this update (fixes the "identical games" issue):
- Added controlled randomness and exploration so self-play doesn't produce identical games every run.
  - `DETERMINISTIC` flag: when True, runs are repeatable. When False, we seed RNG from system time so runs vary.
  - Root-level tie-breaking and candidate ordering include randomness to avoid deterministic tie cycles.
  - Epsilon-greedy exploration: with small probability the bot will pick a random candidate instead of the search-best move (useful during training to diversify data).
  - Opening-book usage includes an exploration probability so the bot doesn't always follow the book.
- Improved safe pickle handling and atomic saves.
- Kept Zobrist + transposition + opening-book (dict) as before.
- Improved root move selection: when multiple moves score nearly equal, choose randomly among top moves.

Save as: gomoku_trainer_v4.py
Run: python gomoku_trainer_v4.py

Adjust parameters under `if __name__ == '__main__'` or in CONFIG block.
"""

import time
import random
import pickle
import os
from collections import defaultdict, Counter

# ---------------- CONFIG ----------------
SIZE = 15
MAX_TIME = 0.9  # seconds per move
CACHE_FILE = "gomoku_cache_v4.pkl"
OPENING_FILE = "gomoku_opening_v4.pkl"
ZOBRIST_FILE = "gomoku_zobrist_v4.pkl"  # optional persistence for Zobrist table

# Randomness / exploration controls
DETERMINISTIC = False  # Set True for reproducible runs (uses SEED). False -> varied runs.
SEED = 123456
OPENING_EXPLORE_PROB = 0.10  # when book available, with this prob ignore it and search
EPSILON_RANDOM_MOVE = 0.05  # with this probability at root pick random candidate (exploration)

MAX_DEPTH_CAP = 4
SAVE_EVERY = 10
VERBOSE = False  # set True to print board each move for debugging

# ---------------- Random / Zobrist init ----------------
if DETERMINISTIC:
    random.seed(SEED)
else:
    random.seed(None)  # system/time-based

# try to load persisted zobrist (keeps opening book compatible across runs)
if os.path.exists(ZOBRIST_FILE):
    try:
        with open(ZOBRIST_FILE, 'rb') as f:
            zobrist = pickle.load(f)
    except Exception:
        zobrist = None
else:
    zobrist = None

if zobrist is None:
    zobrist = [[[random.getrandbits(64) for _ in range(2)] for _ in range(SIZE)] for _ in range(SIZE)]
    # persist zobrist for future runs (atomic)
    try:
        tmp = ZOBRIST_FILE + ".tmp"
        with open(tmp, 'wb') as f:
            pickle.dump(zobrist, f)
        os.replace(tmp, ZOBRIST_FILE)
    except Exception:
        pass

# ------------- Persistence helpers -------------

def load_pickle(path):
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except (EOFError, pickle.PickleError, Exception):
            # file empty/corrupt or other pickle error -> ignore and return None
            return None
    return None


def save_pickle(obj, path):
    tmp = path + ".tmp"
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)

# Load caches (handle corrupted/empty files)
transposition = load_pickle(CACHE_FILE) or {}
opening_book = load_pickle(OPENING_FILE) or {}  # dict: zobrist_hash -> { move_tuple: {plays: int, score: float} }

# ------------- Utilities -------------

def in_bounds(r, c):
    return 0 <= r < SIZE and 0 <= c < SIZE


def zobrist_hash(board):
    h = 0
    for r in range(SIZE):
        for c in range(SIZE):
            v = board[r][c]
            if v == 'X':
                h ^= zobrist[r][c][0]
            elif v == 'O':
                h ^= zobrist[r][c][1]
    return h

# ------------- Patterns & Evaluation -------------
PATTERNS = {
    "11111": 10_000_000,
    "011110": 500_000,
    "011112": 10_000,
    "211110": 10_000,
    "01110": 5_000,
    "010110": 4_000,
    "0110": 500,
}

# extract all lines for scanning
def extract_lines(board):
    lines = []
    for r in range(SIZE):
        lines.append(('row', r, ''.join(board[r])))
    for c in range(SIZE):
        lines.append(('col', c, ''.join(board[r][c] for r in range(SIZE))))
    for d in range(-SIZE+1, SIZE):
        diag = ''.join(board[r][r-d] for r in range(SIZE) if 0 <= r-d < SIZE)
        lines.append(('d1', d, diag))
    for d in range(-SIZE+1, SIZE):
        diag = ''.join(board[r][SIZE-1-r-d] for r in range(SIZE) if 0 <= SIZE-1-r-d < SIZE)
        lines.append(('d2', d, diag))
    return lines


def line_to_code(line, player):
    opp = 'O' if player == 'X' else 'X'
    return line.replace(player, '1').replace(opp, '2').replace(' ', '0')

# count local patterns for a potential move
def count_patterns_for_move(board, move_r, move_c, player):
    board[move_r][move_c] = player
    counts = {'open_three':0, 'open_four':0, 'five':0}
    dirs = [(0,1),(1,0),(1,1),(1,-1)]
    for dr, dc in dirs:
        window = []
        for k in range(-5,6):
            rr = move_r + k*dr
            cc = move_c + k*dc
            if in_bounds(rr, cc):
                window.append(board[rr][cc])
            else:
                window.append('#')
        s = ''.join(window)
        code = s.replace(player, '1').replace(('O' if player=='X' else 'X'), '2').replace(' ', '0').replace('#','3')
        counts['five'] += code.count('11111')
        counts['open_four'] += code.count('011110')
        counts['open_three'] += code.count('01110') + code.count('010110')
    board[move_r][move_c] = ' '
    return counts


def score_for(board, player):
    opp = 'O' if player == 'X' else 'X'
    base = 0
    for _,_,line in extract_lines(board):
        code = line_to_code(line, player)
        for pat,val in PATTERNS.items():
            if pat in code:
                base += val * code.count(pat)
    # detect double-threats among candidates
    candidates = get_candidates(board)
    double_three = 0
    double_four = 0
    for r,c in candidates:
        counts = count_patterns_for_move(board, r, c, player)
        if counts['open_three'] >= 2:
            double_three += 1
        if counts['open_four'] >= 2:
            double_four += 1
        if counts['five'] > 0:
            base += PATTERNS['11111'] * counts['five']
    base += double_three * 200_000
    base += double_four * 500_000
    # opponent threats penalty
    opp_double_three = 0
    opp_double_four = 0
    for r,c in candidates:
        ccounts = count_patterns_for_move(board, r, c, opp)
        if ccounts['open_three'] >= 2:
            opp_double_three += 1
        if ccounts['open_four'] >= 2:
            opp_double_four += 1
    base -= opp_double_three * 400_000
    base -= opp_double_four * 800_000
    return base


def evaluate(board, player):
    opp = 'O' if player == 'X' else 'X'
    return score_for(board, player) - score_for(board, opp)

# ------------- Candidates & ordering -------------

def get_candidates(board, radius=2):
    moves = set()
    for r in range(SIZE):
        for c in range(SIZE):
            if board[r][c] != ' ':
                for dr in range(-radius, radius+1):
                    for dc in range(-radius, radius+1):
                        nr, nc = r+dr, c+dc
                        if in_bounds(nr,nc) and board[nr][nc] == ' ':
                            moves.add((nr,nc))
    if not moves:
        moves.add((SIZE//2, SIZE//2))
    # order by neighbor density (simple heuristic)
    scored = []
    for (r,c) in moves:
        neigh = 0
        for dr in range(-2,3):
            for dc in range(-2,3):
                rr,cc = r+dr, c+dc
                if in_bounds(rr,cc) and board[rr][cc] != ' ':
                    neigh += 1
        scored.append(((r,c), neigh))
    # shuffle for random tie-breaks, then stable sort
    items = scored[:]
    random.shuffle(items)
    items.sort(key=lambda x: -x[1])
    return [m for m,_ in items]

# ---------------- Alpha-Beta + transposition ----------------

def alpha_beta(board, depth, alpha, beta, maximizing, player, start_time, h):
    # time cutoff
    if time.time() - start_time > MAX_TIME * 0.95:
        return evaluate(board, player)

    entry = transposition.get(h)
    if entry and entry.get('depth', -1) >= depth:
        return entry['value']

    if depth == 0:
        val = evaluate(board, player)
        transposition[h] = {'value': val, 'depth': 0}
        return val

    opponent = 'O' if player == 'X' else 'X'
    candidates = get_candidates(board)

    if maximizing:
        max_eval = -1e18
        for r,c in candidates:
            board[r][c] = player
            nh = h ^ zobrist[r][c][0 if player=='X' else 1]
            val = alpha_beta(board, depth-1, alpha, beta, False, player, start_time, nh)
            board[r][c] = ' '
            if val > max_eval:
                max_eval = val
            alpha = max(alpha, val)
            if beta <= alpha:
                break
        transposition[h] = {'value': max_eval, 'depth': depth}
        return max_eval
    else:
        min_eval = 1e18
        for r,c in candidates:
            board[r][c] = opponent
            nh = h ^ zobrist[r][c][0 if opponent=='X' else 1]
            val = alpha_beta(board, depth-1, alpha, beta, True, player, start_time, nh)
            board[r][c] = ' '
            if val < min_eval:
                min_eval = val
            beta = min(beta, val)
            if beta <= alpha:
                break
        transposition[h] = {'value': min_eval, 'depth': depth}
        return min_eval

# quick static ordering value
def quick_move_value(board, mv, player):
    r,c = mv
    neigh = 0
    for dr in range(-2,3):
        for dc in range(-2,3):
            rr,cc = r+dr, c+dc
            if in_bounds(rr,cc) and board[rr][cc] != ' ':
                neigh += 1
    counts = count_patterns_for_move(board, r, c, player)
    val = neigh * 10 + counts['open_three'] * 1000 + counts['open_four'] * 5000 + counts['five'] * 100000
    return val

# -------------- get_move (iterative deepening + opening book + exploration) --------------

def get_move(board, current_player, time_limit=MAX_TIME, explore_prob=EPSILON_RANDOM_MOVE):
    start_time = time.time()
    best_move = None
    best_score = -1e18

    candidates = get_candidates(board)
    h = zobrist_hash(board)

    # opening book: use early-game info, but allow exploration
    stone_count = sum(1 for r in range(SIZE) for c in range(SIZE) if board[r][c] != ' ')
    if stone_count <= 6 and random.random() > OPENING_EXPLORE_PROB:
        k = h
        if k in opening_book and opening_book[k]:
            moves_dict = opening_book[k]
            # choose move with best win-rate, break ties randomly
            best = None
            best_rate = -1e9
            candidates_stats = []
            for mv, stats in moves_dict.items():
                rate = stats['score'] / max(1, stats['plays'])
                candidates_stats.append((mv, rate, stats['plays']))
            if candidates_stats:
                # sort by (rate, plays) and pick among top group randomly
                candidates_stats.sort(key=lambda x: (x[1], x[2]), reverse=True)
                top_rate = candidates_stats[0][1]
                top_choices = [mv for mv,rate,p in candidates_stats if abs(rate - top_rate) < 1e-9]
                choice = random.choice(top_choices)
                if random.random() > explore_prob:
                    return choice
                # else proceed to search (explore)

    depth = 1
    while True:
        if time.time() - start_time > time_limit * 0.9:
            break
        local_moves = []  # (move, score)

        # reorder candidates by quick static heuristic (shuffle first for tie-break randomness)
        ordered = candidates[:]
        random.shuffle(ordered)
        ordered.sort(key=lambda mv: -quick_move_value(board, mv, current_player))

        for r,c in ordered:
            if time.time() - start_time > time_limit * 0.9:
                break
            # exploration at root: with small prob pick random candidate immediately
            if random.random() < explore_prob:
                continue  # skip search for this candidate so others get considered; remaining randomness handled below

            board[r][c] = current_player
            nh = h ^ zobrist[r][c][0 if current_player=='X' else 1]
            # immediate win check using exact win detection
            if check_win(board, r, c):
                board[r][c] = ' '
                return (r,c)
            val = alpha_beta(board, depth-1, -1e18, 1e18, False, current_player, start_time, nh)
            board[r][c] = ' '
            local_moves.append(((r,c), val))

        if not local_moves and ordered:
            # all candidates were skipped due to exploration; pick one random from ordered
            best_move = random.choice(ordered)
            break

        # pick best among local_moves, but if many near-equal choose randomly among top-k
        if local_moves:
            local_moves.sort(key=lambda x: -x[1])
            top_score = local_moves[0][1]
            # include moves within a small relative tolerance
            tol = max(1e-6, abs(top_score) * 1e-4)
            top_choices = [mv for mv,sc in local_moves if top_score - sc <= tol]
            choice = random.choice(top_choices)
            best_move = choice
            best_score = top_score

        depth += 1
        if depth > MAX_DEPTH_CAP:
            break
    if best_move is None:
        # fallback to random candidate (prefer high quick_move_value)
        ordered = candidates[:]
        random.shuffle(ordered)
        ordered.sort(key=lambda mv: -quick_move_value(board, mv, current_player))
        best_move = ordered[0]
    return best_move

# ------------- Win detection (accurate) -------------

def check_win(board, last_r, last_c):
    if last_r is None or last_c is None:
        return None
    player = board[last_r][last_c]
    if player == ' ':
        return None
    dirs = [(0,1),(1,0),(1,1),(1,-1)]
    for dr,dc in dirs:
        cnt = 1
        rr,cc = last_r+dr, last_c+dc
        while in_bounds(rr,cc) and board[rr][cc] == player:
            cnt += 1
            rr += dr; cc += dc
        rr,cc = last_r-dr, last_c-dc
        while in_bounds(rr,cc) and board[rr][cc] == player:
            cnt += 1
            rr -= dr; cc -= dc
        if cnt >= 5:
            return player
    return None

# ------------- Self-play / training -------------

def play_game(mode="AI_vs_AI", time_per_move=MAX_TIME, max_moves=SIZE*SIZE, verbose=False, explore_prob=EPSILON_RANDOM_MOVE):
    board = [[' ' for _ in range(SIZE)] for _ in range(SIZE)]
    move_history = []  # (player, (r,c), zobrist_after_move)
    last_r = None; last_c = None
    for turn in range(max_moves):
        current_player = 'X' if turn % 2 == 0 else 'O'

        # exploration policy for training modes
        if mode == "AI_vs_AI":
            mv = get_move(board, current_player, time_limit=time_per_move, explore_prob=explore_prob)
        elif mode == "AI_vs_Random":
            if current_player == 'X':
                mv = get_move(board, current_player, time_limit=time_per_move, explore_prob=explore_prob)
            else:
                mv = random.choice(get_candidates(board))
        elif mode == "Random_vs_AI":
            if current_player == 'X':
                mv = random.choice(get_candidates(board))
            else:
                mv = get_move(board, current_player, time_limit=time_per_move, explore_prob=explore_prob)
        else:
            mv = get_move(board, current_player, time_limit=time_per_move, explore_prob=explore_prob)

        r,c = mv
        if board[r][c] != ' ':
            # fallback to any free candidate
            cand = [m for m in get_candidates(board) if board[m[0]][m[1]] == ' ']
            if not cand:
                break
            r,c = random.choice(cand)
        board[r][c] = current_player
        move_history.append((current_player, (r,c), zobrist_hash(board)))
        last_r, last_c = r, c

        if verbose or VERBOSE:
            print(f"Move {len(move_history)}: {current_player} -> {(r,c)}")
            print_board(board)

        winner = check_win(board, last_r, last_c)
        if winner:
            return winner, move_history, board
    return None, move_history, board


def train(n_games=100, mode="AI_vs_AI", verbose=False, explore_prob=EPSILON_RANDOM_MOVE):
    stats = Counter()
    total_moves = 0
    for i in range(n_games):
        start = time.time()
        winner, moves, final_board = play_game(mode=mode, verbose=verbose, explore_prob=explore_prob)
        elapsed = time.time() - start
        stats[winner] += 1
        total_moves += len(moves)
        print(f"Game {i+1}/{n_games}: winner={winner} moves={len(moves)} time={elapsed:.2f}s")

        # update opening book from replay; result mapping: winner gets +1, loser -1, draw 0.5
        if winner == 'X':
            result_map = {'X': 1.0, 'O': -1.0}
        elif winner == 'O':
            result_map = {'X': -1.0, 'O': 1.0}
        else:
            result_map = {'X': 0.5, 'O': 0.5}

        # replay moves sequentially, record stats for opening positions
        board_replay = [[' ']*SIZE for _ in range(SIZE)]
        for pl, mv, _ in moves:
            k = zobrist_hash(board_replay)
            if k not in opening_book:
                opening_book[k] = {}
            if mv not in opening_book[k]:
                opening_book[k][mv] = {'plays': 0, 'score': 0.0}
            opening_book[k][mv]['plays'] += 1
            opening_book[k][mv]['score'] += result_map[pl]
            board_replay[mv[0]][mv[1]] = pl

        # save periodically
        if (i+1) % SAVE_EVERY == 0:
            save_pickle(transposition, CACHE_FILE)
            save_pickle(opening_book, OPENING_FILE)
    save_pickle(transposition, CACHE_FILE)
    save_pickle(opening_book, OPENING_FILE)
    print("Training finished.")
    print("Stats:", dict(stats))
    print("Avg moves:", total_moves / max(1, n_games))

# ------------- Helpers for debugging -------------

def print_board(board):
    header = '   ' + ' '.join(f"{c:2d}" for c in range(SIZE))
    print(header)
    for r in range(SIZE):
        row = f"{r:2d} " + ' '.join(board[r][c] if board[r][c] != ' ' else '.' for c in range(SIZE))
        print(row)
    print()

# ------------- Main -------------
if __name__ == '__main__':
    N_GAMES = 200
    MODE = 'AI_vs_AI'  # 'AI_vs_AI', 'AI_vs_Random', 'Random_vs_AI'
    VERBOSE = False
    DETERMINISTIC = False  # change True if you need repeatable runs

    print(f"Starting training: {N_GAMES} games, mode={MODE}, deterministic={DETERMINISTIC}")
    train(N_GAMES, mode=MODE, verbose=VERBOSE, explore_prob=EPSILON_RANDOM_MOVE)

    # quick test position
    test_board = [[' ']*SIZE for _ in range(SIZE)]
    test_board[7][7] = 'X'
    test_board[7][8] = 'X'
    test_board[8][7] = 'O'
    test_board[6][8] = 'O'
    print("Sample suggested move for X:", get_move(test_board, 'X'))
    print("Caches saved to:", CACHE_FILE, OPENING_FILE)

