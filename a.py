"""
Gomoku Trainer v2
- Zobrist hashing + transposition table (persist to file)
- Heuristic evaluation with explicit double-three/double-four detection
- Iterative deepening alpha-beta with move ordering and time cutoff per move
- Opening book using Zobrist hash with win-rate based selection; updated from self-play
- Self-play training loop (AI vs AI / AI vs Random / Random vs AI)

How to use:
  python gomoku_trainer_v2.py

You can adjust parameters at the bottom of the file under `if __name__ == '__main__'`.
"""

import time
import random
import pickle
import os
from collections import defaultdict, Counter

# ------------------------ Config ------------------------
SIZE = 15
MAX_TIME = 0.9  # seconds per move
CACHE_FILE = "gomoku_cache_v2.pkl"
OPENING_FILE = "gomoku_opening_v2.pkl"
SEED = 12345
MAX_DEPTH_CAP = 4  # safety cap for iterative deepening
SAVE_EVERY = 10  # save caches every N games
random.seed(SEED)

# ------------------------ Zobrist ------------------------
zobrist = [[[random.getrandbits(64) for _ in range(2)] for _ in range(SIZE)] for _ in range(SIZE)]

# ------------------------ Persistence ------------------------

def load_pickle(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

# Load caches if present
transposition = load_pickle(CACHE_FILE) or {}
# transposition maps zobrist_hash -> {'value':..., 'depth':..., 'flag': 'exact'/'lower'/'upper'}


# ------------------------ Opening Book (Zobrist key) ------------------------
class OpeningBook:
    def __init__(self, path=OPENING_FILE):
        self.path = path
        self.data = load_pickle(path) or defaultdict(lambda: defaultdict(lambda: {"plays":0, "score":0.0}))

    def key(self, board):
        return zobrist_hash(board)

    def record(self, board, move, result):
        k = self.key(board)
        self.data[k][move]["plays"] += 1
        self.data[k][move]["score"] += result

    def choose(self, board):
        k = self.key(board)
        if k not in self.data or not self.data[k]:
            return None
        ranked = sorted(self.data[k].items(), key=lambda x: (x[1]["score"] / max(1, x[1]["plays"]), x[1]["plays"]), reverse=True)
        return ranked[0][0]

    def save(self):
        save_pickle(self.data, self.path)


opening_book = OpeningBook()

# ------------------------ Utilities ------------------------

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

# ------------------------ Heuristics / Patterns ------------------------
PATTERNS = {
    "11111": 10_000_000,
    "011110": 500_000,
    "011112": 10_000,
    "211110": 10_000,
    "01110": 5_000,
    "010110": 4_000,
    "0110": 500,
}

# Quick line extraction (rows, cols, diags)
def extract_lines(board):
    lines = []
    # rows
    for r in range(SIZE):
        lines.append(('row', r, ''.join(board[r])))
    # cols
    for c in range(SIZE):
        col = ''.join(board[r][c] for r in range(SIZE))
        lines.append(('col', c, col))
    # diag TL-BR
    for d in range(-SIZE+1, SIZE):
        diag = ''.join(board[r][r-d] for r in range(SIZE) if 0 <= r-d < SIZE)
        lines.append(('d1', d, diag))
    # diag TR-BL
    for d in range(-SIZE+1, SIZE):
        diag = ''.join(board[r][SIZE-1-r-d] for r in range(SIZE) if 0 <= SIZE-1-r-d < SIZE)
        lines.append(('d2', d, diag))
    return lines


def line_to_code(line, player):
    opp = 'O' if player == 'X' else 'X'
    return line.replace(player, '1').replace(opp, '2').replace(' ', '0')

# For detection of open-three/four around a specific move

def count_patterns_for_move(board, move_r, move_c, player):
    board[move_r][move_c] = player
    counts = {'open_three':0, 'open_four':0, 'five':0}
    dirs = [(0,1),(1,0),(1,1),(1,-1)]
    for dr, dc in dirs:
        line = []
        # collect +/-5 range centered at move
        for k in range(-5,6):
            rr = move_r + k*dr
            cc = move_c + k*dc
            if in_bounds(rr, cc):
                line.append(board[rr][cc])
            else:
                line.append('#')
        s = ''.join(line)
        code = s.replace(player, '1').replace(('O' if player=='X' else 'X'), '2').replace(' ', '0').replace('#','3')
        counts['five'] += code.count('11111')
        counts['open_four'] += code.count('011110')
        counts['open_three'] += code.count('01110') + code.count('010110')
    board[move_r][move_c] = ' '
    return counts


# Full board scoring with double-threat awareness

def score_for(board, player):
    opp = 'O' if player == 'X' else 'X'
    base = 0
    # pattern occurrences across lines
    for kind, idx, line in extract_lines(board):
        code = line_to_code(line, player)
        for pat, val in PATTERNS.items():
            if pat in code:
                base += val * code.count(pat)
    # candidate moves for detecting double threats
    candidates = get_candidates(board)
    double_three = 0
    double_four = 0
    for (r,c) in candidates:
        counts = count_patterns_for_move(board, r, c, player)
        if counts['open_three'] >= 2:
            double_three += 1
        if counts['open_four'] >= 2:
            double_four += 1
        if counts['five'] > 0:
            base += PATTERNS['11111'] * counts['five']
    base += double_three * 200_000
    base += double_four * 500_000
    # subtract opponent double threats penalty
    opp_double_three = 0
    opp_double_four = 0
    for (r,c) in candidates:
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

# ------------------------ Candidate generation & ordering ------------------------

def get_candidates(board, radius=2):
    moves = set()
    for r in range(SIZE):
        for c in range(SIZE):
            if board[r][c] != ' ':
                for dr in range(-radius, radius+1):
                    for dc in range(-radius, radius+1):
                        nr, nc = r+dr, c+dc
                        if in_bounds(nr, nc) and board[nr][nc] == ' ':
                            moves.add((nr, nc))
    if not moves:
        moves.add((SIZE//2, SIZE//2))
    # quick ordering: neighbor count and simple pattern potential
    scored = []
    for (r,c) in moves:
        neigh = 0
        for dr in range(-1,2):
            for dc in range(-1,2):
                rr, cc = r+dr, c+dc
                if in_bounds(rr,cc) and board[rr][cc] != ' ':
                    neigh += 1
        scored.append(((r,c), neigh))
    scored.sort(key=lambda x: -x[1])
    return [m for m,_ in scored]

# ------------------------ Alpha-Beta with transposition ------------------------

def alpha_beta(board, depth, alpha, beta, maximizing, player, start_time, h):
    # time cutoff
    if time.time() - start_time > MAX_TIME * 0.95:
        return evaluate(board, player)

    # transposition lookup (only use if depth sufficient)
    entry = transposition.get(h)
    if entry and entry['depth'] >= depth:
        # Use stored value (we store exact approximations)
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

# ------------------------ get_move with iterative deepening & opening book ------------------------

def get_move(board, current_player, time_limit=MAX_TIME):
    start_time = time.time()
    opponent = 'O' if current_player == 'X' else 'X'
    best_move = None
    best_score = -1e18

    candidates = get_candidates(board)
    h = zobrist_hash(board)

    # Opening book lookup for early game
    stone_count = sum(1 for r in range(SIZE) for c in range(SIZE) if board[r][c] != ' ')
    if stone_count <= 6:
        mv = opening_book.choose(board)
        if mv:
            return mv

    depth = 1
    while True:
        if time.time() - start_time > time_limit * 0.9:
            break
        local_best = None
        local_best_score = -1e18
        # re-order candidates per depth by quick heuristic: evaluate move statically
        ordered = sorted(candidates, key=lambda mv: -quick_move_value(board, mv, current_player))
        for r,c in ordered:
            if time.time() - start_time > time_limit * 0.9:
                break
            # play
            board[r][c] = current_player
            nh = h ^ zobrist[r][c][0 if current_player=='X' else 1]
            # immediate win detection
            if score_for(board, current_player) >= PATTERNS['11111']:
                board[r][c] = ' '
                return (r,c)
            val = alpha_beta(board, depth-1, -1e18, 1e18, False, current_player, start_time, nh)
            board[r][c] = ' '
            if val > local_best_score:
                local_best_score = val
                local_best = (r,c)
        if local_best is not None:
            best_move = local_best
            best_score = local_best_score
        depth += 1
        if depth > MAX_DEPTH_CAP:
            break
    if best_move is None:
        best_move = candidates[0]
    return best_move

# Quick static heuristic for ordering candidate moves
def quick_move_value(board, mv, player):
    r,c = mv
    # neighbor count
    neigh = 0
    for dr in range(-2,3):
        for dc in range(-2,3):
            rr,cc = r+dr, c+dc
            if in_bounds(rr,cc) and board[rr][cc] != ' ':
                neigh += 1
    # temporary counts for patterns
    counts = count_patterns_for_move(board, r, c, player)
    val = neigh * 10 + counts['open_three'] * 1000 + counts['open_four'] * 5000 + counts['five'] * 100000
    return val

# ------------------------ Win detection ------------------------

def check_win(board, last_r, last_c):
    if last_r is None:
        return None
    player = board[last_r][last_c]
    if player == ' ':
        return None
    dirs = [(0,1),(1,0),(1,1),(1,-1)]
    for dr,dc in dirs:
        cnt = 1
        rr,cc = last_r+dr, last_c+dc
        while in_bounds(rr,cc) and board[rr][cc]==player:
            cnt += 1; rr += dr; cc += dc
        rr,cc = last_r-dr, last_c-dc
        while in_bounds(rr,cc) and board[rr][cc]==player:
            cnt += 1; rr -= dr; cc -= dc
        if cnt >= 5:
            return player
    return None

# ------------------------ Self-play training ------------------------

def play_game(mode="AI_vs_AI", time_per_move=MAX_TIME, max_moves=SIZE*SIZE):
    board = [[' ' for _ in range(SIZE)] for _ in range(SIZE)]
    move_history = []
    last_r = None; last_c = None
    for turn in range(max_moves):
        current_player = 'X' if turn % 2 == 0 else 'O'
        if mode == "AI_vs_AI":
            mv = get_move(board, current_player, time_limit=time_per_move)
        elif mode == "AI_vs_Random":
            mv = get_move(board, current_player, time_limit=time_per_move) if current_player == 'X' else random.choice(get_candidates(board))
        elif mode == "Random_vs_AI":
            mv = random.choice(get_candidates(board)) if current_player == 'X' else get_move(board, current_player, time_limit=time_per_move)
        else:
            mv = get_move(board, current_player, time_limit=time_per_move)

        r,c = mv
        if board[r][c] != ' ':
            # collision fallback
            cand = [m for m in get_candidates(board) if board[m[0]][m[1]]==' ']
            if not cand:
                break
            r,c = random.choice(cand)
        board[r][c] = current_player
        move_history.append((current_player, (r,c), zobrist_hash(board)))
        last_r, last_c = r, c
        winner = check_win(board, last_r, last_c)
        if winner:
            return winner, move_history, board
    return None, move_history, board


def train(n_games=100, mode="AI_vs_AI"):
    stats = Counter()
    total_moves = 0
    for i in range(n_games):
        start = time.time()
        winner, moves, final_board = play_game(mode=mode)
        elapsed = time.time() - start
        stats[winner] += 1
        total_moves += len(moves)
        print(f"Game {i+1}/{n_games}: winner={winner} moves={len(moves)} time={elapsed:.2f}s")

        # update opening book using replayed moves
        if winner == 'X':
            result_map = {'X': 1.0, 'O': -1.0}
        elif winner == 'O':
            result_map = {'X': -1.0, 'O': 1.0}
        else:
            result_map = {'X': 0.5, 'O': 0.5}

        board_replay = [[' ']*SIZE for _ in range(SIZE)]
        for pl, mv, hsh in moves:
            opening_book.record(board_replay, mv, result_map[pl])
            board_replay[mv[0]][mv[1]] = pl

        # periodically save
        if (i+1) % SAVE_EVERY == 0:
            save_pickle(transposition, CACHE_FILE)
            opening_book.save()
    save_pickle(transposition, CACHE_FILE)
    opening_book.save()
    print("Training finished.")
    print("Stats:", dict(stats))
    print("Avg moves:", total_moves / max(1, n_games))

# ------------------------ Tests helpers ------------------------

def test_positions(cases):
    results = []
    for i, case in enumerate(cases):
        board = [list(row) for row in case['board']]
        player = case['player']
        mv = get_move(board, player)
        print(f"Case {i+1}, player={player} -> move={mv}")
        results.append(mv)
    return results

# ------------------------ Main ------------------------
if __name__ == '__main__':
    # Example default run: train a number of self-play games
    N_GAMES = 50
    MODE = 'AI_vs_AI'  # 'AI_vs_AI', 'AI_vs_Random', 'Random_vs_AI'
    print(f"Starting training: {N_GAMES} games, mode={MODE}")
    train(N_GAMES, MODE)

    # Quick test scenario
    sample_board = [[' ']*SIZE for _ in range(SIZE)]
    sample_board[7][7] = 'X'
    sample_board[7][8] = 'X'
    sample_board[8][7] = 'O'
    sample_board[6][8] = 'O'
    print("Testing sample position for X...")
    mv = get_move(sample_board, 'X')
    print("Suggested move:", mv)

    print("Done. Caches saved to:", CACHE_FILE, OPENING_FILE)
