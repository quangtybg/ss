import time
import random
import pickle
import os
from collections import namedtuple

BOARD_SIZE = 15
CACHE_PATH = "gomoku_cache.pkl"
ZOBRIST_PATH = "gomoku_zobrist.pkl"

# Scoring weights (tuneable)
SCORES = {
    "FIVE": 10**9,
    "OPEN_FOUR": 10**6,
    "FOUR": 10**5,
    "OPEN_THREE": 10**4,
    "THREE": 10**3,
    "TWO": 100,
}

directions = [(1,0),(0,1),(1,1),(1,-1)]

def in_bounds(r,c):
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE

# --- Zobrist hashing and persistent cache ---
def init_or_load_zobrist():
    if os.path.exists(ZOBRIST_PATH):
        with open(ZOBRIST_PATH, "rb") as f:
            zobrist = pickle.load(f)
            return zobrist
    random.seed(1234567)
    zobrist = [[[random.getrandbits(64) for _ in range(2)] for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    with open(ZOBRIST_PATH, "wb") as f:
        pickle.dump(zobrist, f)
    return zobrist

ZOBRIST = init_or_load_zobrist()

def board_hash(board):
    h = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            v = board[r][c]
            if v == 'X':
                h ^= ZOBRIST[r][c][0]
            elif v == 'O':
                h ^= ZOBRIST[r][c][1]
    return h

def load_tt():
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "rb") as f:
                return pickle.load(f)
        except Exception:
            return {}
    return {}

def save_tt(tt):
    try:
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(tt, f)
    except Exception:
        pass

# --- Basic helpers ---
def is_win_after_move(board, row, col, player):
    # assumes board[row][col] is ' ' (or already player's), check if placing player leads to >=5 in a row
    for dr,dc in directions:
        cnt = 1
        r,c = row+dr, col+dc
        while in_bounds(r,c) and board[r][c] == player:
            cnt += 1; r += dr; c += dc
        r,c = row-dr, col-dc
        while in_bounds(r,c) and board[r][c] == player:
            cnt += 1; r -= dr; c -= dc
        if cnt >= 5:
            return True
    return False

def generate_candidates(board, radius=2, max_candidates=80):
    occupied = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != ' ':
                occupied.append((r,c))
    if not occupied:
        return [(BOARD_SIZE//2, BOARD_SIZE//2)]
    neigh = set()
    for (r,c) in occupied:
        for dr in range(-radius, radius+1):
            for dc in range(-radius, radius+1):
                rr,cc = r+dr, c+dc
                if in_bounds(rr,cc) and board[rr][cc] == ' ':
                    neigh.add((rr,cc))
    candidates = list(neigh)
    if len(candidates) > max_candidates:
        def closeness(cell):
            r,c = cell
            md = min(abs(r-ox)+abs(c-oy) for ox,oy in occupied)
            return md
        candidates.sort(key=closeness)
        candidates = candidates[:max_candidates]
    return candidates

def quick_heuristic_move_score(board, row, col, player):
    s = 0
    opponent = 'O' if player == 'X' else 'X'
    for dr,dc in directions:
        cnt = 1
        open_ends = 0
        r,c = row+dr, col+dc
        while in_bounds(r,c) and board[r][c] == player:
            cnt += 1; r += dr; c += dc
        if in_bounds(r,c) and board[r][c] == ' ':
            open_ends += 1
        r,c = row-dr, col-dc
        while in_bounds(r,c) and board[r][c] == player:
            cnt += 1; r -= dr; c -= dc
        if in_bounds(r,c) and board[r][c] == ' ':
            open_ends += 1
        if cnt >= 5:
            s += SCORES["FIVE"]
        elif cnt == 4 and open_ends >= 1:
            s += SCORES["OPEN_FOUR"]
        elif cnt == 3 and open_ends == 2:
            s += SCORES["OPEN_THREE"]
        else:
            s += cnt*10
    return s

def evaluate(board, player):
    opponent = 'O' if player == 'X' else 'X'
    def line_score(line, who):
        n = len(line)
        score = 0
        i = 0
        while i < n:
            if line[i] != who:
                i += 1; continue
            j = i
            while j < n and line[j] == who:
                j += 1
            length = j - i
            left_empty = (i-1 >= 0 and line[i-1] == ' ')
            right_empty = (j < n and line[j] == ' ')
            if length >= 5:
                return SCORES["FIVE"]
            if length == 4 and (left_empty or right_empty):
                score += SCORES["OPEN_FOUR"]
            elif length == 4:
                score += SCORES["FOUR"]
            elif length == 3 and (left_empty and right_empty):
                score += SCORES["OPEN_THREE"]
            elif length == 3:
                score += SCORES["THREE"]
            elif length == 2:
                score += SCORES["TWO"]
            else:
                score += length * 2
            i = j
        return score

    total = 0
    for r in range(BOARD_SIZE):
        row = board[r]
        total += line_score(row, player)
        total -= line_score(row, opponent)
    for c in range(BOARD_SIZE):
        col = [board[r][c] for r in range(BOARD_SIZE)]
        total += line_score(col, player)
        total -= line_score(col, opponent)
    for k in range(-BOARD_SIZE+1, BOARD_SIZE):
        diag = []
        for r in range(BOARD_SIZE):
            c = r - k
            if 0 <= c < BOARD_SIZE:
                diag.append(board[r][c])
        if len(diag) >= 2:
            total += line_score(diag, player)
            total -= line_score(diag, opponent)
    for k in range(0, 2*BOARD_SIZE):
        diag = []
        for r in range(BOARD_SIZE):
            c = k - r
            if 0 <= c < BOARD_SIZE:
                diag.append(board[r][c])
        if len(diag) >= 2:
            total += line_score(diag, player)
            total -= line_score(diag, opponent)
    return total

def count_next_winning_moves(board, player):
    cnt = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] == ' ':
                if is_win_after_move(board, r, c, player):
                    cnt += 1
    return cnt

# --- Bot with double-threat handling ---
NodeCacheEntry = namedtuple("NodeCacheEntry", ["depth","score"])

class Bot:
    def __init__(self, player, time_limit=0.88):
        self.player = player
        self.time_limit = time_limit
        self.tt = load_tt()
        self.start_time = None
        self.node_count = 0

    def time_exceeded(self):
        return (time.perf_counter() - self.start_time) > self.time_limit

    def save_cache(self):
        save_tt(self.tt)

    def detect_immediate_wins(self, board, player):
        wins = []
        candidates = generate_candidates(board, radius=2, max_candidates=BOARD_SIZE*BOARD_SIZE)
        for (r,c) in candidates:
            if is_win_after_move(board, r, c, player):
                wins.append((r,c))
        return wins

    # ----- New: detect if opponent will have double-threat if we play at (r,c)
    def opponent_can_create_double_after_our_move(self, board, my_r, my_c, opp, limit_check_moves=40):
        # simulate our move at (my_r,my_c), then see if opponent has any reply that creates >=2 immediate winning spots
        board[my_r][my_c] = self.player
        candidates = generate_candidates(board, radius=2, max_candidates=limit_check_moves)
        # order candidate opponent replies by heuristic for efficiency
        candidates.sort(key=lambda mv: -quick_heuristic_move_score(board, mv[0], mv[1], opp))
        for (orow, ocol) in candidates:
            if board[orow][ocol] != ' ':
                continue
            board[orow][ocol] = opp
            cnt = self.count_next_winning_moves_limited(board, opp, limit=100)
            board[orow][ocol] = ' '
            if cnt >= 2:
                board[my_r][my_c] = ' '
                return True
        board[my_r][my_c] = ' '
        return False

    def count_next_winning_moves_limited(self, board, player, limit=100):
        cnt = 0
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board[r][c] == ' ':
                    if is_win_after_move(board, r, c, player):
                        cnt += 1
                        if cnt >= limit:
                            return cnt
        return cnt

    # ----- New: detect cells that if opponent placed there now would create double-threat (we should block these)
    def detect_cells_that_create_double_if_opponent_placed(self, board, opp, limit_check=200):
        dangerous = []
        candidates = generate_candidates(board, radius=2, max_candidates=limit_check)
        for (r,c) in candidates:
            if board[r][c] != ' ':
                continue
            board[r][c] = opp
            cnt = self.count_next_winning_moves_limited(board, opp, limit=2)  # we only need to know >=2
            board[r][c] = ' '
            if cnt >= 2:
                dangerous.append((r,c))
        return dangerous

    def choose_move(self, board):
        self.start_time = time.perf_counter()
        self.node_count = 0
        me = self.player
        opp = 'O' if me == 'X' else 'X'

        # 0) quick list of candidates
        candidates = generate_candidates(board, radius=2, max_candidates=200)

        # 1) immediate win?
        for (r,c) in candidates:
            if is_win_after_move(board, r, c, me):
                return (r,c)

        # 2) opponent immediate win(s) -> must block
        opp_wins = self.detect_immediate_wins(board, opp)
        if opp_wins:
            # if multiple, prefer a blocking move that also helps us
            best = None; best_score = -10**9
            for (r,c) in opp_wins:
                if is_win_after_move(board, r, c, me):
                    return (r,c)
                s = quick_heuristic_move_score(board, r, c, me)
                if s > best_score:
                    best_score = s; best = (r,c)
            return best

        # 2b) detect critical cells that if opponent *placed now* would immediately create a double-threat (must block these)
        dangerous_by_opponent = self.detect_cells_that_create_double_if_opponent_placed(board, opp, limit_check=200)
        if dangerous_by_opponent:
            # prefer one that also improves our position
            best = None; best_score = -10**9
            for (r,c) in dangerous_by_opponent:
                s = quick_heuristic_move_score(board, r, c, me)
                if s > best_score:
                    best_score = s; best = (r,c)
            return best

        # 3) moves that create immediate double-threat for me (prefer)
        scored = []
        for (r,c) in candidates:
            s = quick_heuristic_move_score(board, r, c, me)
            scored.append(((r,c), s))
        scored.sort(key=lambda x: -x[1])
        topk = [p for p,_ in scored[:60]]
        for (r,c) in topk:
            board[r][c] = me
            cnt = self.count_next_winning_moves_limited(board, me, limit=100)
            board[r][c] = ' '
            if cnt >= 2:
                return (r,c)

        # 4) Avoid moves that allow opponent to create double-threat next (simulate limited)
        safe_moves = []
        unsafe_moves = []
        for (r,c) in candidates:
            if self.time_exceeded():
                break
            # if we play at (r,c), will opponent have a reply that creates double-threat?
            creates_for_opp = self.opponent_can_create_double_after_our_move(board, r, c, opp, limit_check_moves=40)
            if creates_for_opp:
                unsafe_moves.append((r,c))
            else:
                safe_moves.append((r,c))
        # if there are safe moves, restrict to them
        if safe_moves:
            candidates = safe_moves
        else:
            # If no safe moves (rare), keep original candidates but deprioritize unsafe ones
            candidates = candidates

        # 5) iterative deepening alpha-beta with move ordering
        best_move = None
        best_score = -10**18
        max_depth = 4
        try:
            depth = 1
            while depth <= max_depth:
                if self.time_exceeded():
                    break
                alpha = -10**18
                beta = 10**18
                root_moves = list(candidates)
                root_moves.sort(key=lambda mv: -quick_heuristic_move_score(board, mv[0], mv[1], me))
                root_moves = root_moves[:60]
                local_best = None
                local_best_score = -10**18
                for mv in root_moves:
                    if self.time_exceeded():
                        raise TimeoutError
                    r,c = mv
                    board[r][c] = me
                    score = -self.alphabeta(board, depth-1, -beta, -alpha, opp)
                    board[r][c] = ' '
                    if score > local_best_score:
                        local_best_score = score; local_best = mv
                    if score > alpha:
                        alpha = score
                    if alpha >= beta:
                        break
                if not self.time_exceeded() and local_best is not None:
                    best_move = local_best
                    best_score = local_best_score
                depth += 1
        except TimeoutError:
            pass
        except Exception:
            pass

        if best_move is None:
            if candidates:
                candidates.sort(key=lambda mv: -quick_heuristic_move_score(board, mv[0], mv[1], me))
                return candidates[0]
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    if board[r][c] == ' ':
                        return (r,c)
        return best_move

    def alphabeta(self, board, depth, alpha, beta, player_to_move):
        if self.time_exceeded():
            raise TimeoutError
        self.node_count += 1
        h = board_hash(board)
        key = (h, depth, player_to_move)
        if key in self.tt:
            return self.tt[key]

        # quick terminal check
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board[r][c] != ' ' and is_win_after_move(board, r, c, board[r][c]):
                    score = SCORES["FIVE"] if board[r][c] == self.player else -SCORES["FIVE"]
                    self.tt[key] = score
                    return score

        if depth == 0:
            val = evaluate(board, self.player)
            self.tt[key] = val
            return val

        moves = generate_candidates(board, radius=2, max_candidates=200)
        if not moves:
            return 0
        moves.sort(key=lambda mv: -quick_heuristic_move_score(board, mv[0], mv[1], self.player))
        max_branch = 12 if depth > 2 else 20
        moves = moves[:max_branch]

        best = -10**18
        for (r,c) in moves:
            if self.time_exceeded():
                raise TimeoutError
            board[r][c] = player_to_move
            val = -self.alphabeta(board, depth-1, -beta, -alpha, 'O' if player_to_move=='X' else 'X')
            board[r][c] = ' '
            if val > best:
                best = val
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        self.tt[key] = best
        return best

# --- Train helper (optional) ---
def train_bots(botA, botB, games=50, save_every=5):
    bots = [botA, botB]
    for g in range(games):
        board = [[' ']*BOARD_SIZE for _ in range(BOARD_SIZE)]
        current = 0 if g%2==0 else 1
        moves = 0
        winner = None
        while moves < BOARD_SIZE*BOARD_SIZE:
            bot = bots[current]
            r,c = bot.choose_move(board)
            if board[r][c] != ' ':
                found = False
                for rr in range(BOARD_SIZE):
                    for cc in range(BOARD_SIZE):
                        if board[rr][cc] == ' ':
                            r,c = rr,cc; found=True; break
                    if found: break
            board[r][c] = bot.player
            moves += 1
            if is_win_after_move(board, r, c, bot.player):
                winner = bot.player
                break
            current = 1-current
        botA.tt.update(botB.tt)
        if (g+1) % save_every == 0:
            save_tt(botA.tt)
    save_tt(botA.tt)
    return

# --- Public function ---
def get_move(board, current_player):
    """
    board: list[list] 15x15 with 'X','O' or ' '
    current_player: 'X' or 'O'
    returns (row, col)
    """
    bot = Bot(current_player, time_limit=0.88)
    mv = bot.choose_move(board)
    try:
        bot.save_cache()
    except Exception:
        pass
    return mv

# Quick demo (run as script)
if __name__ == "__main__":
    board = [[' ']*BOARD_SIZE for _ in range(BOARD_SIZE)]
    mv = get_move(board, 'X')
    print("Chosen:", mv)
