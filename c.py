import random, json, os, time
from typing import List, Tuple

SIZE = 15
WIN_LEN = 5
CACHE_FILE = "gomoku_cache.json"

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)

def opponent(player: str) -> str:
    return "O" if player == "X" else "X"

# -------------------------------------------------------
# Bot Gomoku
# -------------------------------------------------------
class Bot:
    def __init__(self, player: str):
        self.player = player
        self.cache = load_cache()

    def board_to_key(self, board: List[List[str]]) -> str:
        return "".join("".join(row) for row in board)

    def is_winner(self, board, player: str) -> bool:
        # kiểm tra thắng
        for r in range(SIZE):
            for c in range(SIZE):
                if board[r][c] != player:
                    continue
                for dr, dc in [(1,0),(0,1),(1,1),(1,-1)]:
                    cnt = 0
                    nr, nc = r, c
                    while 0 <= nr < SIZE and 0 <= nc < SIZE and board[nr][nc] == player:
                        cnt += 1
                        if cnt >= WIN_LEN:
                            return True
                        nr += dr
                        nc += dc
        return False

    def count_next_winning_moves(self, board, player: str) -> int:
        # đếm số ô nếu player đánh thì thắng ngay
        cnt = 0
        for r in range(SIZE):
            for c in range(SIZE):
                if board[r][c] != " ":
                    continue
                board[r][c] = player
                if self.is_winner(board, player):
                    cnt += 1
                board[r][c] = " "
        return cnt

    def heuristic(self, board, me: str) -> int:
        # hàm đánh giá cơ bản
        opp = opponent(me)
        if self.is_winner(board, me):
            return 10**6
        if self.is_winner(board, opp):
            return -10**6
        return random.randint(-10,10)

    def get_candidates(self, board) -> List[Tuple[int,int]]:
        candidates = set()
        for r in range(SIZE):
            for c in range(SIZE):
                if board[r][c] != " ":
                    for dr in range(-2,3):
                        for dc in range(-2,3):
                            nr, nc = r+dr, c+dc
                            if 0 <= nr < SIZE and 0 <= nc < SIZE and board[nr][nc] == " ":
                                candidates.add((nr,nc))
        if not candidates: # bàn trống
            return [(SIZE//2, SIZE//2)]
        return list(candidates)

    def alpha_beta(self, board, depth, alpha, beta, maximizing, me: str):
        opp = opponent(me)
        if depth == 0 or self.is_winner(board, me) or self.is_winner(board, opp):
            return self.heuristic(board, me)
        if maximizing:
            value = -10**9
            for r,c in self.get_candidates(board):
                board[r][c] = me
                score = self.alpha_beta(board, depth-1, alpha, beta, False, me)
                board[r][c] = " "
                value = max(value, score)
                alpha = max(alpha, score)
                if alpha >= beta:
                    break
            return value
        else:
            value = 10**9
            for r,c in self.get_candidates(board):
                board[r][c] = opp
                score = self.alpha_beta(board, depth-1, alpha, beta, True, me)
                board[r][c] = " "
                value = min(value, score)
                beta = min(beta, score)
                if beta <= alpha:
                    break
            return value

    def choose_move(self, board) -> Tuple[int,int]:
        me = self.player
        opp = opponent(me)

        key = self.board_to_key(board) + me
        if key in self.cache:
            return tuple(self.cache[key])

        candidates = self.get_candidates(board)

        # 1. thắng ngay
        for (r,c) in candidates:
            board[r][c] = me
            if self.is_winner(board, me):
                board[r][c] = " "
                self.cache[key] = (r,c); save_cache(self.cache)
                return (r,c)
            board[r][c] = " "

        # 2. chặn đối thủ thắng ngay
        for (r,c) in candidates:
            board[r][c] = opp
            if self.is_winner(board, opp):
                board[r][c] = " "
                self.cache[key] = (r,c); save_cache(self.cache)
                return (r,c)
            board[r][c] = " "

        # 3. tạo nước đôi
        for (r,c) in candidates:
            board[r][c] = me
            if self.count_next_winning_moves(board, me) >= 2:
                board[r][c] = " "
                self.cache[key] = (r,c); save_cache(self.cache)
                return (r,c)
            board[r][c] = " "

        # 4. chặn nước đôi đối thủ
        for (r,c) in candidates:
            board[r][c] = opp
            if self.count_next_winning_moves(board, opp) >= 2:
                board[r][c] = " "
                self.cache[key] = (r,c); save_cache(self.cache)
                return (r,c)
            board[r][c] = " "

        # 5. tránh rơi vào bẫy (lọc move nguy hiểm)
        safe = []
        for (r,c) in candidates:
            board[r][c] = me
            if self.count_next_winning_moves(board, opp) < 2:
                safe.append((r,c))
            board[r][c] = " "
        if safe:
            candidates = safe

        # 6. alpha-beta search
        best_score = -10**9
        best_move = random.choice(candidates)
        deadline = time.time() + 0.85
        for (r,c) in candidates:
            if time.time() > deadline:
                break
            board[r][c] = me
            score = self.alpha_beta(board, 2, -10**9, 10**9, False, me)
            board[r][c] = " "
            if score > best_score:
                best_score, best_move = score, (r,c)

        self.cache[key] = best_move
        save_cache(self.cache)
        return best_move

# -------------------------------------------------------
# API cho bên ngoài
# -------------------------------------------------------
def get_move(board: List[List[str]], current_player: str) -> Tuple[int,int]:
    bot = Bot(current_player)
    return bot.choose_move(board)

# -------------------------------------------------------
# Train 2 con bot để tăng cache
# -------------------------------------------------------
def train_bots(n_games=50):
    for g in range(n_games):
        board = [[" "]*SIZE for _ in range(SIZE)]
        botX, botO = Bot("X"), Bot("O")
        player = "X"
        moves = 0
        while moves < SIZE*SIZE:
            bot = botX if player=="X" else botO
            r,c = bot.choose_move(board)
            board[r][c] = player
            if bot.is_winner(board, player):
                break
            player = opponent(player)
            moves += 1
        print(f"Game {g+1} done, moves={moves}")
    print("Training finished, cache updated.")

# Example:
# train_bots(100)
