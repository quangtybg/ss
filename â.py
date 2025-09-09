import time
import os
import json
from copy import deepcopy

BOARD_SIZE = 15
CACHE_FILE = "data/trans_table_tss.json"

# --- Cache nước đi ---
trans_table = {}
if not os.path.exists("data"):
    os.makedirs("data")
if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, "r") as f:
            trans_table = json.load(f)
    except:
        trans_table = {}

# --- Hàm cơ bản ---
def all_empty_cells(board):
    return [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if board[r][c]==' ']

def check_win(board, player):
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if c+4<BOARD_SIZE and all(board[r][c+i]==player for i in range(5)):
                return True
            if r+4<BOARD_SIZE and all(board[r+i][c]==player for i in range(5)):
                return True
            if r+4<BOARD_SIZE and c+4<BOARD_SIZE and all(board[r+i][c+i]==player for i in range(5)):
                return True
            if r+4<BOARD_SIZE and c-4>=0 and all(board[r+i][c-i]==player for i in range(5)):
                return True
    return False

def generate_moves(board, distance=2):
    moves = set()
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != ' ':
                for dr in range(-distance, distance+1):
                    for dc in range(-distance, distance+1):
                        nr, nc = r+dr, c+dc
                        if 0<=nr<BOARD_SIZE and 0<=nc<BOARD_SIZE and board[nr][nc]==' ':
                            moves.add((nr,nc))
    if not moves:
        moves.add((BOARD_SIZE//2, BOARD_SIZE//2))
    return list(moves)

# --- Heuristic đánh giá nâng cao ---
def evaluate(board, player):
    opponent = 'O' if player=='X' else 'X'
    def count_patterns(line, target):
        s = 0
        l = ''.join(line)
        # ưu tiên open-4, double-three
        if target*5 in l:
            s += 100000
        if f' {target*4} ' in l:
            s += 1000
        if f' {target*3} ' in l:
            s += 100
        if f' {target*2} ' in l:
            s += 10
        return s

    score = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            # ngang
            if c+5<=BOARD_SIZE:
                score += count_patterns(board[r][c:c+5], player)
                score -= count_patterns(board[r][c:c+5], opponent)
            # dọc
            if r+5<=BOARD_SIZE:
                col = [board[r+i][c] for i in range(5)]
                score += count_patterns(col, player)
                score -= count_patterns(col, opponent)
            # chéo \
            if r+5<=BOARD_SIZE and c+5<=BOARD_SIZE:
                diag = [board[r+i][c+i] for i in range(5)]
                score += count_patterns(diag, player)
                score -= count_patterns(diag, opponent)
            # chéo /
            if r+5<=BOARD_SIZE and c-4>=0:
                diag = [board[r+i][c-i] for i in range(5)]
                score += count_patterns(diag, player)
                score -= count_patterns(diag, opponent)
    return score

# --- Minimax + Alpha-Beta + TSS ---
def alphabeta_tss(board, depth, alpha, beta, maximizing, player, start_time, time_limit=0.9):
    if time.time() - start_time > time_limit:
        return evaluate(board, player)
    board_key = ''.join(''.join(row) for row in board)
    if board_key in trans_table:
        return trans_table[board_key]

    opponent = 'O' if player=='X' else 'X'
    if depth==0:
        score = evaluate(board, player)
        trans_table[board_key] = score
        return score

    moves = generate_moves(board)
    # Threat Space Search: chỉ giữ moves tạo threat (4 hoặc 3)
    threat_moves = []
    for r,c in moves:
        board[r][c] = player
        if evaluate(board, player) >= 100:  # 4 mở hoặc 3 mở
            threat_moves.append((r,c))
        board[r][c] = ' '
    if threat_moves:
        moves = threat_moves

    if maximizing:
        max_eval = -float('inf')
        for r,c in moves:
            board[r][c] = player
            if check_win(board, player):
                board[r][c] = ' '
                return 100000
            eval = alphabeta_tss(board, depth-1, alpha, beta, False, player, start_time, time_limit)
            board[r][c] = ' '
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        trans_table[board_key] = max_eval
        return max_eval
    else:
        min_eval = float('inf')
        for r,c in moves:
            board[r][c] = opponent
            if check_win(board, opponent):
                board[r][c] = ' '
                return -100000
            eval = alphabeta_tss(board, depth-1, alpha, beta, True, player, start_time, time_limit)
            board[r][c] = ' '
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        trans_table[board_key] = min_eval
        return min_eval

# --- Iterative Deepening ---
def get_move(board, current_player):
    start_time = time.time()
    opponent = 'O' if current_player=='X' else 'X'

    # 1. Nước thắng ngay
    for r,c in all_empty_cells(board):
        board[r][c] = current_player
        if check_win(board, current_player):
            board[r][c] = ' '
            return (r,c)
        board[r][c] = ' '

    # 2. Chặn đối thủ thắng
    for r,c in all_empty_cells(board):
        board[r][c] = opponent
        if check_win(board, opponent):
            board[r][c] = ' '
            return (r,c)
        board[r][c] = ' '

    candidates = generate_moves(board)
    best_move = None
    best_score = -float('inf')

    depth = 1
    while True:
        for r,c in candidates:
            board[r][c] = current_player
            score = alphabeta_tss(board, depth, -float('inf'), float('inf'), False, current_player, start_time)
            board[r][c] = ' '
            if score > best_score:
                best_score = score
                best_move = (r,c)
        depth += 1
        if time.time() - start_time > 0.85:  # an toàn dưới 0.9s
            break

    # Lưu cache
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(trans_table, f)
    except:
        pass

    if best_move is None:
        return (BOARD_SIZE//2, BOARD_SIZE//2)
    return best_move

# --- Ví dụ ---
if __name__ == "__main__":
    board = [[' ']*BOARD_SIZE for _ in range(BOARD_SIZE)]
    move = get_move(board, 'X')
    print("X đánh:", move)
