import time
import os
import json
from copy import deepcopy

BOARD_SIZE = 15
DATA_DIR = "data"
CACHE_FILE = f"{DATA_DIR}/trans_table.json"
BOOK_FILE = f"{DATA_DIR}/opening_book.json"
GAMES_DIR = f"{DATA_DIR}/games"

for d in [DATA_DIR, GAMES_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

trans_table = {}
opening_book = {}
if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, "r") as f:
            trans_table = json.load(f)
    except: pass
if os.path.exists(BOOK_FILE):
    try:
        with open(BOOK_FILE, "r") as f:
            opening_book = json.load(f)
    except: pass

def all_empty_cells(board):
    return [(r,c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if board[r][c]==' ']

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

def evaluate(board, player):
    opponent = 'O' if player=='X' else 'X'
    def count_patterns(line, target):
        s = 0
        l = ''.join(line)
        if target*5 in l: s += 100000
        if f' {target*4} ' in l: s += 1000
        if f' {target*3} ' in l: s += 100
        if f' {target*2} ' in l: s += 10
        return s
    score = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if c+5<=BOARD_SIZE:
                score += count_patterns(board[r][c:c+5], player)
                score -= count_patterns(board[r][c:c+5], opponent)
            if r+5<=BOARD_SIZE:
                col = [board[r+i][c] for i in range(5)]
                score += count_patterns(col, player)
                score -= count_patterns(col, opponent)
            if r+5<=BOARD_SIZE and c+5<=BOARD_SIZE:
                diag = [board[r+i][c+i] for i in range(5)]
                score += count_patterns(diag, player)
                score -= count_patterns(diag, opponent)
            if r+5<=BOARD_SIZE and c-4>=0:
                diag = [board[r+i][c-i] for i in range(5)]
                score += count_patterns(diag, player)
                score -= count_patterns(diag, opponent)
    return score

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
    threat_moves = []
    for r,c in moves:
        board[r][c] = player
        if evaluate(board, player) >= 100:
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
            if beta <= alpha: break
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
            if beta <= alpha: break
        trans_table[board_key] = min_eval
        return min_eval

def get_move(board, current_player):
    start_time = time.time()
    opponent = 'O' if current_player=='X' else 'X'

    # Nước thắng ngay
    for r,c in all_empty_cells(board):
        board[r][c] = current_player
        if check_win(board, current_player):
            board[r][c] = ' '
            return (r,c)
        board[r][c] = ' '

    # Chặn đối thủ thắng
    for r,c in all_empty_cells(board):
        board[r][c] = opponent
        if check_win(board, opponent):
            board[r][c] = ' '
            return (r,c)
        board[r][c] = ' '

    # Opening book
    board_key = ''.join(''.join(row) for row in board)
    if board_key in opening_book:
        return tuple(opening_book[board_key])

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
        if time.time() - start_time > 0.85:
            break

    # Lưu cache
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(trans_table, f)
    except: pass

    return best_move if best_move else (BOARD_SIZE//2, BOARD_SIZE//2)
