#!/usr/bin/env python3
import time
import json
import os
import random
from typing import List, Tuple, Optional

SIZE = 15
INF = 10**9
CACHE_FILE = "gomoku_cache.json"

# Load cache
if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, "r") as f:
            TT = json.load(f)
    except:
        TT = {}
else:
    TT = {}

# Zobrist hashing
ZOBRIST = [[[random.getrandbits(64) for _ in range(3)] for _ in range(SIZE)] for _ in range(SIZE)]
SYMBOL_TO_INDEX = {" ":0,"X":1,"O":2}

def zobrist_hash(board: List[List[str]]) -> str:
    h = 0
    for r in range(SIZE):
        for c in range(SIZE):
            h ^= ZOBRIST[r][c][SYMBOL_TO_INDEX[board[r][c]]]
    return str(h)

# Move generator
def generate_moves(board: List[List[str]]) -> List[Tuple[int,int]]:
    moves = set()
    radius = 2
    for r in range(SIZE):
        for c in range(SIZE):
            if board[r][c] != " ":
                for dr in range(-radius, radius+1):
                    for dc in range(-radius, radius+1):
                        nr,nc = r+dr,c+dc
                        if 0<=nr<SIZE and 0<=nc<SIZE and board[nr][nc]==" ":
                            moves.add((nr,nc))
    if not moves:
        moves = {(SIZE//2,SIZE//2)}
    return list(moves)

# Evaluate line
def pattern_score(line: List[str], player: str) -> int:
    opp = "O" if player=="X" else "X"
    s = "".join(line)
    score = 0
    # Win / lose
    if player*5 in s: score += 100000
    if opp*5 in s: score -= 100000
    # Open 4
    if player*4 in s and s.count(" ")>=1: score += 10000
    if opp*4 in s and s.count(" ")>=1: score -= 50000
    # Open 3
    if player*3 in s and s.count(" ")>=2: score += 1000
    if opp*3 in s and s.count(" ")>=2: score -= 20000
    # Open 2
    if player*2 in s and s.count(" ")>=3: score += 100
    if opp*2 in s and s.count(" ")>=3: score -= 100
    return score

# Count number of threats from a move
def count_threats(board: List[List[str]], player: str, r:int, c:int) -> int:
    """Return number of simultaneous open 3/4 created by placing at (r,c)"""
    threats = 0
    board[r][c] = player
    lines = []
    # Row
    if c-4>=0: lines.append([board[r][c-i] for i in range(5)])
    if c+4<SIZE: lines.append([board[r][c+i] for i in range(5)])
    # Col
    if r-4>=0: lines.append([board[r-i][c] for i in range(5)])
    if r+4<SIZE: lines.append([board[r+i][c] for i in range(5)])
    # Diag \
    if r-4>=0 and c-4>=0: lines.append([board[r-i][c-i] for i in range(5)])
    if r+4<SIZE and c+4<SIZE: lines.append([board[r+i][c+i] for i in range(5)])
    # Diag /
    if r-4>=0 and c+4<SIZE: lines.append([board[r-i][c+i] for i in range(5)])
    if r+4<SIZE and c-4>=0: lines.append([board[r+i][c-i] for i in range(5)])
    for line in lines:
        s = "".join(line)
        if player*3 in s and s.count(" ")>=2: threats += 1
        if player*4 in s and s.count(" ")>=1: threats += 2
    board[r][c] = " "
    return threats

# Evaluation with double-threat
def evaluate(board: List[List[str]], player: str) -> int:
    total = 0
    for r in range(SIZE):
        for c in range(SIZE):
            # Row
            if c+4<SIZE: total += pattern_score([board[r][c+i] for i in range(5)], player)
            # Col
            if r+4<SIZE: total += pattern_score([board[r+i][c] for i in range(5)], player)
            # Diagonal \
            if r+4<SIZE and c+4<SIZE: total += pattern_score([board[r+i][c+i] for i in range(5)], player)
            # Diagonal /
            if r+4<SIZE and c-4>=0: total += pattern_score([board[r+i][c-i] for i in range(5)], player)
    # Double-threat bonus
    for r,c in generate_moves(board):
        threats = count_threats(board, player, r, c)
        if threats>=2: total += 50000  # very high priority
    return total

# Immediate win/block including double-threat
def find_immediate_win_or_block(board: List[List[str]], player: str) -> Optional[Tuple[int,int]]:
    opp = "O" if player=="X" else "X"
    moves = generate_moves(board)
    # Player win
    for r,c in moves:
        board[r][c] = player
        if evaluate(board, player) >= 100000:
            board[r][c] = " "
            return (r,c)
        board[r][c] = " "
    # Opponent win → block
    for r,c in moves:
        board[r][c] = opp
        if evaluate(board, opp) >= 100000:
            board[r][c] = " "
            return (r,c)
        board[r][c] = " "
    # Double-threat own move first
    for r,c in moves:
        if count_threats(board, player, r, c) >= 2:
            return (r,c)
    # Opponent double-threat → block
    for r,c in moves:
        if count_threats(board, opp, r, c) >= 2:
            return (r,c)
    # Threat: open 3 /4 opponent
    for r,c in moves:
        board[r][c] = opp
        if evaluate(board, opp) >= 20000:
            board[r][c] = " "
            return (r,c)
        board[r][c] = " "
    return None

# Minimax + alpha-beta
def minimax(board, depth, alpha, beta, maximizing, player, start, limit) -> int:
    if time.time()-start>limit: return 0
    key = zobrist_hash(board)
    if key in TT:
        entry = TT[key]
        if entry.get("depth",0)>=depth and not entry.get("bad",False):
            return entry["score"]
    if depth==0: return evaluate(board, player)
    moves = generate_moves(board)
    if not moves: return 0
    opp = "O" if player=="X" else "X"
    best = -INF if maximizing else INF
    for r,c in moves:
        board_key = zobrist_hash(board)
        if board_key in TT and TT[board_key].get("bad",False) and TT[board_key].get("move")==(r,c): continue
        board[r][c] = player if maximizing else opp
        val = minimax(board, depth-1, -INF, INF, not maximizing, player, start, limit)
        board[r][c] = " "
        if maximizing:
            best = max(best,val)
            alpha = max(alpha,val)
        else:
            best = min(best,val)
            beta = min(beta,val)
        if beta<=alpha: break
    if time.time()-start<limit:
        TT[key] = {"score":best,"depth":depth,"bad":False}
    return best

# Mark bad move
def mark_bad_state(board: List[List[str]], move: Tuple[int,int]):
    key = zobrist_hash(board)
    TT[key] = {"move": move,"score": -INF,"depth":0,"bad":True}
    with open(CACHE_FILE,"w") as f: json.dump(TT,f)

# Main get_move
def get_move(board: List[List[str]], current_player: str, time_limit=4.0, max_depth_cap=6) -> Tuple[int,int]:
    start = time.time()
    key = zobrist_hash(board)
    # 1. Immediate win/block/double-threat
    imm = find_immediate_win_or_block(board, current_player)
    if imm: return imm
    # 2. Cache
    best_move = None
    best_score = -INF
    if key in TT and not TT[key].get("bad",False):
        entry = TT[key]
        if entry.get("score",-INF)>-5000:
            best_move = tuple(entry.get("move",(SIZE//2,SIZE//2)))
            best_score = entry["score"]
            start_depth = entry.get("depth",0)+1
        else: start_depth = 1
    else: start_depth = 1
    # 3. Iterative deepening
    for depth in range(start_depth, max_depth_cap+1):
        if time.time()-start>time_limit: break
        moves = generate_moves(board)
        random.shuffle(moves)
        for r,c in moves:
            if time.time()-start>time_limit: break
            board_key = zobrist_hash(board)
            if board_key in TT and TT[board_key].get("bad",False) and TT[board_key].get("move")==(r,c): continue
            board[r][c] = current_player
            val = minimax(board, depth-1, -INF, INF, False, current_player, start, time_limit)
            board[r][c] = " "
            if val>best_score:
                best_score = val
                best_move = (r,c)
        if best_move:
            TT[key] = {"move":best_move,"score":best_score,"depth":depth,"bad":False}
    if not best_move: best_move = (SIZE//2,SIZE//2)
    with open(CACHE_FILE,"w") as f: json.dump(TT,f)
    return best_move
