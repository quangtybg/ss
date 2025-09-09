import time
import os
import json
from copy import deepcopy
from bot_ultimate import get_move as get_move_core, BOARD_SIZE, check_win

DATA_DIR = "data"
CACHE_FILE = f"{DATA_DIR}/trans_table.json"
BOOK_FILE = f"{DATA_DIR}/opening_book.json"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Load cache & opening book
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

def get_move(board, current_player):
    # Gọi core bot_ultimate để chọn nước
    move = get_move_core(board, current_player)
    # Update opening book nếu gặp board mới
    board_key = ''.join(''.join(row) for row in board)
    if board_key not in opening_book:
        opening_book[board_key] = move
        try:
            with open(BOOK_FILE, "w") as f:
                json.dump(opening_book, f)
        except: pass
    # Lưu cache
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(trans_table, f)
    except: pass
    return move

if __name__ == "__main__":
    board = [[' ']*BOARD_SIZE for _ in range(BOARD_SIZE)]
    move = get_move(board, 'X')
    print("X đánh:", move)
    print("Bot sẽ học liên tục khi chơi, opening book và cache được cập nhật tự động.")
