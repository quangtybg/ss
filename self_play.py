import json
import os
from copy import deepcopy
from bot_ultimate import get_move, BOARD_SIZE, check_win

DATA_DIR = "data"
GAMES_DIR = f"{DATA_DIR}/games"
BOOK_FILE = f"{DATA_DIR}/opening_book.json"
CACHE_FILE = f"{DATA_DIR}/trans_table.json"

for d in [DATA_DIR, GAMES_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

opening_book = {}
trans_table = {}
if os.path.exists(BOOK_FILE):
    try:
        with open(BOOK_FILE, "r") as f:
            opening_book = json.load(f)
    except: pass
if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, "r") as f:
            trans_table = json.load(f)
    except: pass

def board_to_key(board):
    return ''.join(''.join(row) for row in board)

def play_single_game(game_id):
    global opening_book
    board = [[' ']*BOARD_SIZE for _ in range(BOARD_SIZE)]
    moves_history = []
    current_player = 'X'
    while True:
        key = board_to_key(board)
        move = None
        if key in opening_book:
            move = tuple(opening_book[key])
        else:
            move = get_move(deepcopy(board), current_player)
        r,c = move
        board[r][c] = current_player
        moves_history.append((key, move, current_player))

        if check_win(board, current_player):
            winner = current_player
            break
        if all(board[r][c]!=' ' for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)):
            winner = None
            break

        current_player = 'O' if current_player=='X' else 'X'

    for idx, (bkey, move, player) in enumerate(moves_history[:5]):
        if winner == player and bkey not in opening_book:
            opening_book[bkey] = move

    with open(f"{GAMES_DIR}/game_{game_id}.json", "w") as f:
        json.dump({"moves": moves_history, "winner": winner}, f)

def self_play_total(total_games=10000, batch_size=500):
    batches = total_games // batch_size
    for b in range(batches):
        print(f"--- Batch {b+1}/{batches} ---")
        for i in range(batch_size):
            game_id = b*batch_size + i
            play_single_game(game_id)
        with open(BOOK_FILE, "w") as f:
            json.dump(opening_book, f)
        with open(CACHE_FILE, "w") as f:
            json.dump(trans_table, f)
        print(f"Batch {b+1} completed. Opening book size: {len(opening_book)}")

if __name__ == "__main__":
    self_play_total()
    print("Self-play 10,000 ván hoàn tất. Bot đã mạnh ngay từ đầu.")
