import json
import os
from copy import deepcopy
from bot_ultimate import get_move, BOARD_SIZE, check_win, generate_moves

DATA_DIR = "data"
GAMES_DIR = f"{DATA_DIR}/games"
BOOK_FILE = f"{DATA_DIR}/opening_book.json"

if not os.path.exists(GAMES_DIR):
    os.makedirs(GAMES_DIR)

opening_book = {}
if os.path.exists(BOOK_FILE):
    try:
        with open(BOOK_FILE, "r") as f:
            opening_book = json.load(f)
    except: pass

def board_to_key(board):
    return ''.join(''.join(row) for row in board)

def self_play(num_games=500):
    global opening_book
    for g in range(num_games):
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

        with open(f"{GAMES_DIR}/game_{g}.json", "w") as f:
            json.dump({"moves": moves_history, "winner": winner}, f)

        if (g+1) % 50 == 0:
            print(f"{g+1}/{num_games} ván self-play xong...")

    with open(BOOK_FILE, "w") as f:
        json.dump(opening_book, f)
    print(f"Self-play nâng cao hoàn tất. Opening book size: {len(opening_book)}")

if __name__ == "__main__":
    self_play(500)
