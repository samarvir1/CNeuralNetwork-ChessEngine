import os
import random
import numpy as np
import pygame
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F

# ================= CONFIG =================

BOARD_SIZE = 640
SQUARE_SIZE = BOARD_SIZE // 8

MODEL_PATH = "chess_net.pth"

SEARCH_DEPTH = 4
EPOCHS = 1500
LR = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WHITE = (240, 217, 181)
BROWN = (181, 136, 99)

# ================= MODEL =================

class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(12, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x))

# ================= ENCODING =================

PIECE_TO_PLANE = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

def board_to_tensor(board):
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            plane = PIECE_TO_PLANE[piece.piece_type]
            if piece.color == chess.BLACK:
                plane += 6
            r = 7 - chess.square_rank(sq)
            c = chess.square_file(sq)
            tensor[plane, r, c] = 1.0
    return tensor

# ================= EVALUATION =================

def material_score(board):
    values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    score = 0
    for p in values:
        score += len(board.pieces(p, chess.WHITE)) * values[p]
        score -= len(board.pieces(p, chess.BLACK)) * values[p]
    return score / 39.0

def evaluate(model, board):
    t = torch.tensor(board_to_tensor(board)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        nn_val = model(t).item()
    return 0.7 * nn_val + 0.3 * material_score(board)

# ================= SEARCH =================

def minimax(board, depth, alpha, beta, maximizing, model):
    if depth == 0 or board.is_game_over():
        return evaluate(model, board)

    if maximizing:
        best = -1e9
        for move in board.legal_moves:
            board.push(move)
            val = minimax(board, depth-1, alpha, beta, False, model)
            board.pop()
            best = max(best, val)
            alpha = max(alpha, val)
            if beta <= alpha:
                break
        return best
    else:
        best = 1e9
        for move in board.legal_moves:
            board.push(move)
            val = minimax(board, depth-1, alpha, beta, True, model)
            board.pop()
            best = min(best, val)
            beta = min(beta, val)
            if beta <= alpha:
                break
        return best

def best_move(board, model):
    best_val = -1e9 if board.turn else 1e9
    choice = None

    for move in board.legal_moves:
        board.push(move)
        val = minimax(board, SEARCH_DEPTH-1, -1e9, 1e9, not board.turn, model)
        board.pop()

        if board.turn and val > best_val:
            best_val = val
            choice = move
        elif not board.turn and val < best_val:
            best_val = val
            choice = move

    return choice

# ================= TRAINING =================

def train_model():
    print("Training (stabilized)...")

    model = ChessNet().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=300, gamma=0.5)
    loss_fn = nn.MSELoss()

    for epoch in range(EPOCHS):
        board = chess.Board()
        states = []

        while not board.is_game_over():
            states.append(board_to_tensor(board))
            board.push(random.choice(list(board.legal_moves)))

        result = board.result()
        game_value = 1.0 if result == "1-0" else -1.0 if result == "0-1" else 0.0

        total_loss = 0.0
        n = len(states)

        for i, s in enumerate(states):
            discount = (i + 1) / n
            target_value = game_value * discount * 0.8

            target = torch.tensor([[target_value]], device=DEVICE)
            s = torch.tensor(s).unsqueeze(0).to(DEVICE)

            pred = model(s)
            loss = loss_fn(pred, target)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()

        scheduler.step()

        avg_loss = total_loss / max(1, n)
        lr_now = opt.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {avg_loss:.6f} | "
            f"LR: {lr_now:.6f}"
        )

    torch.save(model.state_dict(), MODEL_PATH)
    print("Model saved.")

# ================= GUI =================

def load_piece_images():
    images = {}
    for color in ['w', 'b']:
        for p in ['p', 'n', 'b', 'r', 'q', 'k']:
            key = color + p
            img = pygame.image.load(f"assets/{key}.png")
            img = pygame.transform.smoothscale(img, (SQUARE_SIZE, SQUARE_SIZE))
            images[key] = img
    return images

def draw_board(screen, board, piece_images):
    for r in range(8):
        for c in range(8):
            color = WHITE if (r + c) % 2 == 0 else BROWN
            pygame.draw.rect(
                screen, color,
                pygame.Rect(c*SQUARE_SIZE, r*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            )

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            c = chess.square_file(sq)
            r = 7 - chess.square_rank(sq)
            key = ('w' if piece.color else 'b') + piece.symbol().lower()
            screen.blit(piece_images[key], (c*SQUARE_SIZE, r*SQUARE_SIZE))

def run_game():
    if not os.path.exists(MODEL_PATH):
        train_model()

    model = ChessNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    pygame.init()
    screen = pygame.display.set_mode((BOARD_SIZE, BOARD_SIZE))
    pygame.display.set_caption("Neural Chess")

    piece_images = load_piece_images()

    board = chess.Board()
    selected = None
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN and board.turn:
                x, y = pygame.mouse.get_pos()
                c = x // SQUARE_SIZE
                r = 7 - (y // SQUARE_SIZE)
                sq = chess.square(c, r)

                if selected is None:
                    selected = sq
                else:
                    move = chess.Move(selected, sq)
                    if move in board.legal_moves:
                        board.push(move)
                    selected = None

        if not board.turn and not board.is_game_over():
            move = best_move(board, model)
            if move:
                board.push(move)

        draw_board(screen, board, piece_images)
        pygame.display.flip()

    pygame.quit()

# ================= MAIN =================

if __name__ == "__main__":
    run_game()
