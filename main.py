#AI Tic-Tac-Toe with DQN and Model Persistence
import os
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

MODEL_PATH = "tictactoe_dqn.pth"

# Game Environment
class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros(9, dtype=np.int8)
        self.done = False
        return self.board.copy()

    def available_actions(self):
        return [i for i in range(9) if self.board[i] == 0]

    def check_winner(self):
        wins = [
            (0,1,2),(3,4,5),(6,7,8),
            (0,3,6),(1,4,7),(2,5,8),
            (0,4,8),(2,4,6)
        ]
        for a,b,c in wins:
            if self.board[a] == self.board[b] == self.board[c] != 0:
                return self.board[a]
        if 0 not in self.board:
            return 0
        return None

    def step(self, action, player):
        if self.board[action] != 0:
            return self.board.copy(), -1.0, True

        self.board[action] = player
        result = self.check_winner()

        if result is not None:
            if result == player:
                return self.board.copy(), 1.0, True
            elif result == 0:
                return self.board.copy(), 0.5, True
            else:
                return self.board.copy(), -1.0, True

        return self.board.copy(), 0.0, False

# Tiny Neural Network for DQN
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 9)
        )

    def forward(self, x):
        return self.net(x)

# Replay Buffer (AI's Memory)
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)

# Training Function (Saves model file after training)
def train_and_save():
    print("Training model for the first time...")
    env = TicTacToe()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    buffer = ReplayBuffer()

    GAMMA = 0.99
    BATCH_SIZE = 64
    EPSILON = 1.0
    EPSILON_MIN = 0.05
    EPSILON_DECAY = 0.995

    for episode in range(6000):
        state = env.reset()
        done = False

        while not done:
            if random.random() < EPSILON:
                action = random.choice(env.available_actions())
            else:
                with torch.no_grad():
                    q = policy_net(torch.tensor(state, dtype=torch.float32).to(device)).cpu().numpy()
                    for i in range(9):
                        if state[i] != 0:
                            q[i] = -1e9
                    action = int(np.argmax(q))

            next_state, reward, done = env.step(action, 1)

            if not done:
                opp_action = random.choice(env.available_actions())
                next_state, opp_reward, done = env.step(opp_action, -1)
                if done and opp_reward == 1.0:
                    reward = -1.0

            buffer.push(state, action, reward, next_state, done)
            state = next_state

            if len(buffer) >= BATCH_SIZE:
                states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)
                states, actions = states.to(device), actions.to(device)
                rewards, next_states = rewards.to(device), next_states.to(device)
                dones = dones.to(device)

                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    target = rewards + GAMMA * target_net(next_states).max(1)[0] * (1 - dones)

                loss = nn.MSELoss()(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

    torch.save(policy_net.state_dict(), MODEL_PATH)
    print("Training complete. Model saved.")

# Play Mode
def print_board(board):
    symbols = {1: "X", -1: "O", 0: " "}
    c = [symbols[v] for v in board]
    print("\n")
    print(f" {c[0]} | {c[1]} | {c[2]} ")
    print("---+---+---")
    print(f" {c[3]} | {c[4]} | {c[5]} ")
    print("---+---+---")
    print(f" {c[6]} | {c[7]} | {c[8]} ")
    print("\n")


def play():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DQN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    env = TicTacToe()
    board = env.reset()
    done = False

    print("You are O. AI is X.")
    print("Choose positions using numbers 0-8:")
    print(" 0 | 1 | 2 ")
    print("---+---+---")
    print(" 3 | 4 | 5 ")
    print("---+---+---")
    print(" 6 | 7 | 8 \n")

    while not done:
        print_board(board)
        while True:
            try:
                move = int(input("Your move (0-8): "))
                if move in env.available_actions():
                    break
                print("That position is already taken. Try again.")
            except ValueError:
                print("Please enter a number between 0 and 8.")

        board, _, done = env.step(move, -1)
        if done:
            break
        with torch.no_grad():
            q = model(torch.tensor(board, dtype=torch.float32).to(device)).cpu().numpy()
            for i in range(9):
                if board[i] != 0:
                    q[i] = -1e9
            ai_move = int(np.argmax(q))

        board, _, done = env.step(ai_move, 1)

    print_board(board)
    result = env.check_winner()
    if result == 1:
        print("AI wins!")
    elif result == -1:
        print("You win!")
    else:
        print("It's a draw!")

# Accesses model file; if not found, trains and saves it.
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        train_and_save()
    play()