import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from collections import namedtuple, deque

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GameBoard class
class GameBoard:
    def __init__(self, size=4):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.score = 0
        self.add_new_tile()
        self.add_new_tile()

    def add_new_tile(self):
        empty_cells = [(i, j) for i in range(self.size) for j in range(self.size) if self.board[i][j] == 0]
        if empty_cells:
            i, j = random.choice(empty_cells)
            self.board[i][j] = 2 if random.random() < 0.9 else 4

    def move_tiles(self, direction):
        moved = False
        if direction == 'left':
            for i in range(self.size):
                new_row, merged = self.merge(self.compress(self.board[i]))
                if not np.array_equal(new_row, self.board[i]):
                    self.board[i] = new_row
                    moved = True
        elif direction == 'right':
            for i in range(self.size):
                new_row, merged = self.merge(self.compress(self.board[i][::-1]))
                new_row = new_row[::-1]
                if not np.array_equal(new_row, self.board[i]):
                    self.board[i] = new_row
                    moved = True
        elif direction == 'up':
            for j in range(self.size):
                column = self.board[:, j]
                new_column, merged = self.merge(self.compress(column))
                if not np.array_equal(new_column, column):
                    self.board[:, j] = new_column
                    moved = True
        elif direction == 'down':
            for j in range(self.size):
                column = self.board[:, j][::-1]
                new_column, merged = self.merge(self.compress(column))
                new_column = new_column[::-1]
                if not np.array_equal(new_column, self.board[:, j][::-1]):
                    self.board[:, j] = new_column
                    moved = True

        if moved:
            self.add_new_tile()
        return moved

    def compress(self, row):
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def merge(self, row):
        merged = False
        for j in range(self.size - 1):
            if row[j] != 0 and row[j] == row[j + 1]:
                row[j] *= 2
                row[j + 1] = 0
                self.score += row[j]
                merged = True
        return self.compress(row), merged

    def is_game_over(self):
        if np.any(self.board == 0):
            return False
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i][j] == self.board[i][j + 1] or self.board[j][i] == self.board[j + 1][i]:
                    return False
        return True

    def get_state(self):
        return self.board.flatten()

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_new_tile()
        self.add_new_tile()
        return self.get_state()

    def step(self, action):
        direction = ['left', 'right', 'up', 'down'][action]
        previous_board = self.board.copy()
        moved = self.move_tiles(direction)
        reward = self.score
        done = self.is_game_over()
        if not moved:
            reward = -10
        else:
            reward = self.score - np.sum(previous_board)
        #print(f"Action: {direction}, Moved: {moved}, Reward: {reward}")
        #self.display()
        return self.get_state(), reward, done, moved

    def display(self):
        for row in self.board:
            print('|', end='')
            for cell in row:
                if cell:
                    print(f'{cell:4}', end='|')
                else:
                    print(' ' * 4, end='|')
            print()
        print('-' * (self.size * 5 + 1))

# DQN neural network
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Select action based on epsilon-greedy strategy
def select_action(state, policy_net, epsilon):
    if random.random() < epsilon:
        return random.randint(0, 3)
    else:
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32, device=device)
            q_values = policy_net(state)
            return torch.argmax(q_values).item()

# Experience replay memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Training loop
def train_dqn():
    num_episodes = 500  # Increased the number of episodes
    memory = ReplayMemory(10000)
    batch_size = 128
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 0.995
    learning_rate = 0.001
    target_update = 10

    policy_net = DQN(input_size=16, output_size=4).to(device)
    target_net = DQN(input_size=16, output_size=4).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

    epsilon = epsilon_start

    for episode in range(num_episodes):
        game = GameBoard()
        state = game.reset()
        total_reward = 0
        while True:
            action = select_action(state, policy_net, epsilon)
            next_state, reward, done, moved = game.step(action)
            memory.push(state, action, next_state, reward)
            state = next_state
            total_reward += reward

            if done:
                break

            if len(memory) >= batch_size:
                transitions = memory.sample(batch_size)
                batch = Transition(*zip(*transitions))

                non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool, device=device)
                non_final_next_states = torch.tensor([s for s in batch.next_state if s is not None], dtype=torch.float32, device=device)
                state_batch = torch.tensor(batch.state, dtype=torch.float32, device=device)
                action_batch = torch.tensor(batch.action, dtype=torch.int64, device=device).unsqueeze(1)
                reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=device)

                state_action_values = policy_net(state_batch).gather(1, action_batch)
                next_state_values = torch.zeros(batch_size, device=device)
                next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

                expected_state_action_values = (next_state_values * gamma) + reward_batch

                loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                for param in policy_net.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()

        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        epsilon = max(epsilon_end, epsilon_decay * epsilon)
        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon}")

    torch.save(policy_net.state_dict(), "dqn_model.pth")
    return policy_net

# Playing the game using the trained DQN model
def play_dqn(policy_net):
    game = GameBoard()
    state = game.reset()
    total_reward = 0
    while True:
        action = select_action(state, policy_net, 0.5)  # Use the trained policy
        next_state, reward, done, moved = game.step(action)
        state = next_state
        total_reward += reward
        game.display()
        print(f"Score: {game.score}")
        if done:
            print("Game over!")
            break

if __name__ == "__main__":
    if os.path.exists("dqn_model.pth"):
        policy_net = DQN(input_size=16, output_size=4).to(device)
        policy_net.load_state_dict(torch.load("dqn_model.pth", map_location=device))
        play_dqn(policy_net)
    else:
        trained_policy_net = train_dqn()
        play_dqn(trained_policy_net)

