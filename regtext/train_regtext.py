import os
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim

class TextDataset(Dataset):
    def __init__(self, directory):
        self.data = []
        for file_name in os.listdir(directory):
            with open(os.path.join(directory, file_name), 'r') as file:
                self.data.append(file.read())
        self.data = ' '.join(self.data)
        self.chars = sorted(list(set(self.data)))
        self.char2idx = {ch: idx for idx, ch in enumerate(self.chars)}
        self.idx2char = {idx: ch for idx, ch in enumerate(self.chars)}
        self.data_indices = [self.char2idx[ch] for ch in self.data]

    def __len__(self):
        return len(self.data_indices) - 1

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data_indices[idx], dtype=torch.long),
            torch.tensor(self.data_indices[idx + 1], dtype=torch.long)
        )

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden
    
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)

# Hyperparameters
hidden_size = 128
num_epochs = 10
learning_rate = 0.001
batch_size = 1

# Load data
dataset = TextDataset('../data/text')
input_size = output_size = len(dataset.chars)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model, loss, optimizer
model = CharRNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    hidden = model.init_hidden()
    for i, (input_char, target_char) in enumerate(dataloader):
        input_char = torch.nn.functional.one_hot(input_char, num_classes=input_size).float().unsqueeze(0)
        target_char = target_char.unsqueeze(0)
        
        optimizer.zero_grad()
        output, hidden = model(input_char, hidden.detach())
        loss = criterion(output, target_char.view(-1))
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

# Save the model
model_save_path = 'char_rnn_model.pth'
torch.save(model.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')
