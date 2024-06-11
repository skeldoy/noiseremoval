import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os

class CharGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(CharGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden):
        out, hidden = self.gru(x, hidden)
        out = self.fc(out.reshape(-1, self.hidden_size))
        return out, hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

class TextDataset(Dataset):
    def __init__(self, directory, seq_length):
        self.data = []
        for file_name in os.listdir(directory):
            with open(os.path.join(directory, file_name), 'r') as file:
                self.data.append(file.read())
        self.data = ' '.join(self.data)
        self.chars = sorted(list(set(self.data)))
        self.char2idx = {ch: idx for idx, ch in enumerate(self.chars)}
        self.idx2char = {idx: ch for idx, ch in enumerate(self.chars)}
        self.data_indices = [self.char2idx[ch] for ch in self.data]
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data_indices) - self.seq_length

    def __getitem__(self, idx):
        seq_x = self.data_indices[idx:idx+self.seq_length]
        seq_y = self.data_indices[idx+1:idx+self.seq_length+1]
        return torch.tensor(seq_x), torch.tensor(seq_y)

def train_model(model, dataset, epochs=20, batch_size=64, lr=0.001, seq_length=100, model_path='char_gru_model.pth'):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for i, (inputs, targets) in enumerate(dataloader):
            batch_size = inputs.size(0)
            hidden = model.init_hidden(batch_size)
            inputs = torch.nn.functional.one_hot(inputs, num_classes=len(dataset.chars)).float()
            targets = targets.view(-1)
            hidden = hidden.detach()
            optimizer.zero_grad()

            outputs, hidden = model(inputs, hidden)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    dataset = TextDataset('../data/text', seq_length=100)
    input_size = output_size = len(dataset.chars)
    hidden_size = 256
    num_layers = 2

    model = CharGRU(input_size, hidden_size, output_size, num_layers)
    train_model(model, dataset, epochs=20, batch_size=64, lr=0.001, seq_length=100, model_path='char_gru_model.pth')

