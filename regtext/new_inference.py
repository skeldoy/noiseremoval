import torch
import torch.nn as nn
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
        out = self.fc(out[:, -1, :])
        return out, hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

# Load the trained model
def load_model(model_path, input_size, hidden_size, output_size, num_layers=2):
    model = CharGRU(input_size, hidden_size, output_size, num_layers)
    model.load_state_dict(torch.load(model_path))
    return model

# Generate text using the trained model
def generate_text(model, start_text='The', max_length=100, temperature=0.8):
    model.eval()
    with torch.no_grad():
        input_indices = [dataset.char2idx[ch] for ch in start_text]
        input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0)
        hidden = model.init_hidden(input_tensor.size(0))

        output_text = start_text

        for _ in range(max_length):
            input_tensor = torch.nn.functional.one_hot(input_tensor, num_classes=len(dataset.chars)).float()
            output, hidden = model(input_tensor, hidden)
            
            # Scale the logits by temperature
            output_dist = torch.softmax(output.view(-1) / temperature, dim=0)
            top_char_idx = torch.multinomial(output_dist, 1)[0]
            
            input_tensor = torch.tensor([[top_char_idx]], dtype=torch.long)
            next_char = dataset.idx2char[top_char_idx.item()]
            output_text += next_char

            if next_char == '.':
                break

    return output_text

# Load data
class TextDataset:
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

# Main function for inference
def main():
    # Parameters
    model_path = 'char_rnn_model.pth'  # Ensure the path matches your saved model
    input_size = output_size = len(dataset.chars)
    hidden_size = 256  # Increased hidden size for more capacity
    num_layers = 2  # Use more layers for better learning

    # Load the trained model
    model = load_model(model_path, input_size, hidden_size, output_size, num_layers)

    # Take input from the user
    start_text = input("Enter a starting sentence: ")

    # Generate text
    generated_text = generate_text(model, start_text=start_text, temperature=0.8)
    print("Generated Text:")
    print(generated_text)

if __name__ == "__main__":
    # Load data
    dataset = TextDataset('../data/text')
    main()
