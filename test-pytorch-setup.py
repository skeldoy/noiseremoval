import torch
import torch.nn as nn
import torch.optim as optim

# Generate some sample data
torch.manual_seed(0)
X = torch.randn(100, 1)
y = 3*X + 2 + 0.1*torch.randn(100, 1)

# Define a simple linear regression model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):
    model.train()
    
    # Forward pass
    predictions = model(X)
    loss = criterion(predictions, y)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Testing the model
model.eval()
with torch.no_grad():
    predicted = model(X)
    print(f'Predicted values: {predicted[:5]}')

