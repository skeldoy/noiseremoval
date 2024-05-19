import tensorflow as tf
import numpy as np

# Generate some sample data
np.random.seed(0)
X = np.random.randn(100, 1)
y = 3*X + 2 + 0.1*np.random.randn(100, 1)

# Define a simple linear regression model using Keras API
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# Compile the model with a loss function and optimizer
model.compile(optimizer=tf.optimizers.SGD(learning_rate=0.01), loss='mean_squared_error')

# Training the model
model.fit(X, y, epochs=1000, verbose=0)

# Evaluate the model
loss = model.evaluate(X, y)
print(f'Loss: {loss}')

# Testing the model
predicted = model.predict(X)
print(f'Predicted values: {predicted[:5]}')

