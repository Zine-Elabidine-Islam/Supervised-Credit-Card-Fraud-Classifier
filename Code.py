import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the datasets
train_df = pd.read_excel(r'C:\Users\zinou\OneDrive\Bureau\Machine Learning\Advaned Learning Algorithms\Supervised_Training_Set.xlsx')
cv_df = pd.read_excel(r'C:\Users\zinou\OneDrive\Bureau\Machine Learning\Advaned Learning Algorithms\Supervised_CV_Set.xlsx')

features = ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long']

X_train = train_df[features]
y_train = train_df['is_fraud']

X_val = cv_df[features]
y_val = cv_df['is_fraud']

# Scale the data properly
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Define the neural network architecture
model = Sequential([
    Dense(units=16, activation='relu', input_shape=(6,)),    
    Dense(units=8, activation='relu'),
    Dense(units=1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss=BinaryCrossentropy(), metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train_scaled, y_train,
    epochs=30,
    batch_size=8,
    validation_data=(X_val_scaled, y_val)
)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Bias-Variance Diagnosis')
plt.show()