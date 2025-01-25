# CNN-LSTM-Model

## Project: CNN+LSTM Model for Autonomous Vehicle Motion Prediction
This project utilizes a combination of **CNN** and **LSTM** to predict vehicle trajectories. The dataset used is from **Kaggle Lyft Motion Prediction Dataset**.

## Project Structure
```
📁 CNN-LSTM-Model/
│── 📄 Main Notebook: cnn_lstm_motion_prediction.py
│── 📄 Project Documentation: README.md
│── 📁 Processed Data (if needed)
│── 📄 Required Libraries: requirements.txt
│── 📄 dataset.csv (Dataset for training)
```

## Why Use LSTM?
LSTMs (Long Short-Term Memory networks) are specifically designed for sequence prediction tasks. In this project, LSTM is used because:

- **Temporal Dependency:** Vehicle motion is sequential, and LSTM can capture long-term dependencies in trajectory data.
- **Memory Retention:** Unlike traditional RNNs, LSTMs avoid vanishing gradient problems, making them more effective in handling long sequences.
- **Predicting Future States:** LSTM learns from historical data and provides more accurate motion trajectory predictions compared to feedforward networks.
- **Handling Noise in Data:** GPS and motion data often contain fluctuations; LSTM helps smooth out predictions over time.

## Model Features
- Utilizes **CNN** for spatial-temporal feature extraction
- Uses **LSTM** for motion trajectory modeling
- Preprocesses GPS data
- Evaluates and analyzes model errors

## Model Architecture
```python
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(sequence_length, len(features))),
    BatchNormalization(),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    LSTM(32),
    Dense(64, activation="relu"),
    Dense(2, activation="linear")  # Future X and Y output
])
```

**Optimizer:** `Adam (learning_rate=0.0005)`  
**Loss Function:** `MAE (Mean Absolute Error)`

## Installation and Execution
### Install Required Libraries
```sh
pip install -r requirements.txt
```

### Run the Script
```sh
python cnn_lstm_motion_prediction.py
```

## Required Libraries
- TensorFlow
- Keras
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Model Accuracy Issues and Error Analysis
- Model accuracy was moderate
- Some errors were high, but one was acceptable relative to output range
- **MAE** was reasonable for data scale, but **MSE** was high in some cases

## How to Improve Model Accuracy?
### Optimizing Input Data
- **Standardizing data** using `StandardScaler`
- **Selecting key features** to reduce model noise

### Model Optimization
- Increase the number of neurons in **LSTM**
- Use **Dropout(0.3)** to prevent overfitting
- Change activation function to **LeakyReLU**

### Better Training Settings
- Increase **Epochs** and reduce learning rate:
```python
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
```
- Use **Early Stopping** to prevent overfitting:
```python
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
```

### Using Advanced Models
Transformer-based models like **Attention Mechanism** can be used instead of **CNN+LSTM**.

## Sample Model Output (Before Optimization)
The model predicts the following trajectory points:
```python
Predicted Trajectory:
X: 12.3, Y: 8.9
X: 13.5, Y: 9.4
X: 14.1, Y: 10.2
X: 15.0, Y: 10.8
```
**Result:** The model had moderate accuracy but showed deviation in some points. The predictions were close to actual values in some cases but needed further optimization.
