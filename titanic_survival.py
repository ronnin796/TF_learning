import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the Titanic training and evaluation datasets from URLs
# The datasets contain features like sex, class, and survival status for passengers
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')

# Separate the target variable ('survived') from the feature data
# y_train and y_eval contain the labels (0 = did not survive, 1 = survived)
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

# Define categorical columns to be one-hot encoded
# These columns contain non-numeric data that will be converted to binary columns
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']

# One-hot encode categorical columns to convert them into numeric format
# This creates new binary columns for each category (e.g., 'sex_male', 'sex_female')
dftrain = pd.get_dummies(dftrain, columns=CATEGORICAL_COLUMNS)
dfeval = pd.get_dummies(dfeval, columns=CATEGORICAL_COLUMNS)

# Align the evaluation dataset columns with the training dataset
# Ensures dfeval has the same columns as dftrain, filling missing columns with 0
dfeval = dfeval.reindex(columns=dftrain.columns, fill_value=0)

# Normalize numeric columns using StandardScaler
# This scales features to have mean=0 and variance=1 for better model performance
scaler = StandardScaler()
X_train = scaler.fit_transform(dftrain)  # Fit and transform the training data
X_eval = scaler.transform(dfeval)        # Transform the evaluation data using the same scaler

# Function to create a TensorFlow dataset from features and labels
# Converts numpy arrays to a batched, shuffled, and repeated dataset for training/evaluation
def make_tf_dataset(X, y, num_epochs=10, shuffle=True, batch_size=32):
    # Create a dataset from feature (X) and label (y) tensors
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        # Shuffle the dataset to randomize the order of samples (buffer_size=1000 for efficiency)
        ds = ds.shuffle(buffer_size=1000)
    # Batch the dataset (group into batches of size 32) and repeat for the specified number of epochs
    ds = ds.batch(batch_size).repeat(num_epochs)
    return ds

# Create TensorFlow datasets for training and evaluation
# Training dataset: shuffled, batched, and repeated for 10 epochs
# Evaluation dataset: not shuffled, batched, and run for 1 epoch
train_ds = make_tf_dataset(X_train, y_train)
eval_ds = make_tf_dataset(X_eval, y_eval, num_epochs=1, shuffle=False)

# Build a simple Keras neural network model
# The model has an input layer, one hidden layer (16 neurons, ReLU activation), and an output layer (sigmoid for binary classification)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),  # Input layer with shape equal to number of features
    tf.keras.layers.Dense(16, activation='relu'),      # Hidden layer with 16 neurons and ReLU activation
    tf.keras.layers.Dense(1, activation='sigmoid')     # Output layer with 1 neuron and sigmoid activation for binary classification
])

# Compile the model
# Uses Adam optimizer, binary cross-entropy loss (suitable for binary classification), and tracks accuracy
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model on the training dataset
# Runs for 10 epochs with steps per epoch calculated based on dataset size and batch size (32)
model.fit(train_ds, epochs=10, steps_per_epoch=len(X_train) // 32)

# Evaluate the model on the evaluation dataset
# Computes the loss and accuracy on the test set
loss, accuracy = model.evaluate(eval_ds)
print("Accuracy:", accuracy)

# Get predicted probabilities for the evaluation dataset
# Outputs probabilities of survival (values between 0 and 1)
pred_probs = model.predict(eval_ds)

# Convert probabilities to binary classes
# Probabilities > 0.5 are classified as 1 (survived), otherwise 0 (did not survive)
pred_classes = (pred_probs > 0.5).astype("int32")

# Print predictions and actual results for the first 10 passengers
# Shows the predicted probability, predicted class, and actual survival outcome
for i in range(10):
    print(f"Passenger {i+1}: Probability of survival = {pred_probs[i][0]:.4f}, Predicted class = {pred_classes[i][0]} , Actual Result = {y_eval.loc[i]}")