import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the data
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')

# Separate labels
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

# One-hot encode categorical columns
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
dftrain = pd.get_dummies(dftrain, columns=CATEGORICAL_COLUMNS)
dfeval = pd.get_dummies(dfeval, columns=CATEGORICAL_COLUMNS)

# Align columns (ensures dfeval has same columns as dftrain)
dfeval = dfeval.reindex(columns=dftrain.columns, fill_value=0)

# Normalize numeric columns
scaler = StandardScaler()
X_train = scaler.fit_transform(dftrain)
X_eval = scaler.transform(dfeval)

# Convert to TensorFlow datasets
def make_tf_dataset(X, y, num_epochs=10, shuffle=True, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size).repeat(num_epochs)
    return ds

train_ds = make_tf_dataset(X_train, y_train)
eval_ds = make_tf_dataset(X_eval, y_eval, num_epochs=1, shuffle=False)

# Build a simple Keras model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(train_ds, epochs=10, steps_per_epoch=len(X_train) // 32)

# Evaluate
loss, accuracy = model.evaluate(eval_ds)
print("Accuracy:", accuracy)

##Get predicted probabilities
# Get predicted probabilities
pred_probs = model.predict(eval_ds)

# Convert probabilities to binary classes
pred_classes = (pred_probs > 0.5).astype("int32")

# Now you can analyze or print:

for i in range(10):
    print(f"Passenger {i+1}: Probability of survival = {pred_probs[i][0]:.4f}, Predicted class = {pred_classes[i][0]} , Actual Result = {y_eval.loc[i]}")
