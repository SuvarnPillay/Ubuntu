import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from tensorflow import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score

def create_model(optimizer='adam', activation='relu', kernel_regularizer=None):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(784, activation=activation, input_shape=(X_train.shape[1],), kernel_regularizer=keras.regularizers.l2(0.04)))
    model.add(keras.layers.Dense(256, activation=activation))
    model.add(keras.layers.Dense(256, activation=activation))
    model.add(keras.layers.Dense(num_classes, activation='softmax')) 
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Read the data from the files
data = pd.read_csv('dataMNIST.txt', header=None)
labels = pd.read_csv('labelsMNIST.txt', header=None)

# Split the data into features and target
X = data.iloc[:, :-1]  # Features
y_labels = labels.iloc[:, 0]   # Target labels

# Apply PCA to reduce dimensionality
pca = PCA(n_components=71)  # Set the desired number of components
X_reduced = pca.fit_transform(X)

# Convert labels to one-hot encoded format
num_classes = len(np.unique(y_labels))
y = keras.utils.to_categorical(y_labels, num_classes=num_classes)

# Split the data into training, validation, and test sets
X_train, X_remain, y_train, y_remain = train_test_split(X_reduced, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_remain, y_remain, test_size=0.5, random_state=42)

# Create the KerasClassifier
model = KerasClassifier(build_fn=create_model, verbose=1)

# Define the parameter search space
parameters = {
    'optimizer': ['adam'],
    'activation': ['relu'],
    'epochs': [2],
    'batch_size': [64]
}

# Perform the parameter search
search = GridSearchCV(estimator=model, param_grid=parameters, cv=2)

# Fit the model with the callback
search.fit(X_train, y_train, shuffle=True)

# Print the results
print("Best parameters: ", search.best_params_)
print("Best accuracy: ", search.best_score_)

# Fit the model with the best parameters on the combined training and validation sets
best_model = search.best_estimator_
best_model.fit(X_val, y_val, epochs=1, batch_size=64, verbose=1, shuffle=True)

test_best_model = best_model
test_best_model.fit(X_test, y_test, epochs=1, batch_size=64, verbose=1, shuffle=True)

# Evaluate the model on the test data
y_test_pred = test_best_model.predict(X_test)
y_test_pred_one_hot = keras.utils.to_categorical(y_test_pred, num_classes=num_classes)
test_accuracy = accuracy_score(y_test, y_test_pred_one_hot)
print("Test accuracy on the test data: ", test_accuracy)

test_best_model.model.save('suv10.h5')

# Load the saved model
loaded_model = keras.models.load_model('suv10.h5')

# Use the loaded model for predictions on new data
new_data = pd.read_csv('testdata.txt', header=None)
new_X = new_data.iloc[:, :]
new_labels = loaded_model.predict(new_X)
predicted_labels = np.argmax(new_labels, axis=1)  # Convert probabilities to class labels

# Write the predicted labels to a text file
np.savetxt('testlabels.txt', predicted_labels, fmt='%d')

# Evaluate the loaded model on the test data
loaded_model_accuracy = accuracy_score(y_test, y_test_pred_one_hot)
print("Test accuracy for the loaded model: ", loaded_model_accuracy)
