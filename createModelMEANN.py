
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import numpy as np
import joblib
from termcolor import colored

# Load
X = np.load('C:\\Users\\lsimon\\Desktop\\autocaptioning\\LandmarkToPose(NN)\\trainingdata\\XN.npy')
y = np.load('C:\\Users\\lsimon\\Desktop\\autocaptioning\\LandmarkToPose(NN)\\trainingdata\\yN.npy')

# Convert labels to one-hot encoding
num_classes = len(np.unique(y))  # Get the number of unique classes
if num_classes > 2:
    y = to_categorical(y, num_classes)  # Convert to one-hot encoding

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Define the imputer
imputer = SimpleImputer(missing_values=-1, strategy='mean')

# Fit the imputer on the training data and transform it
X_train = imputer.fit_transform(X_train)

# Only transform the test data
X_test = imputer.transform(X_test)

# Define the model
model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
if num_classes==2:##si classif binaire
    model.add(Dense(1, activation='sigmoid'))  
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

else:
    model.add(Dense(num_classes, activation='softmax'))  # Use num_classes here
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Compile the model

# Fit the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=66, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print()
print(colored(f'Test accuracy: {accuracy}','green'))

accuracy=round(accuracy,2)

# Save your model
if accuracy >= 0.85:
    print()
    print(colored('modèle sauvegardé','green'))
    model.save(f'C:\\Users\\lsimon\\Desktop\\autocaptioning\\LandmarkToPose(NN)\\models\\PCM_{accuracy}_N.keras')
    joblib.dump(imputer, f'C:\\Users\\lsimon\\Desktop\\autocaptioning\\LandmarkToPose(NN)\\models\\imputer_{accuracy}_N.pkl')









































































