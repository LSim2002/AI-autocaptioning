import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
def getlandmarks(impath):
    res = []
    
    image = mp.Image.create_from_file(impath)

    
    base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)
    detection_result = detector.detect(image)
    pose_landmarks_list = detection_result.pose_landmarks
    
    
    if len(pose_landmarks_list)==0 :
        print( "aucun corps humain détecté" )
        return [[k,-1,-1,-1,0] for k in range(33)]
 
    else:
        for idx in range(len(pose_landmarks_list[0])):
            landmark = pose_landmarks_list[0][idx]
            res.append([idx,landmark.x,landmark.y,landmark.z,landmark.visibility])
        return res
    
    

    
def getAngles(impath):
    coordinates=getlandmarks(impath)
    
    # Make sure you have the right keypoints
    if len(coordinates) != 33:
        raise ValueError('Expected a list with 33 keypoints')
    
    # Create a dictionary to easily access keypoints by their id
    keypoints_dict = {keypoint[0]: np.array(keypoint[1:5]) for keypoint in coordinates}

    # Define a function to calculate the angle between two vectors
    def calculate_angle(a, b, c):
        # Check visibility of keypoints
        if a[3] < 0.4 or b[3] < 0.4 or c[3] < 0.4:
            return -1

        vector_a_b = b[:3] - a[:3]
        vector_b_c = c[:3] - b[:3]

        dot_product = np.dot(vector_a_b, vector_b_c)

        magnitude_a_b = np.linalg.norm(vector_a_b)
        magnitude_b_c = np.linalg.norm(vector_b_c)

        angle_radians = np.arccos(dot_product / (magnitude_a_b * magnitude_b_c))

        return np.degrees(angle_radians)

    # Compute the angles for the left and right knee
    left_knee_angle = calculate_angle(keypoints_dict[23], keypoints_dict[25], keypoints_dict[27])
    right_knee_angle = calculate_angle(keypoints_dict[24], keypoints_dict[26], keypoints_dict[28])
    # Compute the angles for the left and right elbow
    left_elbow_angle = calculate_angle(keypoints_dict[11], keypoints_dict[13], keypoints_dict[15])
    right_elbow_angle = calculate_angle(keypoints_dict[12], keypoints_dict[14], keypoints_dict[16])
    # Compute the angles for the left and right hip
    left_hip_angle = calculate_angle(keypoints_dict[11], keypoints_dict[23], keypoints_dict[25])
    right_hip_angle = calculate_angle(keypoints_dict[11], keypoints_dict[24], keypoints_dict[26])
    # Compute the angles for the left and right shoulder
    left_shoulder_angle = calculate_angle(keypoints_dict[11], keypoints_dict[13], keypoints_dict[23])
    right_shoulder_angle = calculate_angle(keypoints_dict[11], keypoints_dict[14], keypoints_dict[24])
    # Compute the angles for the left and right ankle
    left_ankle_angle = calculate_angle(keypoints_dict[25], keypoints_dict[27], keypoints_dict[31])
    right_ankle_angle = calculate_angle(keypoints_dict[26], keypoints_dict[28], keypoints_dict[32])

    return [('left knee angle', round(left_knee_angle, 1)), ('right knee angle', round(right_knee_angle, 1)), 
        ('left elbow angle', round(left_elbow_angle, 1)), ('right elbow angle', round(right_elbow_angle, 1)),
        ('left hip angle', round(left_hip_angle, 1)), ('right hip angle', round(right_hip_angle, 1)),
        ('left shoulder angle', round(left_shoulder_angle, 1)), ('right shoulder angle', round(right_shoulder_angle, 1)),
        ('left ankle angle', round(left_ankle_angle, 1)), ('right ankle angle', round(right_ankle_angle, 1))]    
    














































from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import os



# Your data preparation here
# X, y = ...

# Initialize directories
running_dir = "C:\\Users\\lsimon\\Desktop\\autocaptioning\\LandmarkToPose(NN)\\running2"  # Replace with your actual directory
walking_dir = "C:\\Users\\lsimon\\Desktop\\autocaptioning\\LandmarkToPose(NN)\\walking2"  # Replace with your actual directory

# Initialize your X and y
X = []
y = []

##1=running, 0=walking

# Get your data
for directory, label in zip([running_dir, walking_dir], [1, 0]):  # Assuming 1: running, 0: walking
    for filename in os.listdir(directory):
        if filename.endswith(".png"):  # assuming images are .jpg, modify as needed
            filepath = os.path.join(directory, filename)
            
            # Get the angles from your function
            angles = getAngles(filepath)  # Replace with your actual function
            
            # Convert angles to a list of values while preserving the order
            angle_values = [angle[1] for angle in sorted(angles, key=lambda x: x[0])]
            
            # Append to your data
            X.append(angle_values)
            y.append(label)

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)


# Impute missing values in X with mean value of the feature
imp = SimpleImputer(missing_values=-1, strategy='mean')
X = imp.fit_transform(X)

# Replace -1 values (missing values) with numpy NaN
X[X == -1] = np.nan






# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the imputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Fit the imputer on the training data
imputer.fit(X_train)

# Transform the training and test data
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)

# Define the model
model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32)

# Save your model
model.save('C:\\Users\\lsimon\\.spyder-py3\\scriptsLouis\\models\\PCM.h5')

# Later you can load the model like this:
# loaded_model = load_model('C:\\Users\\lsimon\\.spyder-py3\\scriptsLouis\\models\\PCM.h5')


# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy}')











































































