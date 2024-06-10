import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 as cv
import firebase_admin
from firebase_admin import credentials, db

# Initialize Firebase Admin SDK
try:
    cred = credentials.Certificate("./litbebe-a66b1-firebase-adminsdk-fjqwg-ee9fcda21f.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://litbebe-a66b1-default-rtdb.europe-west1.firebasedatabase.app'
    })
except Exception as e:
    print(f"Error initializing Firebase Admin SDK: {e}")
    exit()

# Print TensorFlow version
print(tf.__version__)
print(tf.keras.__version__)

# Load model
try:
    model = tf.keras.models.load_model('bebeM.keras')
    model.summary()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Get image path from Firebase Realtime Database
try:
    ref = db.reference('/camera/current_image')
    image_path = ref.get()
    if not image_path:
        raise ValueError("Image path is empty")
except Exception as e:
    print(f"Error fetching image path from Firebase: {e}")
    exit()

# Reading and displaying the image
try:
    first_path = 'C:\\Users\\MSI\\OneDrive\\Bureau\\pfe_project\\litbebe_server\\current\\'
    print("Image path:", first_path + image_path)
    img = cv.imread( first_path + image_path)
    if img is None:
        raise ValueError("Image not found or unable to read")
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title("test image")
    plt.xticks([]) # Hide x-axis
    plt.yticks([]) # Hide y-axis
    #plt.show()
except Exception as e:
    print(f"Error reading or displaying the image: {e}")
    exit()

# Preprocess the image
try:
    image = tf.keras.preprocessing.image.load_img(first_path + image_path, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) # Convert single image to batch
    print(input_arr.shape)
except Exception as e:
    print(f"Error preprocessing the image: {e}")
    exit()

# Perform model prediction
try:
    prediction = model.predict(input_arr)
    print(prediction, prediction.shape)
    result_index = np.argmax(prediction)
    print(result_index)
except Exception as e:
    print(f"Error during model prediction: {e}")
    exit()

# Map result to class name and display
try:
    class_name = ['dangerous2', 'dangerous3', 'normal2', 'normal3']
    class_name = list(class_name)
    model_prediction = class_name[result_index]
    plt.imshow(img)
    plt.title(f"Position Name: {model_prediction}")
    plt.xticks([]) # Hide x-axis
    plt.yticks([]) # Hide y-axis
    #plt.show()
except Exception as e:
    print(f"Error mapping result to class name or displaying: {e}")
    exit()

# Update the prediction result in Firebase Realtime Database
try:
    result_ref = db.reference('/camera/ai_listener')
    print(model_prediction)
    result_ref.set(model_prediction)
except Exception as e:
    print(f"Error updating prediction result in Firebase: {e}")
    exit()
