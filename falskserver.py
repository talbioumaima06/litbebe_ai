from flask import Flask, request
import tensorflow as tf
import base64
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
new_model = tf.keras.models.load_model("C:\\Users\\hiche\\Desktop\\ia\\bebeM.keras")

@app.route('/api', methods=['PUT'])
def index():
    inputchar = request.get_data()
    imgdata = base64.b64decode(inputchar)
    filename = 'something.jpg'
    with open(filename, 'wb') as f:
        f.write(imgdata)

    # Predict the disease using the loaded model
    predicted_class = predict_disease('something.jpg', new_model)
    
    # Return the predicted disease
    return predicted_class

def predict_disease(filename, model):
    # Load and preprocess the image
    image = tf.keras.preprocessing.image.load_img(filename, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    
    # Perform model prediction
    prediction = model.predict(input_arr)
    
    # Get the index of the predicted class
    result_index = np.argmax(prediction)
    
    # Define class names
    class_names  = ['dangerous2', 'dangerous3', 'normal2', 'normal3']
    
    # Get the predicted class name
    predicted_class = class_names[result_index]
    
    return predicted_class

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
