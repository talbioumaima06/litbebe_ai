import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 
print(tf.__version__)
print(tf.keras.__version__)
## load model
model  = tf.keras.models.load_model('bebeM.keras') 
model.summary()
## visuaalizing single image of test set
import cv2 as cv 
image_path = "C:\\Users\\MSI\\OneDrive\\Bureau\\pfe_project\\libebe_ai\\train 1\\dangerous3\\1717944357389.png"

#reading image
img = cv.imread(image_path)
img = cv.cvtColor(img,cv.COLOR_BGR2RGB) 
plt.imshow(img)
plt.title("test image")
plt.xticks([])#tfassekh l'axe des abscisses 
plt.yticks([])#tfassekh l'axe des ordonnées
plt.show()
## test model
image =tf.keras.preprocessing.image.load_img(image_path,target_size = (128 ,128 ))
input_arr = tf.keras.preprocessing.image.img_to_array(image)#neurone te9bel kan array form hadheka alech 7awelnaha
input_arr =np.array([input_arr])## conversion single image to batch yaani mdakhlyn bacth wa7da
print (input_arr.shape)
#perform model prediction 
prediction = model.predict(input_arr) 
print (prediction , prediction.shape)#
result_index = np.argmax(prediction) ## traje3lik el classe ely akber e7timalia enno s7i7
print(result_index)
class_name  = ['dangerous2', 'dangerous3', 'normal2', 'normal3']
#Display result prediction 
# Assuming class_name is a set
# Convert it to a list
class_name = list(class_name)
model_prediction = class_name[result_index]
plt.imshow(img)
plt.title(f"Position Name : {model_prediction}")
plt.xticks([])#tfassekh l'axe des abscisses 
plt.yticks([])#tfassekh l'axe des ordonnées
plt.show()