#importing libraries
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2

# Now you can use Conv2D, MaxPool2D, etc., directly without any issues.

data_dir = "train 1"

#Data preprocessing
#training image processing 
training_set= tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels="inferred",#i said to tensorflow go inside this folder (dataset) and go inside my train folder whatever the name of directory select it as my label 
    label_mode="categorical",#ki nabdo net3amlo m3a akther men 2 classes nesta3mlo categorical w ta3ni les labels codées as categorical vector
    color_mode="rgb",
    batch_size=32,#ki nebo nsar3o l'entrainement nzido feha n7otoha 64 ou 128
    image_size=(128, 128),
    shuffle=True,# at the time of feeding to my model for the training shuffle the entire thing and pass it and it reduce the biasness of the model if i shuffle it my model will learn from all end if i don't shuffle it it will pass some classes
)
# validation images preprocessing 
for x,y in training_set: #x value of each pixel y label
    print(x,x.shape)
    print(y,y.shape)
    break
# Building model 

model = Sequential() # Cet objet model sera utilisé pour construire votre réseau de neurones couche par couche
#Building convolutions layers
#convolution process
model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=[128,128,3], kernel_regularizer=tf.keras.regularizers.l2(0.001)))#nesta3mlo 32 filters tkhrajellna 32 different matrix(feature map) kernel size yaany input lowel matrice (3,3)
model.add(Conv2D(filters=32,kernel_size=3,activation='relu'))
#Pooling process
model.add(MaxPool2D(pool_size=2,strides=2))
model.add(Conv2D(filters=64,kernel_size=3,padding='same',activation='relu' ,  kernel_regularizer=tf.keras.regularizers.l2(0.001)))#
#Pooling process
model.add(MaxPool2D(pool_size=2,strides=2))
model.add(Conv2D(filters=128,kernel_size=3,padding='same',activation='relu' , kernel_regularizer=tf.keras.regularizers.l2(0.001)))#nesta3mlo 32 filters tkhrajellna 32 different matrix(feature map) kernel size yaany input lowel matrice (3,3)
model.add(Conv2D(filters=128,kernel_size=3,activation='relu'))
#Pooling process
model.add(MaxPool2D(pool_size=2,strides=2))
model.add(Conv2D(filters=256,kernel_size=3,padding='same',activation='relu' , kernel_regularizer=tf.keras.regularizers.l2(0.001)))#nesta3mlo 32 filters tkhrajellna 32 different matrix(feature map) kernel size yaany input lowel matrice (3,3)
model.add(Conv2D(filters=256,kernel_size=3,activation='relu'))
#Pooling process
model.add(MaxPool2D(pool_size=2,strides=2))
model.add(Conv2D(filters=512,kernel_size=3,padding='same',activation='relu' , kernel_regularizer=tf.keras.regularizers.l2(0.001)))#padding means whatever the input are coming take the same size
model.add(Conv2D(filters=512,kernel_size=3,activation='relu'))
#Pooling process
model.add(MaxPool2D(pool_size=2,strides=2))
model.add(Dropout(0.25)) #to avoid overfiting
#create mcuh of feature map to understand the caracteristiques and proprities of the image
model.add(Flatten())
#Add fully connected layer
model.add(Dense(units=1500,activation='relu'))#units means how many number of neurones i want
model.add(Dropout(0.5))
#output layer  
model.add(Dense(units=4,activation='softmax'))#11 neurones in output 3la gued les classes ely 3endy softmax give me probality of each class or neurones
#compiling the model
# Compile the model with appropriate loss function
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.00001) ,loss='categorical_crossentropy',metrics=['accuracy']) #Using adam as optimizer and categorical cross we use optimizer to minimize the loss 
model.summary()
#Training model
# Assuming your training and validation datasets are properly created
# training_set and validation_set should yield tuples (x_train, y_train) and (x_val, y_val) respectively
training_history = model.fit(training_set, epochs=40)
#Saving Model 
model.save('bebeM.keras')
print("Model saved")
