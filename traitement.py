#importing libraries
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2

# Now you can use Conv2D, MaxPool2D, etc., directly without any issues.

data_dir = "train 1"

#Data preprocessing
#training image processing 
training_set = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels="inferred",  # Automatically infer labels
    label_mode="categorical",  # Use categorical labels for multi-class classification
    color_mode="rgb",  # Use RGB images
    batch_size=32,  # Batch size
    image_size=(128, 128),  # Image size to resize to
    shuffle=True  # Shuffle the dataset
)

# validation images preprocessing 
for x, y in training_set:  # Print a batch of images and labels
    print(x, x.shape)
    print(y, y.shape)
    break

# Building model 
model = Sequential()  # Create a Sequential model

# Add an Input layer
model.add(Input(shape=(128, 128, 3)))

#Building convolutions layers
model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(0.001)))  # First Conv layer
model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))  # Second Conv layer
model.add(MaxPool2D(pool_size=2, strides=2))  # First MaxPooling layer

model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(0.001)))  # Third Conv layer
model.add(MaxPool2D(pool_size=2, strides=2))  # Second MaxPooling layer

model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(0.001)))  # Fourth Conv layer
model.add(Conv2D(filters=128, kernel_size=3, activation='relu'))  # Fifth Conv layer
model.add(MaxPool2D(pool_size=2, strides=2))  # Third MaxPooling layer

model.add(Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(0.001)))  # Sixth Conv layer
model.add(Conv2D(filters=256, kernel_size=3, activation='relu'))  # Seventh Conv layer
model.add(MaxPool2D(pool_size=2, strides=2))  # Fourth MaxPooling layer

model.add(Conv2D(filters=512, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(0.001)))  # Eighth Conv layer
model.add(Conv2D(filters=512, kernel_size=3, activation='relu'))  # Ninth Conv layer
model.add(MaxPool2D(pool_size=2, strides=2))  # Fifth MaxPooling layer

model.add(Dropout(0.25))  # Add Dropout to avoid overfitting

model.add(Flatten())  # Flatten the output from the convolutional layers

#Add fully connected layers
model.add(Dense(units=1500, activation='relu'))  # First Dense layer
model.add(Dropout(0.5))  # Add Dropout to avoid overfitting

# Output layer  
model.add(Dense(units=4, activation='softmax'))  # Output layer with 4 classes

# Compile the model with appropriate loss function
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()  # Print the model summary

#Training model
training_history = model.fit(training_set, epochs=40)  # Train the model

#Saving Model 
model.save('bebeM.keras')  # Save the model
print("Model saved")
