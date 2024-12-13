import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

####STILL NEED TO ADD THE CLASSIFICATION FOR NATURAL AND MAN_MADE, RN IT ONLY CLASSIFIES THE OBJECTS TO ITS CATEGORY####

#paths to train and test directories
train_dir = "C:/Users/aryan/Documents/VSCode Python/478/project/data2bUsed/train"
test_dir = "C:/Users/aryan/Documents/VSCode Python/478/project/data2bUsed/test"

#preprocess the images to rescale them
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

#load the testing and training data from the directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),  # Resize images to 128x128
    batch_size=64,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=64,
    class_mode="categorical"
)

#create the model (refer to hw)
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(len(train_generator.class_indices), activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

#save the model locally so it can be used to test for classifying images categorically and natural vs man-made
model.save("C:/Users/aryan/Documents/VSCode Python/478/project/object_recognition_model.h5")
