import numpy as np
from pathlib import Path
from tensorflow import keras
from keras.optimizers import Adam
from keras.preprocessing import image as image_utils
from keras.applications import vgg16
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input

HOME_PATH = './IA'


def load_images(paths, target_size=(224, 224)):
    images = []
    labels = []
    for i, path in enumerate(paths):
        path = Path(HOME_PATH) / path
        for extension in ['*.png', '*.jpg', '*.jpeg', '*.webp']:
            for img in path.glob(extension):
                img = image_utils.load_img(img, target_size=target_size)
                image_array = image_utils.img_to_array(img)
                images.append(image_array)
                labels.append(i)
    return images, labels


class_labels = ['cachorro', 'cavalo', 'gato']
number_of_classes = len(class_labels)
images, labels = load_images(class_labels)

labels = keras.utils.to_categorical(labels, number_of_classes)

x_train = np.array(images)
y_train = np.array(labels)

x_train = vgg16.preprocess_input(x_train)
pretrained_nn = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x_train = pretrained_nn.predict(x_train)

model = Sequential([
    Input(shape=(7, 7, 512)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(number_of_classes, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=20, shuffle=True)

model.save(f"{HOME_PATH}/model.keras")
