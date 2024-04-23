import numpy as np
from keras.preprocessing import image as image_utils
from keras.applications import vgg16
from keras.models import load_model


def predict_image(image_path, model, preprocess_input_fn, pretrained_nn, class_labels):
    img = image_utils.load_img(image_path, target_size=(224, 224))
    img = image_utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input_fn(img)
    features = pretrained_nn.predict(img)
    predictions = model.predict(features)
    class_index = np.argmax(predictions[0])
    return class_labels[class_index], predictions[0][class_index]


HOME_PATH = './IA'

model_path = f"{HOME_PATH}/model.keras"
model = load_model(model_path)

pretrained_nn = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

class_labels = ['cachorro', 'cavalo', 'gato']

class_label, probability = predict_image('horse.jpeg', model, vgg16.preprocess_input, pretrained_nn, class_labels)
print(f'A imagem é de um {class_label.upper()} - Probabilidade: {probability:.2%}')
