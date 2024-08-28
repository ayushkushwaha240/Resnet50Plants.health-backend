import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np

def load_model():
    # Load the pre-trained ResNet50 model
    model = tf.keras.models.load_model('path_to_your_model/resnet50_model.h5')
    return model

def predict(model, img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    # Make a prediction
    predictions = model.predict(x)
    
    # # Assuming you have a list of your 8 class labels
    # class_labels = ['Healthy', 'Powdery', 'Rusty']
    
    # # Get the index of the highest probability class
    # predicted_index = np.argmax(predictions[0])
    
    # # Get the corresponding label
    # predicted_label = class_labels[predicted_index]
    
    # Return the result
    return predictions
