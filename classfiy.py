from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import cv2
import os 
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("./models/keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

images = os.listdir("./images_to_test")
for image_path in images:
    image = Image.open("./images_to_test/"+image_path).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    image_to_show = cv2.imread("./images_to_test/"+image_path)
    image_to_show = cv2.resize(image_to_show, (724, 724))
    # Print prediction and confidence score
    image_to_show = cv2.putText(image_to_show, class_name[2:][:-1], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('image', image_to_show)
    cv2.waitKey(0)
    
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", confidence_score)
