import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as Img

import tensorflow as tf

from keras import backend
from keras.preprocessing import image
from keras.layers.core import K
from keras.utils import to_categorical
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions


from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import FastGradientMethod



IMG_PATH="path_to_input_image"
TARGET_CLASS=849 # teapot



def prep_image(image):
    image = np.clip(a=image, a_min=0.0, a_max=255.0)
    image = preprocess_input(image)
    return image


def plot_images(image, noise, both):
    fig, axes = plt.subplots(1, 3, figsize=(10,10))
    ax = axes.flat[0]
    ax.imshow(image/255.0)
    ax = axes.flat[1]
    ax.imshow(noise) # leave amplified
    ax = axes.flat[2]
    ax.imshow(both/255.0)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def adversarial_noise(model, image, target_class, noise_limit, sess, confidence=0.99, eps=1.0, max_iter=200):
    original = np.expand_dims(image, axis=0)
    target = np.array([target_class])
    encoded_target = to_categorical(target, num_classes=1000)


    wrap = KerasModelWrapper(model)
    fgsm = FastGradientMethod(wrap, sess=sess)
    fgsm_params = {'eps': eps, 'clip_min': 0.0, 'clip_max': 255.0, 
                   'y_target': encoded_target} 
     


    noisy = original
    for i in range(0, max_iter):
        noisy = fgsm.generate_np(noisy, **fgsm_params)
        current_confidence = model.predict(prep_image(noisy))[0][target_class]
        print(decode_predictions(model.predict(prep_image(noisy)), top=3)[0])
        print(current_confidence)
        if current_confidence > confidence:
            break      


    
    return np.reshape(noisy, noisy.shape[1:])




K.set_learning_phase(0)
sess = tf.Session()
backend.set_session(sess)


img = image.load_img(IMG_PATH, target_size=(224, 224))
input_image = image.img_to_array(img)
x = np.expand_dims(input_image, axis=0)
x = prep_image(x)



# pre-trained ResNet50 - get original prediction for our input image
model = ResNet50(weights='imagenet')
preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])


# generate adversarial example with a targeted iterative fast gradient sign method
noise = adversarial_noise(model, input_image, TARGET_CLASS, 0.1, sess)
noisy_image = np.expand_dims(noise, axis=0)
adv_preds = model.predict(prep_image(noisy_image))
print('Adversarial Predicted:', decode_predictions(adv_preds, top=1)[0])



plot_images(input_image,np.clip(a=input_image-noise, a_min=0.0, a_max=255.0),noise)
