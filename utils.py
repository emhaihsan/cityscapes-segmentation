import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical


## Preprocess Data
def load_and_preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, (192, 256))  # Resize to desired dimensions
    image = tf.image.convert_image_dtype(image, tf.float32) / 255.0  # Normalize to [0, 1]
    return image

def load_and_preprocess_label(label_path):
    label = tf.io.read_file(label_path)
    label = tf.image.decode_png(label, channels=1)
    label = tf.math.reduce_max(label, axis=-1, keepdims=True)
    label = tf.image.resize(label, (192, 256), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)  # Resize labels
    return label

def augment(image, label):
    image = tf.image.random_flip_left_right(image)  
    label = tf.image.random_flip_left_right(label)
    return image, label

# Visualisasi Segmentasi
class SegmentationVisualizationCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, dataset, frequency=10):
        super(SegmentationVisualizationCallback, self).__init__()
        self.model = model
        self.dataset = dataset
        self.frequency = frequency

    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) % self.frequency == 0:
            images, labels = next(iter(self.dataset))
            predicted_masks = self.model.predict(images)

            image = images[0]
            true_mask = labels[0]
            predicted_fcn32 = predicted_masks[0][0]
            predicted_fcn16 = predicted_masks[1][0]
            predicted_fcn8 = predicted_masks[2][0]

            fig, axes = plt.subplots(1, 5, figsize=(20, 5))
            axes[0].imshow(image)
            axes[0].set_title('Image')
            axes[1].imshow(true_mask, cmap='jet', vmin=0, vmax=12-1)
            axes[1].set_title('True Mask')
            axes[2].imshow(np.argmax(predicted_fcn32, axis=-1), cmap='jet', vmin=0, vmax=12-1)
            axes[2].set_title('Predicted FCN 32')
            axes[3].imshow(np.argmax(predicted_fcn16,axis=-1), cmap='jet', vmin=0, vmax=12-1)
            axes[3].set_title('Predicted FCN 16')
            axes[4].imshow(np.argmax(predicted_fcn8, axis=-1), cmap='jet', vmin=0, vmax=12-1)
            axes[4].set_title('Predicted FCN 8')

