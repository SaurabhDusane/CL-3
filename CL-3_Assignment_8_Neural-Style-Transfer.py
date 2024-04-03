import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras import Model
import numpy as np
import matplotlib.pyplot as plt

# Load the content and style images
content_image_path = "content.jpg"
style_image_path = "style.jpg"

# Define the target image shape
target_shape = (400, 400)

# Load and preprocess the images
def load_and_process_image(image_path):
    img = load_img(image_path, target_size=target_shape)
    img = img_to_array(img)
    img = preprocess_input(img)
    return img

content_image = load_and_process_image(content_image_path)
style_image = load_and_process_image(style_image_path)

# Display the content and style images
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.imshow(content_image)
plt.title('Content Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(style_image)
plt.title('Style Image')
plt.axis('off')
plt.show()

# Load the VGG19 model pretrained on ImageNet without the fully connected layers
vgg = VGG19(include_top=False, weights='imagenet')

# Get the output layers corresponding to style and content layers
content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

# Create a model that returns the style and content layer outputs
style_outputs = [vgg.get_layer(name).output for name in style_layers]
content_outputs = [vgg.get_layer(name).output for name in content_layers]
model_outputs = style_outputs + content_outputs

# Build the model
model = Model(inputs=vgg.input, outputs=model_outputs)

# Create a function to compute the Gram matrix
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations

# Define the style and content weights
style_weight = 1e-2
content_weight = 1e4

# Use the Adam optimizer for optimization
optimizer = tf.optimizers.Adam(learning_rate=0.02)

# Define a function to compute the total loss
def total_loss(outputs):
    style_outputs = outputs[:len(style_layers)]
    content_outputs = outputs[len(style_layers):]

    style_score = 0
    content_score = 0

    for target_style, comb_style in zip(style_outputs, style_targets):
        style_score += tf.reduce_mean((gram_matrix(comb_style) - gram_matrix(target_style))**2)

    for target_content, comb_content in zip(content_outputs, content_targets):
        content_score += tf.reduce_mean((comb_content - target_content)**2)

    style_score *= style_weight / len(style_layers)
    content_score *= content_weight / len(content_layers)

    total_loss = style_score + content_score
    return total_loss

# Define the training loop
@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = model(image)
        loss = total_loss(outputs)

    grad = tape.gradient(loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))

# Initialize the generated image from the content image
generated_image = tf.Variable(content_image)

# Define the style and content targets
style_targets = model(style_image)
content_targets = model(content_image)

# Number of optimization steps
epochs = 1000

# Perform Neural Style Transfer
for epoch in range(epochs):
    train_step(generated_image)

# Display the generated image
plt.imshow(np.squeeze(generated_image.read_value(), 0))
plt.title('Generated Image')
plt.axis('off')
plt.show()
