from PIL import Image
import numpy as np
import PIL.ImageOps
import torch
import matplotlib.pyplot as plt

# This function replicates the image processing process that pytorch
# uses to prep image files for training/testing
def test_custom_image(file, network):
    print('Testing a custom image...')

    img = Image.open(file)
    img = img.resize((28,28))
    img = img.convert("L")
    img = PIL.ImageOps.invert(img)

    plt.imshow(img)
    plt.show()

    img = np.array(img)
    img = img / 255
    image = torch.from_numpy(img)
    image = image.float()

    result = network.forward(image.view(-1,28*28))
    print(f'Guess: {torch.argmax(result)}')