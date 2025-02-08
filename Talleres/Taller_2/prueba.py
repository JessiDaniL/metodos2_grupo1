from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt
# # load the image
# image = Image.open('Talleres\Taller_2\catto.png')
# # convert image to numpy array
# data = asarray(image)
# print(type(data))
# # summarize shape
# print(data.shape)

# # create Pillow image
# image2 = Image.fromarray(data)
# print(type(image2))

img_gato = plt.imread("Talleres\Taller_2\catto.png")
print(img_gato)