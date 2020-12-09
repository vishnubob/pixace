from PIL import Image
from funcs import *
import animals
import random

adb = animals.Animals()
itr = adb.get_cursor()
img = list(itr)
img = random.choice(img)()

img = Image.open("in.png")
a_img = image_to_tokens(img, size=(4, 4))
print(a_img)
im2 = tokens_to_image(a_img)
im2 = im2.resize((512, 512))
im2.save("out.png")

print(im2)
