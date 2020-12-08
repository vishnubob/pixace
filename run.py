import funcs
import animals

adb = animals.Animals()
itr = adb.get_cursor()
img = next(itr)

img.save("in.png")
im = funcs.ImageBlocks(16, 16)
a_img = funcs.process_image(img, im)
im2 = funcs.process_array(a_img)
im2 = im2.resize((512, 512))
im2.save("out.png")

print(im2)
