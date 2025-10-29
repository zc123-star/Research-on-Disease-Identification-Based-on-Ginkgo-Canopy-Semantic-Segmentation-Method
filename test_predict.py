from PIL import Image

input_path = "./datasets/before/1_t.JPG" 
output_path = "./datasets/JPEGImages/01.JPG" 

img = Image.open(input_path)

if img.mode == 'RGBA':

    rgb_img = img.convert('RGB')

    rgb_img.save(output_path)


