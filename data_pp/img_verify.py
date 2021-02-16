from PIL import Image
from os import listdir


mypath = "/home/GPU/tsathiak/local_storage/VisionTransformer/vision_transformer/data/cat_dog_dataset/PetImages/Cat/"
#img_file_names = [mypath+f for f in listdir(mypath)]
img_file_names = listdir(mypath)


for img_f in img_file_names:
   im =Image.open(mypath+img_f)
   #if(im.verify()!=None):
   print(im.verify())
