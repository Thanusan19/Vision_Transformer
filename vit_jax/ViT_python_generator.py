import math
import random
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import re

import time
import sys
from os.path import dirname
sys.path.append(dirname('/home/GPU/tsathiak/local_storage/Vision_Transformer'))


######################################################################################
#Functions to create the exclusion list and the global description
######################################################################################

def checkExistanceAndEmptiness(output_file_path: str, doOverwrite: bool):
  okayToOverwrite = True
  output_path = Path(output_file_path)
  if output_path.exists():
    print('File exists')
    if output_path.stat().st_size != 0:
      print('File is not empty')
      if not doOverwrite:
        okayToOverwrite = False
        print('not over-writing')
      else:
        mode = 'w+'
        print('over-writing')

    else:
      print('File is empty')
      mode = 'w+'

  else:
    print('File don\'t exist')
    mode = 'w'
  return mode, okayToOverwrite


def createExclusionFile(dataset_dir_path: str, mogrify_output_file_path: str,
                        output_file_path: str, doOverwrite: bool = False):
  """
  dataset_dir_path le chemin d'accès au dossier du dataset
  output_file_path le chemin du fichier que l'on veut créer
  doOverwrite permet d'écraser le fichier, si il existe déjà, si le paramètre
    est passé à True (False par defaut).
  """
  print

  # Check if file exists or not and gives the write or write and read depending,
  # as well as the bolean to overwrite or not the file
  mode, okayToOverwrite = checkExistanceAndEmptiness(
      output_file_path, doOverwrite)

  dataset_path = Path(dataset_dir_path)
  output_path = Path(output_file_path)
  print(dataset_path)
  if okayToOverwrite:
    with output_path.open(mode) as outfile:
      #writing in the file

      # Lecture du fichier d'exclusion
      mogrify_output = Path(mogrify_output_file_path)
      regex_files = re.compile('dataset/.*/[0-9]*.jpg')

      added_lines = []
      with mogrify_output.open('r') as infile:
        for line in infile.readlines():
          # time.sleep(1)
          if line.endswith("\n"):
            line = line[:-1]

            first_match = regex_files.findall(line)[0]
            first_path = Path(first_match)
            string = str(first_path.relative_to(dataset_path))
            # string = first_match.replace(str(dataset_path)+"/", "")

            if string not in added_lines:
              outfile.write(string+"\n")
              added_lines.append(string)


def createGlobalDescription(dataset_dir_path: str, exclude_img_file_path: str,
                            output_file_path: str, doOverwrite: bool = False):
  """
  Va generer le fichier de tout le dataset
  dataset_dir_path le chemin d'accès au dossier du dataset
  exclude_img_file_path le chemin d'accès d'un fichier d'exclusion de fichiers
    corrompus dans la liste. De la forme :
      path/vers/le/fichier1.jpg
      path/vers/le/fichier2.jpg
      path/vers/le/fichier3.jpg
      path/vers/le/fichier4.jpg
  output_file_path le chemin du fichier que l'on veut créer
  doOverwrite permet d'écraser le fichier, si il existe déjà, si le paramètre
    est passé à True (False par defaut).
  """

  # Lecture du fichier d'exclusion
  exclude_path = Path(exclude_img_file_path)
  exclude_img_list = []
  with exclude_path.open('r') as file:
    for line in file.readlines():
      if line.endswith("\n"):
        line = line[:-1]
        line = str(Path(line))  # To be able to compare it to other file path
        #print("exclude file line :", line)

      exclude_img_list.append(line)
  print("exclude_img_list", exclude_img_list)

  # Compter celui qui a le plus d'exclus, pour en avoir le même nombre de
  # chaque coté
  count_cat = 0
  count_dog = 0
  for exclude_file in exclude_img_list:
    #print("Cat or Dog ?", exclude_file.split("/")[-2])
    if exclude_file.split("/")[-2] == 'Cat':
      count_cat += 1
    else:
      count_dog += 1
  print("count_cat", count_cat, "count_dog", count_dog)
  left_to_exclude_dogs = count_cat-count_dog if count_cat >= count_dog else 0
  left_to_exclude_cats = count_dog-count_cat if count_dog >= count_cat else 0

  # Check if file exists or not and gives the write or write and read depending,
  # as well as the bolean to overwrite or not the file
  mode, okayToOverwrite = checkExistanceAndEmptiness(
      output_file_path, doOverwrite)

  output_path = Path(output_file_path)
  # Ecriture du fichier
  if okayToOverwrite:
    with output_path.open(mode) as file:
      #writing in the file
      ds_dir_path = Path(dataset_dir_path)
      print("ds_dir_path", ds_dir_path)

      class_num = -1
      #for class_dir in ds_dir_path.joinpath("PetImages").iterdir():
      for class_dir in ds_dir_path.iterdir():
        print("CLASS DIR : ", class_dir)
        if class_dir.is_dir():
          class_num += 1
          print("  class_dir", class_dir)
          print("  class_num", class_num)

          if str(class_dir).endswith('Cat'):
            left_to_exclude_count = left_to_exclude_cats
            print("  left_to_exclude_count for Cats is :", left_to_exclude_count)
          else:
            left_to_exclude_count = left_to_exclude_dogs
            print("  left_to_exclude_count for Dogs is :", left_to_exclude_count)

          added_count = 0
          for class_img in class_dir.iterdir():
            #if class_img.match('[0-9]*.jpg'):

            local_image_path = class_img.relative_to(ds_dir_path)
            # Check for exclusion

            #print("class_img:", class_img)
            #print("exclude_img_list:", exclude_img_list)
            #print("class_img relative to:", str(class_img.relative_to(ds_dir_path)))
            #time.sleep(2)
            if str(local_image_path) not in exclude_img_list:
              #print("    ds_dir_path", ds_dir_path)
              #print("    class_dir", class_dir)
              #print("    class_img", class_img)
              if left_to_exclude_count > 0:
                left_to_exclude_count -= 1
                #print("    class_img", class_img)
                print("    > that was a left to exclude", local_image_path)
                #time.sleep(1)
              else:
                file.write(str(local_image_path) +
                           "\t" + str(class_num) + "\n")
                added_count += 1
            else:
              #print("    class_img", class_img)
              print("    > excluded from the exclusion list", local_image_path)
              #time.sleep(1)

          if str(class_dir).endswith('Cat'):
            print("Added", added_count, "cats to the description file")
          else:
            print("Added", added_count, "dogs to the description file")


#####################################################################
#Create the exclusion list and the global description
#####################################################################

#MAIN_PATH = "" #"/home/GPU/tsathiak/local_storage/Vision_Transformer"
#dataset_dir_path = MAIN_PATH +  "dataset/CatsAndDogs"
#mogrify_output_file_path = MAIN_PATH + "dataset/CatsAndDogs/mogrify_output"
#exclude_img_file_path = MAIN_PATH + "dataset/CatsAndDogs/exclude.txt"
#output_description_file_path = MAIN_PATH + "dataset/CatsAndDogs/description.txt"

MAIN_PATH = ""  # "/home/GPU/tsathiak/local_storage/Vision_Transformer"
dataset_dir_path = MAIN_PATH + "dataset/diatom_dataset"
mogrify_output_file_path = MAIN_PATH + "dataset/diatom_dataset/mogrify_output"
exclude_img_file_path = MAIN_PATH + "dataset/diatom_dataset/exclude.txt"
output_description_file_path = MAIN_PATH + "dataset/diatom_dataset/description.txt"


#createExclusionFile(dataset_dir_path=dataset_dir_path,
#                    mogrify_output_file_path=mogrify_output_file_path,
#                    output_file_path=exclude_img_file_path,
#                    doOverwrite=True)

createGlobalDescription(dataset_dir_path=dataset_dir_path,
                        exclude_img_file_path=exclude_img_file_path,
                        output_file_path=output_description_file_path,
                        doOverwrite=True)


######################################################################
#Create a training and a test set
######################################################################
class MyDogsCats:
  def __init__(self, dataset_path: str, X, y, num_class: int, set_type: str,
               doDataAugmentation: bool = False) -> None:
    """
    ds_description_path : fichier avec les paths de chaque fichiers du dataset et sa classe
    Exemple de fichier (tabulation entre le path et la classe):
        /truc/bidule/chat/01.jpg    0
        /truc/bidule/chien/01.jpg   1
        Etc ...
    """

    #print(img_list_par_classes)

    # Obtenir la liste de train OU de test
    self._img_list = X
    self._lbl_list = y

    #print("_img_list", self._img_list[0:100])
    #print("_lbl_list", self._lbl_list[0:100])
    assert(len(self._lbl_list) == len(self._img_list))

    self.num_samples = len(self._lbl_list)

    if set_type == "train" or set_type == "test" or set_type == "val":
      self._set_type = set_type
    else:
      self._set_type = "whole"

    self._start_im_size = 256
    self._end_im_size = 384
    self._start_im_dim = np.array((self._start_im_size, self._start_im_size))
    self._end_im_dim = np.array((self._end_im_size, self._end_im_size))

    self._num_channels = 3
    self._one_hot_depth = num_class

    self._do_data_augmentation = doDataAugmentation

    self._ds_path = Path(dataset_path)

  def getDataset(self):
      generator = self._generator
      print("nbr samples ", self.num_samples)
      return tf.data.Dataset.from_generator(generator,
                                            args=[],
                                            output_types={'image': tf.float32, 'label': tf.int32},
                                            output_shapes={'image': tf.TensorShape((self._end_im_size, self._end_im_size, self._num_channels)),
                                                           'label': tf.TensorShape((self._one_hot_depth))})

  #####
  # Generator
  #####

  def _generator(self):

    # Setup for Data Augmentation
    if self._do_data_augmentation and self._set_type == 'train':
      start_im_center = tuple(self._start_im_dim//2)
      end_im_center = tuple(self._end_im_dim//2)

      # Parameters
      # blend_size = (120, 120)
      translate_range = (-80, 80)
      # total_size = blend_size[0] + blend_size[1]

      # left = blend_size[0]
      fond = np.zeros(tuple(np.append(tuple(self._start_im_dim), 3)), np.uint8)

      # blank mask in uint8 representation (255)
      blank = 255 * \
          np.ones(tuple(np.append(tuple(self._start_im_dim), 3)), np.uint8)

    img_list = self._img_list
    lbl_list = self._lbl_list

    # Shuffle
    c = list(zip(img_list, lbl_list))
    random.shuffle(c)
    img_list, lbl_list = zip(*c)

    # #Stratified Split into Train and Test dataset
    # X_train, X_test, y_train, y_test = train_test_split(img_list, lbl_list, test_size=0.2, random_state=42, stratify=lbl_list)
    # #Stratified Split into Train and Validation dataset
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    for i in range(self.num_samples):
      # Read the image
      im = cv2.imread(str(self._ds_path/img_list[i]), -1)

      # If we couldn't read the image or other errors, we take the first image in the list
      if im is None:
        i = 0
        im = cv2.imread(str(self._ds_path/img_list[0]), -1)

      # If in black and white, replicate in 3 channels
      if len(im.shape) < 3:
        im = np.repeat(np.expand_dims(im, -1), 3, -1)

      # Convert from BGR to RGB
      im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

      # Data Augmentation :
      if self._do_data_augmentation and self._set_type == 'train':
        # start by ensuring images are the size of the start dim
        im = cv2.resize(im, tuple(self._start_im_dim))

        # Find the color to blend (update : take the median color of the image)
        # fond_couleur = np.median(im[1:30, left//2:left//2+10, :], axis=[0,1])
        fond_couleur = np.median(im[:, :, :], axis=[0, 1])
        # fond_couleur = (0, 255, 0)

        fond[:, :, :] = fond_couleur

        #####
        # OLD blending
        #####
        # Make the blend between the image and the background color
        # im = np.round((im/255.0 * blend_all + fond/255.0 * (1 - blend_all))*255.0).astype(np.uint8)

        # Seamless cloning between the image and the new background
        # From https://learnopencv.com/seamless-cloning-using-opencv-python-cpp/
        # Small tweak could be between cv2.MIXED_CLONE or cv2.NORMAL_CLONE, but I can't see the difference for this dataset
        im = cv2.seamlessClone(src=im, dst=fond, mask=blank,
                               p=start_im_center, flags=cv2.MIXED_CLONE)

        # Translate to the center of the bigger image
        tx_ty = np.floor(((self._start_im_dim - self._end_im_dim)/2))
        trans_mat = np.column_stack([[1, 0], [0, 1], tx_ty])
        im = cv2.warpAffine(im, trans_mat, tuple(self._end_im_dim),
                            borderMode=cv2.BORDER_CONSTANT, borderValue=fond_couleur)

        # Choose a random angle
        angle = np.random.rand()*360

        # Rotate the image
        rot_mat = cv2.getRotationMatrix2D(end_im_center, angle, 1.0)
        im = cv2.warpAffine(im, rot_mat, tuple(self._end_im_dim),
                            borderMode=cv2.BORDER_CONSTANT, borderValue=fond_couleur)

        # Choose a random translation
        a, b = translate_range
        rand_tx_ty = (b - a) * np.random.random_sample(2) + a

        # Translate the image
        rand_trans_mat = np.column_stack([[1, 0], [0, 1], rand_tx_ty])
        im = cv2.warpAffine(im, rand_trans_mat, tuple(self._end_im_dim),
                            borderMode=cv2.BORDER_CONSTANT, borderValue=fond_couleur)

        img = im
      else:
        img = cv2.resize(im, tuple(self._end_im_dim))

      # Normalization
      img = (img - 127.5) / 127.5

      # Label
      lbl = tf.one_hot(lbl_list[i], depth=self._one_hot_depth, dtype=tf.int32)

      yield {'image': img, 'label': lbl}

  def get_num_samples(self):
      return self.num_samples
