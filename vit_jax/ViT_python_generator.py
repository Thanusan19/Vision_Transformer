from pathlib import Path
import re

import time
import sys
from os.path import dirname
sys.path.append(dirname('/home/GPU/tsathiak/local_storage/Vision_Transformer'))


from pathlib import Path
import tensorflow as tf
import numpy as np
import cv2
import random
import math


######################################################################################
#Functions to create the exclusion list and the global description
######################################################################################

def checkExistanceAndEmptiness(output_file_path:str, doOverwrite:bool):
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

def createExclusionFile(dataset_dir_path:str, mogrify_output_file_path:str,
                        output_file_path:str, doOverwrite:bool=False):
  """
  dataset_dir_path le chemin d'accès au dossier du dataset
  output_file_path le chemin du fichier que l'on veut créer
  doOverwrite permet d'écraser le fichier, si il existe déjà, si le paramètre
    est passé à True (False par defaut).
  """
  print

  # Check if file exists or not and gives the write or write and read depending,
  # as well as the bolean to overwrite or not the file
  mode, okayToOverwrite = checkExistanceAndEmptiness(output_file_path, doOverwrite)

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

def createGlobalDescription(dataset_dir_path:str, exclude_img_file_path:str,
                           output_file_path:str, doOverwrite:bool=False):
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
              line = str(Path(line)) # To be able to compare it to other file path
              #print("exclude file line :", line)
          
          exclude_img_list.append(line)
  print("exclude_img_list", exclude_img_list)

  # Compter celui qui a le plus d'exclus, pour en avoir le même nombre de
  # chaque coté
  count_cat = 0; count_dog = 0
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
  mode, okayToOverwrite = checkExistanceAndEmptiness(output_file_path, doOverwrite)

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
                file.write(str(local_image_path) + "\t" + str(class_num) + "\n")
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

MAIN_PATH = "" #"/home/GPU/tsathiak/local_storage/Vision_Transformer"
dataset_dir_path = MAIN_PATH +  "dataset/diatom_dataset"
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
    def __init__(self, ds_description_path:str, dataset_path:str, set_type:str, train_prop:float) -> None:
        """
        ds_description_path : fichier avec les paths de chaque fichiers du dataset et sa classe
        Exemple de fichier (tabulation entre le path et la classe):
          /truc/bidule/chat/01.jpg  0
          /truc/bidule/chien/01.jpg 1
          Etc ...
        """

        # Lire le fichier de description et regrouper par classes
        img_list_par_classes = {}
        path = Path(ds_description_path)
        with path.open('r') as file:
          for line in file.readlines():
            if line.endswith("\n"):
              line = line[:-1]
            splits = line.split("\t")

            if line != "":
              img_text = splits[0]
              lbl_text = int(splits[1])

              if lbl_text in img_list_par_classes.keys():
                img_list_par_classes[lbl_text].append(img_text)
              else:
                img_list_par_classes[lbl_text] = [img_text]

        #print(img_list_par_classes)

        # Obtenir la liste de train OU de test
        self._img_list = []
        self._lbl_list = []
        self._num_class = len(img_list_par_classes)
        print("Num class : ", self._num_class)

        for num_class in img_list_par_classes:
          # Definir les proportions
          num_files = len(img_list_par_classes[num_class])
          if set_type == "train":
            num_per_class_to_keep = math.ceil(num_files * train_prop)

            class_files = img_list_par_classes[num_class][0:num_per_class_to_keep]
          
          elif set_type == "test":
            num_per_class_to_keep = math.floor(num_files * (1 - train_prop))

            class_files = img_list_par_classes[num_class][-num_per_class_to_keep:]
          
          else:
            class_files = img_list_par_classes[num_class]
          
          # Ajouter les images qui correspondent à la liste des images
          self._img_list.extend(class_files)
          # De même pour les labels
          #print("num_class:", num_class)
          #print("type num_class:", type(num_class))
          #print("len num_class:", len(class_files))
          self._lbl_list.extend([num_class for i in range(len(class_files))])

        #print("_img_list", self._img_list[0:100])
        #print("_lbl_list", self._lbl_list[0:100])
        assert(len(self._lbl_list) == len(self._img_list))

        self.num_samples = len(self._lbl_list)

        if set_type == "train" or set_type == "test":
          self._set_type = set_type
        else:
          self._set_type = "whole"
        
        self._img_size = 384 #256
        self._img_dim = (self._img_size, self._img_size)
        self._num_channels = 3
        self._one_hot_depth = self._num_class

        self._ds_path = Path(dataset_path)
    
    def getDataset(self):
        generator = self._generator
        print("nbr samples ", self.num_samples)
        return tf.data.Dataset.from_generator(generator,
                                              args=[],
                                              output_types={'image': tf.float32, 'label': tf.int32},
                                              output_shapes={'image': tf.TensorShape((self._img_size, self._img_size, self._num_channels)),
                                                             'label': tf.TensorShape((self._one_hot_depth))})
    
    def _generator(self):
        img_list = self._img_list
        lbl_list = self._lbl_list

        # Shuffle

        c = list(zip(img_list, lbl_list))
        random.shuffle(c)
        img_list, lbl_list = zip(*c)

        for i in range(self.num_samples):
            #print('Reading from :', img_list[i])
            #print('Good path :', self._ds_path/img_list[i])
            #self._ds_path/img_list[i]
            #print(self._ds_path/img_list[i])
            # img_path_i = Path(img_list[i])
            im = cv2.imread(str(self._ds_path/img_list[i]),-1)
            if im is None:
              i = 0
              im = cv2.imread(str(self._ds_path/img_list[0]),-1)

            if len(im.shape) < 3:
              im = np.repeat(np.expand_dims(im, -1), 3, -1)
            #print(type(im))

            img = cv2.resize(im, self._img_dim)
            img = img/255.0
            #img = np.expand_dims(im, -1)
            lbl = tf.one_hot(lbl_list[i], depth=self._one_hot_depth, dtype=tf.int32)
            yield {'image': img, 'label': lbl}


    def get_num_samples(self):
        return self.num_samples
