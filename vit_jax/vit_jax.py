from ViT_python_generator import MyDogsCats
from clu import metric_writers
import time
import os
import glob
import functools
import tensorflow as tf
import sys
import flax
import jax
from matplotlib import pyplot as plt
import numpy as np
import tqdm
import PIL
import seaborn
from sklearn.model_selection import train_test_split
from pathlib import Path

import checkpoint
import input_pipeline
import models
import momentum_clip
import train
import hyper
import logging_ViT
import flax.jax_utils as flax_utils
import jax.numpy as jnp


from os.path import dirname
sys.path.append(
    dirname('/home/GPU/tsathiak/local_storage/Vision_Transformer/'))


# Shows the number of available devices.
# In a CPU/GPU runtime this will be a single device.
jax.local_devices()

# Pre-trained model name
model = 'ViT-B_16'

logdir = './logs'
logger = logging_ViT.setup_logger('./logs')
INFERENCE = False
FINE_TUNE = True
CHECKPOINTS_TEST = False

# Helper functions for images.

labelnames = dict(
    # https://www.cs.toronto.edu/~kriz/cifar.html
    cifar10=('airplane', 'automobile', 'bird', 'cat', 'deer',
             'dog', 'frog', 'horse', 'ship', 'truck'),
    # https://www.cs.toronto.edu/~kriz/cifar.html
    cifar100=('apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'computer_keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
              'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm')
)


def make_label_getter(dataset):
  """Returns a function converting label indices to names."""
  def getter(label):
    if dataset in labelnames:
      return labelnames[dataset][label]
    return f'label={label}'
  return getter


def show_img(img, ax=None, title=None):
  """Shows a single image."""
  if ax is None:
    ax = plt.gca()
  ax.imshow(img[...])
  ax.set_xticks([])
  ax.set_yticks([])
  if title:
    ax.set_title(title)


def show_img_grid(imgs, titles):
  """Shows a grid of images."""
  n = int(np.ceil(len(imgs)**.5))
  _, axs = plt.subplots(n, n, figsize=(3 * n, 3 * n))
  for i, (img, title) in enumerate(zip(imgs, titles)):
    img = (img + 1) / 2  # Denormalize
    show_img(img, axs[i // n][i % n], title)


def print_banner(message):
  print("\n###################################")
  print(message)
  print("###################################\n")


def _shard(data):
  data['image'] = tf.reshape(data['image'], [num_devices, -1, 384, 384, 3])
  data['label'] = tf.reshape(data['label'], [num_devices, -1, 166])  # 2
  return data


def get_accuracy(params_repl, nbr_samples):
  """Returns accuracy evaluated on the test set."""
  good = total = 0
  steps = nbr_samples // batch_size #20000 // batch_size

  for _, batch in zip(tqdm.trange(steps), ds_test.as_numpy_iterator()):
    predicted = vit_apply_repl(params_repl, batch['image'])
    is_same = predicted.argmax(axis=-1) == batch['label'].argmax(axis=-1)
    good += is_same.sum()
    total += len(is_same.flatten())
  return good / total


def get_accuracy_val(params_repl, nbr_samples):
  """Returns accuracy evaluated on the val set."""
  good = total = 0
  steps = nbr_samples // batch_size

  for _, batch in zip(tqdm.trange(steps), ds_val.as_numpy_iterator()):
    predicted = vit_apply_repl(params_repl, batch['image'])
    is_same = predicted.argmax(axis=-1) == batch['label'].argmax(axis=-1)
    good += is_same.sum()
    total += len(is_same.flatten())
  return good / total


def get_accuracy_train(params_repl, nbr_samples, batch):
  """Returns accuracy evaluated on the val set."""
  good = total = 0
  #steps = nbr_samples // batch_size

  #for _, batch in zip(tqdm.trange(steps), ds_train.as_numpy_iterator()):
  predicted = vit_apply_repl(params_repl, batch['image'])
  is_same = predicted.argmax(axis=-1) == batch['label'].argmax(axis=-1)
  good += is_same.sum()
  total += len(is_same.flatten())

  return good / total


def cross_entropy_loss(*, logits, labels):
  logp = jax.nn.log_softmax(logits)
  return -jnp.mean(jnp.sum(logp * labels, axis=1))

def loss_fn(params, update_rng, images, labels):
  with flax.nn.stochastic(update_rng):
    logits = VisionTransformer.call(params, images, train=True)
  return cross_entropy_loss(logits=logits, labels=labels)


def learning_curve_per_train_steps(Loss_list):
  """Plot learning Curve every trainnig steps"""
  print(Loss_list)
  fig = plt.figure()
  plt.title('Learning Curve : Diatom Dataset')
  plt.plot(Loss_list)
  plt.yscale('log')
  plt.xlabel('training_steps')
  plt.ylabel('Loss : Cross Entropy')
  fig.savefig('Learning_curve_plot_diatom_per_training_steps.png')


def plot_learning_curve_per_epochs(train_loss_per_training_steps, val_loss_per_training_steps, steps_per_epoch, total_steps):
  """Plot learning Curve per epochs"""
  Loss_per_epochs = []
  val_loss_per_epochs = []
  for i in range(0, total_steps, steps_per_epoch):
     Loss_per_epochs.append(train_loss_per_training_steps[i])
     val_loss_per_epochs.append(val_loss_per_training_steps[i])

  fig = plt.figure()
  plt.title('Learning Curve : Diatom Dataset')
  plt.plot(Loss_per_epochs, 'b', label='train')
  plt.plot(val_loss_per_epochs, 'g', label='val')
  plt.yscale('log')
  plt.xlabel('Epochs')
  plt.ylabel('Loss : Cross Entropy')
  fig.savefig('Learning_curve_plot_diatom_per_epochs.png')


def plot_acc_per_epochs(train_acc_per_training_steps, val_acc_per_training_steps, steps_per_epoch, total_steps):
  """Plot learning Curve per epochs"""
  val_acc_per_epochs = []
  train_acc_per_epochs = []
  for i in range(0, total_steps, steps_per_epoch):
     val_acc_per_epochs.append(val_acc_per_training_steps[i])
     train_acc_per_epochs.append(train_acc_per_training_steps[i])

  fig = plt.figure()
  plt.title('Acc Curve : Diatom Dataset')
  plt.plot(train_acc_per_epochs, 'b',label='train')
  plt.plot(val_acc_per_epochs, 'g', label='val')
  plt.yscale('log')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  fig.savefig('Acc_curve_plot_diatom_per_epochs.png')


def plot_confusion_matrix(data, labels, output_filename):
    """Plot confusion matrix using heatmap.
    Args:
        data (list of list): List of lists with confusion matrix data.
        labels (list): Labels which will be plotted across x and y axis.
        output_filename (str): Path to output file.

    """
    seaborn.set(color_codes=True)
    plt.figure(1, figsize=(9, 6))

    plt.title("Confusion Matrix")

    #seaborn.set(font_scale=1.4)
    #ax = seaborn.heatmap(data, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'})

    ax = seaborn.heatmap(data, annot=True, fmt="d")

    #ax.set_xticklabels(labels)
    #ax.set_yticklabels(labels)

    ax.set(ylabel="True Label", xlabel="Predicted Label")

    #plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.savefig(output_filename)
    plt.close()


def plot_confusion_matrix(cm, title, img_save_filename, cmap=None, normalize=True, log_scale=True):
    """Plot confusion matrix.

    Args:
        cm (list of list): List of lists with confusion matrix data.
        title (str): Title of the confusion matrix image.
        img_save_filename (str): Path to save confusion matrix image

    """

    cm = cm.numpy()

    if cmap is None:
      cmap = plt.get_cmap('jet')  # Blues

    if normalize:
      nbr_sample_per_classe = cm.sum(axis=1)
      nbr_sample_per_classe[nbr_sample_per_classe == 0] = 1
      #Divide each row per nbr_sample_per_classe
      cm = cm / nbr_sample_per_classe[:, None]

    if log_scale:
      cm = np.log(cm)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig(img_save_filename)
    plt.close()


def get_predict_labels_on_test_data(params_repl):
  """Returns Predicted Class list and Label Class list """

  steps = 20000 // batch_size
  predicted = []
  labels = []
  for _, batch in zip(tqdm.trange(steps), ds_test.as_numpy_iterator()):
    predicted.append(vit_apply_repl(
        params_repl, batch['image']).argmax(axis=-1))
    labels.append(batch['label'].argmax(axis=-1))

  return predicted, labels

# Function to read the description file, and split the file path and classes
# into three sets


def make_split_from_descfile(ds_description_path: str, dataset_path: str,
                             train_prop: float, test_prop: float, doVal: bool = False, val_prop: float = 0.0):

  # Make sure the proportions are valid
  assert(train_prop + val_prop + test_prop == 1.0)

  # Lire le fichier de description et regrouper par classes
  img_list = []
  lbl_list = []
  path = Path(ds_description_path)
  with path.open('r') as file:
    for line in file.readlines():
      if line.endswith("\n"):
        line = line[:-1]
      splits = line.split("\t")

      if line != "":
        img_list.append(splits[0])
        lbl_list.append(int(splits[1]))

  print("dataset:", len(img_list))

  ## TODO : Comment : The random_state won't change a thing since our label list was written in arbitrary order
  ## Except if we use the same description file.

  if doVal:
    # Stratified Split into Train and Test dataset
    X_train, X_test, y_train, y_test = train_test_split(
        img_list, lbl_list, test_size=val_prop+test_prop, random_state=42, stratify=lbl_list)

    # Calculate the val proportion to the test proportion
    val_prop_to_test = val_prop/(val_prop+test_prop)

    # Stratified Split into Train and Validation dataset
    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, test_size=val_prop_to_test, random_state=42, stratify=y_test)

    print("train:", len(X_train), "test:", len(X_test), "val:", len(X_val))
    return X_train, y_train, X_test, y_test, X_val, y_val

  else:
    # Stratified Split into Train and Test dataset
    X_train, X_test, y_train, y_test = train_test_split(
        img_list, lbl_list, test_size=test_prop, random_state=42, stratify=lbl_list)
    
    print("train:", len(X_train), "test:", len(X_test))
    return X_train, y_train, X_test, y_test


##############
#LOAD DATASET#
##############
"""
DATASET = 0 --> CIFAR-10 dataset
DATASET = 1 --> DOG_CAT dataset
DATASET = 2 --> DIATOM dataset
"""
DATASET = 2
# 127  #64 --> GPU3  #256  # 512 --> Reduce to 256 if running on a single GPU.
batch_size = 512


if(DATASET == 0):
   print_banner("LOAD DATASET : CIFAR-10")

   dataset = 'cifar10'

   # Note the datasets are configured in input_pipeline.DATASET_PRESETS
   # Have a look in the editor at the right.
   num_classes = input_pipeline.get_dataset_info(dataset, 'train')[
       'num_classes']
   # tf.data.Datset for training, infinite repeats.
   ds_train = input_pipeline.get_data(
       dataset=dataset, mode='train', repeats=None, batch_size=batch_size,
   )
   # tf.data.Datset for evaluation, single repeat.
   ds_test = input_pipeline.get_data(
       dataset=dataset, mode='test', repeats=1, batch_size=batch_size,
   )

elif(DATASET == 1):
  print_banner("LOAD DATASET : CATS and DOGS")

  num_devices = len(jax.local_devices())

  # The bypass
  num_classes = 2
  dataset = 'dogscats'

  X_train, y_train, X_test, y_test = make_split_from_descfile(
      ds_description_path='dataset/diatom_dataset/description.txt',
      dataset_path='dataset/diatom_dataset',
      train_prop=0.8, test_prop=0.2, doVal=False)

  dgscts_train = MyDogsCats(dataset_path='dataset/CatsAndDogs',
                            X=X_train, y=y_train, num_class=num_classes, set_type='train', doDataAugmentation=False)

  dgscts_test = MyDogsCats(dataset_path='dataset/CatsAndDogs',
                           X=X_test, y=y_test, num_class=num_classes, set_type='test', doDataAugmentation=False)

  # dgscts_train = MyDogsCats(ds_description_path='dataset/CatsAndDogs/description.txt',
  #                 dataset_path='dataset/CatsAndDogs',
  #                 set_type='train',
  #                 train_prop=0.8)
  # dgscts_test = MyDogsCats(ds_description_path='dataset/CatsAndDogs/description.txt',
  #                 dataset_path='dataset/CatsAndDogs',
  #                 set_type='test',
  #                 train_prop=0.8)

  ds_train = dgscts_train.getDataset().batch(batch_size, drop_remainder=True)
  ds_test = dgscts_test.getDataset().batch(batch_size, drop_remainder=True)

  if num_devices is not None:
    ds_train = ds_train.map(_shard, tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.map(_shard, tf.data.experimental.AUTOTUNE)

  ds_test = ds_test.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  ds_train = ds_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

else:
  print_banner("LOAD DATASET : DIATOM")

  num_devices = len(jax.local_devices())

  # The bypass
  num_classes = 166
  dataset = 'diatom'

  X_train, y_train, X_test, y_test, X_val, y_val = make_split_from_descfile(
      ds_description_path='dataset/diatom_dataset/description.txt',
      dataset_path='dataset/diatom_dataset',
      train_prop=0.7, test_prop=0.2, val_prop=0.1, doVal=True)

  dgscts_train = MyDogsCats(dataset_path='dataset/diatom_dataset',
                            X=X_train, y=y_train, num_class=num_classes, set_type='train', doDataAugmentation=True)

  dgscts_test = MyDogsCats(dataset_path='dataset/diatom_dataset',
                           X=X_test, y=y_test, num_class=num_classes, set_type='test', doDataAugmentation=False)

  dgscts_val = MyDogsCats(dataset_path='dataset/diatom_dataset',
                          X=X_val, y=y_val, num_class=num_classes, set_type='val', doDataAugmentation=False)

  # dgscts_train = MyDogsCats(ds_description_path='dataset/diatom_dataset/description.txt',
  #                 dataset_path='dataset/diatom_dataset',
  #                 set_type='train', #train
  #                 train_prop=0.8,
  #                 doDataAugmentation=True)
  # dgscts_test = MyDogsCats(ds_description_path='dataset/diatom_dataset/description.txt',
  #                 dataset_path='dataset/diatom_dataset',
  #                 set_type='test',
  #                 train_prop=0.8,
  #                 doDataAugmentation=False)

  print("dgscts_train : ", dgscts_train.getDataset())

  ds_train = dgscts_train.getDataset().batch(batch_size, drop_remainder=True)
  ds_test = dgscts_test.getDataset().batch(batch_size, drop_remainder=True)
  ds_val = dgscts_val.getDataset().batch(batch_size, drop_remainder=True)

  ds_train = ds_train.repeat()
  ds_test = ds_test.repeat()
  ds_val = ds_val.repeat()

  if num_devices is not None:
    ds_train = ds_train.map(_shard, tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.map(_shard, tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.map(_shard, tf.data.experimental.AUTOTUNE)

  ds_test = ds_test.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  ds_train = ds_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  ds_val = ds_val.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

  print("ds_train : ", ds_train)
  print("ds_test : ", ds_test)
  print("ds_val : ", ds_val)

# Fetch a batch of test images for illustration purposes.
batch = next(iter(ds_test.as_numpy_iterator()))
# Note the shape : [num_local_devices, local_batch_size, h, w, c]

try:
   batch = next(iter(ds_train.as_numpy_iterator()))
except tf.errors.OutOfRangeError:
   print("End of dataset")  # ==> "End of dataset


########################
#LOAD PRE-TRAINED MODEL#
########################

print_banner("LOAD PRE-TRAINED MODEL")

# Load model definition & initialize random parameters.
VisionTransformer = models.KNOWN_MODELS[model].partial(num_classes=num_classes)
_, params = VisionTransformer.init_by_shape(
    jax.random.PRNGKey(0),
    # Discard the "num_local_devices" dimension of the batch for initialization.
    [(batch['image'].shape[1:], batch['image'].dtype.name)])


# Load and convert pretrained checkpoint.
# This involves loading the actual pre-trained model results, but then also also
# modifying the parameters a bit, e.g. changing the final layers, and resizing
# the positional embeddings.
# For details, refer to the code and to the methods of the paper.
params = checkpoint.load_pretrained(
    pretrained_path=f'{model}.npz',
    init_params=params,
    model_config=models.CONFIGS[model],
    logger=logger,
)


################################################
#EVALUATE PRE-TRAINED MODEL BEFORE FINE-TUNNING#
################################################

print_banner("EVALUATE PRE-TRAINED MODEL BEFORE FINE-TUNNING")

# So far, all our data is in the host memory. Let's now replicate the arrays
# into the devices.
# This will make every array in the pytree params become a ShardedDeviceArray
# that has the same data replicated across all local devices.
# For TPU it replicates the params in every core.
# For a single GPU this simply moves the data onto the device.
# For CPU it simply creates a copy.
params_repl = flax.jax_utils.replicate(params)
print('params.cls:', type(params['cls']).__name__, params['cls'].shape)
print('params_repl.cls:', type(params_repl['cls']).__name__, params_repl['cls'].shape)

# Then map the call to our model's forward pass into all available devices.
vit_apply_repl = jax.pmap(VisionTransformer.call)

# Random performance without fine-tuning.
if 0:
  acc = get_accuracy(params_repl)
  print("Accuracy of the pre-trained model before fine-tunning : ", acc)

###########
#FINE-TUNE#
###########

if FINE_TUNE:

  print_banner("FINE-TUNE")

  # 100 Steps take approximately 15 minutes in the TPU runtime.
  epochs = 2  # 600
  total_steps = (dgscts_train.get_num_samples()//batch_size) * epochs  # 300
  print("Total nbr backward steps : ", total_steps)
  print("Total nbr epochs : ", epochs)
  print("Nbr of train samples :", dgscts_train.get_num_samples())
  print("Batch Size : ", batch_size)
  warmup_steps = 5
  decay_type = 'cosine'
  grad_norm_clip = 1
  # This controls in how many forward passes the batch is split. 8 works well with
  # a TPU runtime that has 8 devices. 64 should work on a GPU. You can of course
  # also adjust the batch_size above, but that would require you to adjust the
  # learning rate accordingly.
  accum_steps = 64  # 64--> GPU3  #8--> TPU
  base_lr = 0.03

  # Check out train.make_update_fn in the editor on the right side for details.
  update_fn_repl = train.make_update_fn(VisionTransformer.call, accum_steps)
  # We use a momentum optimizer that uses half precision for state to save
  # memory. It als implements the gradient clipping.
  opt = momentum_clip.Optimizer(grad_norm_clip=grad_norm_clip).create(params)
  opt_repl = flax.jax_utils.replicate(opt)

  lr_fn = hyper.create_learning_rate_schedule(
      total_steps, base_lr, decay_type, warmup_steps)
  # Prefetch entire learning rate schedule onto devices. Otherwise we would have
  # a slow transfer from host to devices in every step.
  lr_iter = hyper.lr_prefetch_iter(lr_fn, 0, total_steps)
  # Initialize PRNGs for dropout.
  update_rngs = jax.random.split(
      jax.random.PRNGKey(0), jax.local_device_count())

  # Training loop.
  Loss_list = []
  loss_val_list = []
  accuracy_val_list = []
  accuracy_train_list = []
  val_eval_every = 1
  progress_every = 1
  t0 = time.time()

  for step, batch, batch_val, lr_repl in zip(
      tqdm.trange(1, total_steps + 1),
      ds_train.as_numpy_iterator(),
      ds_val.as_numpy_iterator(),
      lr_iter
  ):

    #Training step : update weights
    opt_repl, loss_repl, update_rngs = update_fn_repl(
        opt_repl, lr_repl, batch, update_rngs)

    #Loss of validation set 
    # for batch in ds_val.as_numpy_iterator():
    #   val_loss = loss_fn(opt_repl.target, batch['image'], batch['label'])
    #   loss_val_list.append(val_loss)
    #   print("val_loss : ",val_loss)

    val_loss = loss_fn(opt_repl.target, update_rngs, batch_val['image'], batch_val['label'])
    print("val_loss : ",val_loss)


    if step == 1:
      logger.info(f'First step took {time.time() - t0:.1f} seconds.')
      t0 = time.time()

    if progress_every and step % progress_every == 0:
      done = step / total_steps
      #logger.info(f'Step: {step}/{total_steps} {100*done:.1f}%, '
      #            f'ETA: {(time.time()-t0)/done*(1-done)/3600:.2f}h')

    # Validation set evaluation
    if ((val_eval_every and step % val_eval_every == 0) or
            (step == total_steps)):

      #Get validation accuracy
      nbr_val_samples = dgscts_val.get_num_samples()
      accuracy_val = get_accuracy_val(opt_repl.target, nbr_val_samples)

      #Get training accuracy
      nbr_train_samples = dgscts_train.get_num_samples()
      accuracy_train = get_accuracy_train(opt_repl.target, nbr_train_samples, batch)

      lr = float(lr_repl[0])
      #logger.info(f'Step: {step} '
      #            f'Learning rate: {lr:.7f}, '
      #            f'Validation accuracy: {accuracy_val:0.5f}'
      #            f'Training accuracy: {accuracy_train:0.5f}'
      #            f'Training Loss: {loss_repl:0.5f}'
      #            f'Validation Loss: {val_loss:0.5f}')

    #Store Loss calculate for each trainig step
    Loss_list.append(loss_repl)
    loss_val_list.append(val_loss)

    #Store Accuracy calculate for each trainig step
    accuracy_val_list.append(accuracy_val)
    accuracy_train_list.append(accuracy_train)

    #save weights every 1000 training steps
    if(step % 1000 == 0):
       checkpoint.save(flax_utils.unreplicate(opt_repl.target),
                       f"../models/model_diatom_checkpoint_step_{step}_with_data_aug.npz")



  steps_per_epoch = dgscts_train.get_num_samples()//batch_size

  #Plot Loss per epochs
  plot_learning_curve_per_epochs(Loss_list, loss_val_list, steps_per_epoch, total_steps)

  #Plot Val per epochs
  plot_acc_per_epochs(accuracy_train_list, accuracy_val_list, steps_per_epoch, total_steps)

  #Evaluate test accuracy
  if 1:
    nbr_samples = dgscts_val.get_num_samples() 
    acc = get_accuracy(opt_repl.target, nbr_samples)
    print("Accuracy of the pre-trained model after fine-tunning", acc)
    f = open("acc_log.txt", "w")
    f.write(f"Accuracy of the pre-trained model after fine-tunning : {acc}")
    f.close()

  print("Save Checkpoints :")
  checkpoint.save(flax_utils.unreplicate(opt_repl.target),
                  "../models/model_diatom_final_checkpoints_with_data_aug.npz")


if CHECKPOINTS_TEST:
  print_banner("CHECKPOINTS_TEST")

  # Load model definition & initialize random parameters.
  VisionTransformer = models.KNOWN_MODELS[model].partial(
      num_classes=num_classes)
  _, params = VisionTransformer.init_by_shape(
      jax.random.PRNGKey(0),
      # Discard the "num_local_devices" dimension of the batch for initialization.
      [(batch['image'].shape[1:], batch['image'].dtype.name)])

  #checkpoints_file_path = "../models/model_diatom_final_checkpoints_with_data_aug.npz"
  checkpoints_file_path = "../models/model_diatom_final_checkpoints.npz"
  params = checkpoint.load_pretrained_after_fine_tuning(
      pretrained_path=checkpoints_file_path,
      init_params=params,
      model_config=models.CONFIGS[model],
      logger=logger,
  )

  print('params.cls:', type(params['cls']).__name__, params['cls'].shape)
  print('params_repl.cls:', type(
      params_repl['cls']).__name__, params_repl['cls'].shape)

  # Then map the call to our model's forward pass into all available devices.
  vit_apply_repl = jax.pmap(VisionTransformer.call)

  params_repl = flax.jax_utils.replicate(params)
  #acc = get_accuracy(params_repl)
  #print("Accuracy of the pre-trained model after fine-tunning", acc)

  #Confusion Matrix
  predicted, labels = get_predict_labels_on_test_data(params_repl)
  predicted = np.array(predicted).flatten()
  labels = np.array(labels).flatten()
  print("Predicted Shape : ", predicted.shape, "labels shape : ", labels.shape)
  confusion_matrix = tf.math.confusion_matrix(labels, predicted)

  print(confusion_matrix)

  plot_confusion_matrix(
      confusion_matrix, "Confusion Matrix Diatom", "confusion_matrix.png")


###########
#INFERENCE#
###########

if INFERENCE:
  print_banner("INFERENCE")

  #VisionTransformer = models.KNOWN_MODELS[model].partial(num_classes=1000)
  VisionTransformer = models.KNOWN_MODELS[model].partial(num_classes=166)  # 2

  # Load and convert pretrained checkpoint.
  #params = checkpoint.load(f'{model}_imagenet2012.npz')
  params = checkpoint.load('../models/model_diatom_final_checkpoints.npz')
  params['pre_logits'] = {}  # Need to restore empty leaf for Flax.

  # Get imagenet labels.
  imagenet_labels = dict(enumerate(open('ilsvrc2012_wordnet_lemmas.txt')))

  # Get a random picture with the correct dimensions.
  #img = PIL.Image.open('picsum_1.jpg')

  #Get cat or dog image
  #img = PIL.Image.open('pic_dog.jpg')
  img = PIL.Image.open('BRG_UULN_10750.png')
  img = img.resize((384, 384))

  # Predict on a batch with a single item
  logits, = VisionTransformer.call(
      params, (np.array(img) / 128 - 1)[None, ...])

  preds = flax.nn.softmax(logits)
  # for idx in preds.argsort()[:-11:-1]:
  #   print(f'{preds[idx]:.5f} : {imagenet_labels[idx]}', end='')

  print("Predictions : ", preds)
  for idx in preds.argsort()[:-11:-1]:
     print("Class ID : ", idx, "Proba : ", preds[idx])

  #print("airplane , automobile, bird, cat, deer, dog, frog, horse, ship, truck")
  #print("classes : dog, cat ")
