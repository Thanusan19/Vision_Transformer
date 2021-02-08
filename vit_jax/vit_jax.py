import sys
import flax
import jax
from matplotlib import pyplot as plt
import numpy as np
import tqdm
import PIL

import checkpoint
import input_pipeline
import models
import momentum_clip
import train
import hyper
import logging_ViT
import flax.jax_utils as flax_utils


# Shows the number of available devices.
# In a CPU/GPU runtime this will be a single device.
jax.local_devices()

# Pre-trained model name
model = 'ViT-B_16'

logger = logging_ViT.setup_logger('./logs')
INFERENCE = True
FINE_TUNE = False


# Helper functions for images.

labelnames = dict(
  # https://www.cs.toronto.edu/~kriz/cifar.html
  cifar10=('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'),
  # https://www.cs.toronto.edu/~kriz/cifar.html
  cifar100=('apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'computer_keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm')
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

##############
#LOAD DATASET#
##############

print_banner("LOAD DATASET")

dataset = 'cifar10'
batch_size = 64  #256  # 512 --> Reduce to 256 if running on a single GPU.


# Note the datasets are configured in input_pipeline.DATASET_PRESETS
# Have a look in the editor at the right.
num_classes = input_pipeline.get_dataset_info(dataset, 'train')['num_classes']
# tf.data.Datset for training, infinite repeats.
ds_train = input_pipeline.get_data(
    dataset=dataset, mode='train', repeats=None, batch_size=batch_size,
)
# tf.data.Datset for evaluation, single repeat.
ds_test = input_pipeline.get_data(
    dataset=dataset, mode='test', repeats=1, batch_size=batch_size,
)


# Fetch a batch of test images for illustration purposes.
batch = next(iter(ds_test.as_numpy_iterator()))
# Note the shape : [num_local_devices, local_batch_size, h, w, c]
batch['image'].shape



# Show some imags with their labels.
images, labels = batch['image'][0][:9], batch['label'][0][:9]
titles = map(make_label_getter(dataset), labels.argmax(axis=1))
#show_img_grid(images, titles)


# Same as above, but with train images.
# Do you spot a difference?
# Check out input_pipeline.get_data() in the editor at your right to see how the
# images are preprocessed differently.
batch = next(iter(ds_train.as_numpy_iterator()))
images, labels = batch['image'][0][:9], batch['label'][0][:9]
titles = map(make_label_getter(dataset), labels.argmax(axis=1))
#show_img_grid(images, titles)



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


def get_accuracy(params_repl):
  """Returns accuracy evaluated on the test set."""
  good = total = 0
  steps = input_pipeline.get_dataset_info(dataset, 'test')['num_examples'] // batch_size
  for _, batch in zip(tqdm.trange(steps), ds_test.as_numpy_iterator()):
  #for _, batch in zip(steps, ds_test.as_numpy_iterator()):  
    predicted = vit_apply_repl(params_repl, batch['image'])
    is_same = predicted.argmax(axis=-1) == batch['label'].argmax(axis=-1)
    good += is_same.sum()
    total += len(is_same.flatten())
  return good / total


# Random performance without fine-tuning.
if 0:
  acc = get_accuracy(params_repl)
  print("Accuracy of the pre-trained model before fine-tunning : ", acc)

###########
#FINE-TUNE#
###########

if FINE_TUNE :

  print_banner("FINE-TUNE")

  # 100 Steps take approximately 15 minutes in the TPU runtime.
  total_steps = 2
  warmup_steps = 5
  decay_type = 'cosine'
  grad_norm_clip = 1
  # This controls in how many forward passes the batch is split. 8 works well with
  # a TPU runtime that has 8 devices. 64 should work on a GPU. You can of course
  # also adjust the batch_size above, but that would require you to adjust the
  # learning rate accordingly.
  accum_steps = 64 #8
  base_lr = 0.03


  # Check out train.make_update_fn in the editor on the right side for details.
  update_fn_repl = train.make_update_fn(VisionTransformer.call, accum_steps)
  # We use a momentum optimizer that uses half precision for state to save
  # memory. It als implements the gradient clipping.
  opt = momentum_clip.Optimizer(grad_norm_clip=grad_norm_clip).create(params)
  opt_repl = flax.jax_utils.replicate(opt)


  lr_fn = hyper.create_learning_rate_schedule(total_steps, base_lr, decay_type, warmup_steps)
  # Prefetch entire learning rate schedule onto devices. Otherwise we would have
  # a slow transfer from host to devices in every step.
  lr_iter = hyper.lr_prefetch_iter(lr_fn, 0, total_steps)
  # Initialize PRNGs for dropout.
  update_rngs = jax.random.split(jax.random.PRNGKey(0), jax.local_device_count())


  # The world's simplest training loop.
  # Completes in ~20 min on the TPU runtime.
  for step, batch, lr_repl in zip(
      tqdm.trange(1, total_steps + 1),
      ds_train.as_numpy_iterator(),
      lr_iter
  ):

    opt_repl, loss_repl, update_rngs = update_fn_repl(
        opt_repl, lr_repl, batch, update_rngs)

  if 1 :
    acc = get_accuracy(opt_repl.target)
    print("Accuracy of the pre-trained model after fine-tunning", acc)


  #print("Save Checkpoints :")
  #checkpoint.save(flax_utils.unreplicate(opt_repl.target), "/home/gpu_user/VisionTransformer/vision_transformer/models/")


###########
#INFERENCE#
###########

if INFERENCE :
  print_banner("INFERENCE")

  VisionTransformer = models.KNOWN_MODELS[model].partial(num_classes=1000)

  # Load and convert pretrained checkpoint.
  params = checkpoint.load(f'{model}_imagenet2012.npz')
  #params = opt_repl.target
  #params = checkpoint.load('../models/model_save_1.npz')
  params['pre_logits'] = {}  # Need to restore empty leaf for Flax.


  # Get imagenet labels.
  imagenet_labels = dict(enumerate(open('ilsvrc2012_wordnet_lemmas.txt')))


  # Get a random picture with the correct dimensions.
  img = PIL.Image.open('picsum.jpg')


  # Predict on a batch with a single item
  logits, = VisionTransformer.call(params, (np.array(img) / 128 - 1)[None, ...])


  preds = flax.nn.softmax(logits)
  for idx in preds.argsort()[:-11:-1]:
    print(f'{preds[idx]:.5f} : {imagenet_labels[idx]}', end='')
