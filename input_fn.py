
import tensorflow as tf
import os


def filenames(config, is_training):
    """Returns a list of filenames."""
    if is_training:
        txtlines = open(config.data_cfg.train_list, 'r').read().splitlines()
    else:
        txtlines = open(config.data_cfg.test_list, 'r').read().splitlines()

    filenames = [os.path.join(config.data_cfg.data_dir ,line.split(' ')[0]) 
        for line in txtlines]

    labels = [int(line.split(' ')[1]) for line in txtlines]

    return (filenames, labels)

def dataset_parser(config, filename, label):
    """Parse a data from filename and label."""
    image_string  = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_string)
    image_decoded.set_shape([None, None, config.data_cfg.image_channel])
    image_resized = tf.image.resize_images(image_decoded, 
        [config.data_cfg.image_height, config.data_cfg.image_width])
  
    return image_resized,tf.one_hot(label, config.data_cfg.class_number)

def train_preprocess_fn(config, image, label):
    """Preprocess a single training image of layout [height, width, depth]."""
    # Resize the image to add four extra pixels on each side.
    image = tf.image.resize_image_with_crop_or_pad(
        image, config.data_cfg.image_height + 8, config.data_cfg.image_width + 8)
  
    # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
    image = tf.random_crop(image, [config.data_cfg.image_height, 
        config.data_cfg.image_width, config.data_cfg.image_channel])
  
    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)
  
    return image, label
 
def input_fn(config, is_training, num_epochs=1):

   """Input_fn using the contrib.data input pipeline for dataset.
 
   Args:
     is_training: A boolean denoting whether the input is for training.
     num_epochs: The number of epochs to repeat the dataset.
 
   Returns:
     A tuple of images and labels.
   """
 
   dataset = tf.contrib.data.Dataset.from_tensor_slices(
             filenames(config, is_training))
 
   dataset = dataset.map(
             lambda filename, label : dataset_parser(config, filename, label), 
             num_threads = 1, output_buffer_size = 2 * config.train_cfg.batch_size)
 
   # For training, preprocess the image and shuffle.
   if is_training:
       dataset = dataset.map(lambda image, label : train_preprocess_fn(config, image, label), 
                             num_threads = 1, output_buffer_size = 2 * config.train_cfg.batch_size)
 
       # Ensure that the capacity is sufficiently large to provide good random
       # shuffling.
       buffer_size = int(0.4 * config.data_cfg.train_number)
       dataset = dataset.shuffle(buffer_size=buffer_size)
 
   # Subtract off the mean and divide by the variance of the pixels.
   dataset = dataset.map(
       lambda image, label: (tf.image.per_image_standardization(image), label),
       num_threads=1,
       output_buffer_size= 2 * config.train_cfg.batch_size)
 
   dataset = dataset.repeat(num_epochs)
 
   # Batch results by up to batch_size, and then fetch the tuple from the
   # iterator.
   iterator = dataset.batch(config.train_cfg.batch_size).make_one_shot_iterator()
   images, labels = iterator.get_next()
 
   return images, labels

