''' A tensorflow pipeline example for training by image list '''

# the code is modified from tensorflow/model/official/resnet/cifar10_main.py
# the example dataset is cifar10, and the example model is resnet
# you can easily replace the dataset or model by you own
# the code is tested on tensorflow-gpu 1.3

import argparse
import os

import tensorflow as tf
import json

from model_fn import *
from input_fn import *

parser = argparse.ArgumentParser()

# Parameters.
parser.add_argument('--config_file', type=str, default='',
                    help='The path to the config file.')

Argv = parser.parse_args()


# Load config file
assert os.path.exists(Argv.config_file) ,(
    "config_file " + Argv.config_file + ' does not exist !')

f = open(Argv.config_file)
config_dict = json.load(f)

f.close()
#Convert dict config to class
class DictObj(object):
    def __init__(self,map):
        self.map = map

    def __setattr__(self, name, value):
        if name == 'map':
             object.__setattr__(self, name, value)
             return;
        print 'set attr called ',name,value
        self.map[name] = value

    def __getattr__(self,name):
        v = self.map[name]
        if isinstance(v,(dict)):
            return DictObj(v)
        if isinstance(v, (list)):
            r = []
            for i in v:
                r.append(DictObj(i))
            return r                      
        else:
            return self.map[name];

    def __getitem__(self,name):
        return self.map[name]

CONFIG = DictObj(config_dict)

def main(unused_argv):
    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    # Set up a RunConfig to only save checkpoints once per training cycle.
    run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9)

    cifar_classifier = tf.estimator.Estimator(
        model_fn=lambda features, labels, mode :cifar10_model_fn(CONFIG, features, labels, mode), 
        model_dir=CONFIG.train_cfg.model_dir, config=run_config)

    for _ in range(CONFIG.train_cfg.train_epochs // CONFIG.train_cfg.epochs_per_eval):
        tensors_to_log = {
            'learning_rate': 'learning_rate',
            'cross_entropy': 'cross_entropy',
            'train_accuracy': 'train_accuracy'
        }

        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=CONFIG.train_cfg.log_every_n_iter)

        cifar_classifier.train(
            input_fn=lambda: input_fn(
                config      = CONFIG, 
                is_training = True, 
                num_epochs  = CONFIG.train_cfg.epochs_per_eval),
                hooks       = [logging_hook])

        # Evaluate the model and print results
        eval_results = cifar_classifier.evaluate(
            input_fn=input_fn(config = CONFIG, is_training = False, num_epochs = 1))
        print(eval_results)


if __name__ == '__main__':
    pass
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
