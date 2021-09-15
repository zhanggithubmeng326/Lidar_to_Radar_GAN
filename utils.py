import matplotlib.pyplot as plt
import tensorflow as tf
import yaml
import argparse


# create argparse
def get_args_parser():
    parser = argparse.ArgumentParser(description='model training')
    parser.add_argument('--config', type=str, default='config.yaml', help='config file path')
    return parser


# read config file which contains hyperparameter settings, returns configuration as dictionary
def get_config(config_file='config.yaml'):

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


class lr_decay(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, initial_learning_rate, epoch, epoch_deccay):
        super(lr_decay, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.epoch = epoch
        self.epoch_decay = epoch_deccay
        self.current_learning_rate = tf.Variable(initial_value=initial_learning_rate, trainable=False, dtype=tf.float32)

    def __call__(self, current_epoch):
        self.current_learning_rate.assign(tf.cond(
            current_epoch >= self.epoch_decay,
            true_fn=lambda: self.initial_learning_rate *
                            (1 - (current_epoch - self.epoch_decay) / (self.epoch - self.epoch_decay)),
            false_fn=lambda: self.initial_learning_rate
        ))
        return self.current_learning_rate


def generate_image(model, test_image, target, epoch):
    prediction = model(test_image, training=False)

    plt.figure(figsize=(30, 30))
    display_list = [test_image[0], target[0], prediction[0]]
    title = ['input image', 'ground truth', 'predicted image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')

    plt.savefig('image_at_epoch_{}.png'.format(epoch + 1))
    plt.show()
