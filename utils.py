import tensorflow as tf

class lr_decay(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, initial_learning_rate, epoch, epoch_deccay):
        super(lr_decay, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.epoch = epoch
        self.epoch_decay = epoch_deccay
        self.current_learning_rate = tf.Variable(initial_value=initial_learning_rate, trainable=False,dtype=tf.float32)

    def __call__(self, current_epoch):
        self.current_learning_rate.assign(tf.cond(
            current_epoch >= self.epoch_decay,
            true_fn=lambda: self.initial_learning_rate *
                            (1 - (current_epoch - self.epoch_decay) / (self.epoch - self.epoch_decay)),
            false_fn=lambda: self.initial_learning_rate
        ))
        return self.current_learning_rate