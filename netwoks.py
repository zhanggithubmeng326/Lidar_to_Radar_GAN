import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers

kernel_initializer = tf.random_normal_initializer(0.0, 0.02)

# build a residual block
class ResidualBlock(layers.Layer):

    def __init__(self, channels_out):
        super(ResidualBlock, self).__init__()

        model = tf.keras.Sequential()
        model.add(layers.ZeroPadding2D(1))
        model.add(layers.Conv2D(channels_out, kernel_size=3, kernel_initializer= kernel_initializer))
        model.add(tfa.layers.InstanceNormalization(axis=2, center=False, scale=False))

        model.add(layers.ReLU())

        model.add(layers.ZeroPadding2D(1))
        model.add(layers.Conv2D(channels_out, kernel_size=3, kernel_initializer=kernel_initializer))
        model.add(tfa.layers.InstanceNormalization(axis=-1, center=False, scale=False))   # check the value of the axis

        self.model = model

    def call(self, x, **kwargs):         # strange call here ???

        out = x + self.model(x)
        return out


# build global generator for lower resolution
class GlobalGenerator(tf.keras.Model):

    def __init__(self, input_nc, output_nc, base_channels=64, n_layers=3, residual_blocks=9):
        super(GlobalGenerator, self).__init__()
        self.output_nc = output_nc

        model = tf.keras.Sequential()
        model.add(layers.InputLayer([None, None, input_nc]))
        model.add(layers.ZeroPadding2D(3))
        model.add(layers.Conv2D(base_channels, kernel_size=7, kernel_initializer=kernel_initializer))
        model.add(tfa.layers.InstanceNormalization(axis=-1, center=False, scale=False))
        model.add(layers.ReLU())

        channels = base_channels
        # frontend blocks for downsampling
        for _ in range(n_layers):

            model.add(layers.Conv2D(channels * 2, kernel_size=3, strides=2,
                                    padding='same', kernel_initializer=kernel_initializer))
            model.add(tfa.layers.InstanceNormalization(axis=-1, center=False, scale=False))
            model.add(layers.ReLU())
            channels = channels * 2

        # residual blocks
        for _ in range(residual_blocks):
            model.add(ResidualBlock(channels))

        # backend blocks for transposed convolution
        for _ in range(n_layers):
            model.add(layers.Conv2DTranspose(channels/2, kernel_size=3, strides=2, padding='same',
                                             output_padding=1, kernel_initializer=kernel_initializer))
            model.add(tfa.layers.InstanceNormalization(axis=-1, center=False, scale=False))
            model.add(layers.ReLU())
            channels = channels / 2
        # last layer for output
        model.add(layers.ZeroPadding2D(3))
        model.add(layers.Conv2D(self.output_nc, kernel_size=7, kernel_initializer=kernel_initializer))
        model.add(tf.keras.activations.tanh())
        
        self.model = model

    def call(self, x, training=None, mask=None):       # training=???
        output = self.model(x)
        return output


# build a local enhancer for higher resolution images
class LocalEnhancer(tf.keras.Model):

    def __init__(self, channels_in, channels_out, base_channels=32,
                 global_fb_blocks=3, global_residual_blocks=9, local_residual_blocks=3):
        super(LocalEnhancer, self).__init__()

        # downsampling the high resolution images to low resolution images
        self.downsample = layers.GlobalAveragePooling2D(3, strides=2, padding='same')

        # initialize global generator without last 3 layers
        global_base_channels = base_channels * 2
        model_global = GlobalGenerator(channels_in, channels_out, global_base_channels,
                                       global_fb_blocks, global_residual_blocks)
        model_global = model_global[:-3]                # why can we use List here???  type of model_global
        self.model_G1 = tf.keras.Sequential(model_global)

        # local enhancer layers, downsampling
        model_downsampling = tf.keras.Sequential()

        model_downsampling.add(layers.ZeroPadding2D(3))
        model_downsampling.add(layers.Conv2D(base_channels * 2, kernel_size=7, kernel_initializer=kernel_initializer))
        model_downsampling.add(tfa.layers.InstanceNormalization(axis=-1, center=False, scale=False))
        model_downsampling.add(layers.ReLU())

        model_downsampling.add(layers.Conv2D(base_channels * 4, kernel_size=3, strides=2,
                               kernel_initializer=kernel_initializer))
        model_downsampling.add(tfa.layers.InstanceNormalization(axis=-1, center=False, scale=False))
        model_downsampling.add(layers.ReLU())

        self.model_downsampling = model_downsampling

        # residual blocks
        model_upsampling = tf.keras.Sequential()
        for _ in range(local_residual_blocks):
            model_upsampling.add(ResidualBlock(channels_out=base_channels * 4))

        # upsampling
        model_upsampling.add(layers.Conv2DTranspose(base_channels * 2, kernel_size=3, strides=2, padding='same',
                                                    kernel_initializer=kernel_initializer))
        model_upsampling.add(tfa.layers.InstanceNormalization(axis=-1, center=False, scale=False))
        model_upsampling.add(layers.ReLU())

        # convolution for output
        model_upsampling.add(layers.ZeroPadding2D(3))
        model_upsampling.add(layers.Conv2D(channels_out, kernel_size=7, kernel_initializer=kernel_initializer))
        model_upsampling.add(tf.keras.activations.tanh())
        self.model_upsampling = model_upsampling

    def call(self, x, training=None, mask=None):         # does forward work here???
        downsampled_input = self.downsample(x)
        input_after_G1 = self.model_G1(downsampled_input)
        output = self.model_upsampling(input_after_G1 + self.model_downsampling(x))

        return output

# build discriminator
class Discriminator(tf.keras.Model):

    def __init__(self, base_channels=64, n_layers=3):
        super(Discriminator, self).__init__()
