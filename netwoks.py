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
        model.add(layers.Conv2D(filters=channels_out, kernel_size=3, kernel_initializer= kernel_initializer))
        model.add(tfa.layers.InstanceNormalization(axis=-1, center=False, scale=False))

        model.add(layers.ReLU())

        model.add(layers.ZeroPadding2D(1))
        model.add(layers.Conv2D(filters=channels_out, kernel_size=3, kernel_initializer=kernel_initializer))
        model.add(tfa.layers.InstanceNormalization(axis=-1, center=False, scale=False))   # check the value of the axis

        self.model = model

    def call(self, x, **kwargs):         # strange call here ???

        out = x + self.model(x)
        return out


# build global generator for lower resolution
class GlobalGenerator(tf.keras.Model):

    def __init__(self, input_nc=3, output_nc=3, base_channels=64, n_layers=3, residual_blocks=9):
        super(GlobalGenerator, self).__init__()
        # self.output_nc = output_nc

        model_1 = tf.keras.Sequential(name='model_1')
        model_1.add(layers.InputLayer([800, 800, input_nc]))
        model_1.add(layers.ZeroPadding2D(3))
        model_1.add(layers.Conv2D(filters=base_channels, kernel_size=7, kernel_initializer=kernel_initializer))
        model_1.add(tfa.layers.InstanceNormalization(axis=-1, center=False, scale=False))
        model_1.add(layers.ReLU())

        channels = base_channels

        # frontend blocks for downsampling
        for _ in range(n_layers):

            model_1.add(layers.Conv2D(filters=channels * 2, kernel_size=3, strides=2,
                                    padding='same', kernel_initializer=kernel_initializer))
            model_1.add(tfa.layers.InstanceNormalization(axis=-1, center=False, scale=False))
            model_1.add(layers.ReLU())

            channels = channels * 2

        # residual blocks
        for _ in range(residual_blocks):
            model_1.add(ResidualBlock(channels))

        # backend blocks for transposed convolution
        for _ in range(n_layers):
            model_1.add(layers.Conv2DTranspose(filters=channels/2, kernel_size=3, strides=2, padding='same',
                                             output_padding=1, kernel_initializer=kernel_initializer))
            model_1.add(tfa.layers.InstanceNormalization(axis=-1, center=False, scale=False))
            model_1.add(layers.ReLU())
            channels = channels // 2

        self.model_1 = model_1

        # last layer for output
        model_2 = tf.keras.Sequential(name='model_2')
        #model_2.add(layers.InputLayer([None, None, channels]))  # channels=???
        model_2.add(layers.ZeroPadding2D(3))
        model_2.add(layers.Conv2D(filters=output_nc, kernel_size=7, kernel_initializer=kernel_initializer, activation='tanh'))
        #model_2.add(tf.keras.activations.tanh())
        
        self.model_2 = model_2

    def call(self, x, training=True, mask=None):
        temp = self.model_1(x)
        output = self.model_2(temp)

        return output


# build a local enhancer for higher resolution images
class LocalEnhancer(tf.keras.Model):

    def __init__(self, channels_in=3, channels_out=3, base_channels=32,
                 global_fb_blocks=3, global_residual_blocks=9, local_residual_blocks=3):
        super(LocalEnhancer, self).__init__()

        self.conv_for_downsampling = layers.Conv2D(filters=3, kernel_size=7, padding='same', kernel_initializer=kernel_initializer)
        # downsampling the high resolution images to low resolution images
        self.downsample = layers.AveragePooling2D(3, strides=2, padding='same')

        # initialize global generator without last 3 layers
        global_base_channels = base_channels * 2
        model_global = GlobalGenerator(channels_in, channels_out, global_base_channels,
                                       global_fb_blocks, global_residual_blocks).model_1
        #model_global = model_global[:-3]                # why can we use List here???  type of model_global
        #self.model_G1 = tf.keras.Sequential(model_global)
        self.model_global = model_global

        # local enhancer layers, downsampling
        model_downsampling = tf.keras.Sequential(name='model_downsampling')

        model_downsampling.add(layers.ZeroPadding2D(3))
        model_downsampling.add(layers.Conv2D(filters=base_channels, kernel_size=7, kernel_initializer=kernel_initializer))
        model_downsampling.add(tfa.layers.InstanceNormalization(axis=-1, center=False, scale=False))
        model_downsampling.add(layers.ReLU())

        model_downsampling.add(layers.Conv2D(filters=base_channels * 2, kernel_size=3, strides=2, padding='same',
                               kernel_initializer=kernel_initializer))
        model_downsampling.add(tfa.layers.InstanceNormalization(axis=-1, center=False, scale=False))
        model_downsampling.add(layers.ReLU())

        self.model_downsampling = model_downsampling

        # residual blocks
        model_upsampling = tf.keras.Sequential(name='model_upsampling')         # always give input_shape firstly???
        for _ in range(local_residual_blocks):
            model_upsampling.add(ResidualBlock(channels_out=base_channels * 2))

        # upsampling
        model_upsampling.add(layers.Conv2DTranspose(filters=base_channels, kernel_size=3, strides=2, padding='same',
                                                    kernel_initializer=kernel_initializer))
        model_upsampling.add(tfa.layers.InstanceNormalization(axis=-1, center=False, scale=False))
        model_upsampling.add(layers.ReLU())

        # convolution for output
        model_upsampling.add(layers.ZeroPadding2D(3))
        model_upsampling.add(layers.Conv2D(filters=channels_out, kernel_size=7, kernel_initializer=kernel_initializer, activation='tanh'))
        #model_upsampling.add(tf.keras.activations.tanh())
        self.model_upsampling = model_upsampling

    def call(self, x, training=None, mask=None):         # does forward work here???
        conv_input = self.conv_for_downsampling(x)
        downsampled_input = self.downsample(conv_input)
        input_after_GG = self.model_global(downsampled_input)
        output = self.model_upsampling(input_after_GG + self.model_downsampling(x))

        return output


# build a PatchGAN discriminator, which can be used by different scales
class Discriminator(tf.keras.Model):

    def __init__(self, input_nc, base_channels=64):
        super(Discriminator, self).__init__()

        self.model_list = []

        layer1 = [layers.InputLayer([None, None, input_nc]),
                  layers.Conv2D(filters=base_channels, kernel_size=4, strides=2, padding='same', kernel_initializer=kernel_initializer),
                  layers.LeakyReLU(0.2)]
        model_1 = tf.keras.Sequential(layer1)
        self.model_list.append(model_1)

        # downsampling convolutional layer
        layer2 = [layers.Conv2D(filters=2 * base_channels, kernel_size=4, strides=2, padding='same', kernel_initializer=kernel_initializer),
                  tfa.layers.InstanceNormalization(axis=-1, center=False, scale=False),
                  layers.LeakyReLU(0.2)]
        model_2 = tf.keras.Sequential(layer2)
        self.model_list.append(model_2)

        # downsampling convolutional layer
        layer3 = [layers.Conv2D(filters=4 * base_channels, kernel_size=4, strides=2, padding='same', kernel_initializer=kernel_initializer),
                  tfa.layers.InstanceNormalization(axis=-1, center=False, scale=False),
                  layers.LeakyReLU(0.2)]
        model_3 = tf.keras.Sequential(layer3)
        self.model_list.append(model_3)

        # output convolutional layer
        layer4 = [layers.Conv2D(filters=8 * base_channels, kernel_size=4, strides=1, padding='same', kernel_initializer=kernel_initializer),
                  tfa.layers.InstanceNormalization(axis=-1, center=False, scale=False),
                  layers.LeakyReLU(0.2),
                  layers.Conv2D(filters=1, kernel_size=4, strides=1, padding='same', kernel_initializer=kernel_initializer)]
        model_4 = tf.keras.Sequential(layer4)
        self.model_list.append(model_4)

        #for i in range(len(self.model_list)):
            #setattr(self, 'layer'+str(i+1), self.model_list[i])

    def call(self, x, training=True, mask=None):      # training = True ????
        layer_results = []
        for layer in self.model_list:
            x = layer(x)
            layer_results.append(x)

        return layer_results

# build a multi-scale discriminator for three different image resolutions
class MultiscaleDiscriminator(tf.keras.Model):

    def __init__(self, input_nc=6, n_discrimintors=3):
        super(MultiscaleDiscriminator, self).__init__()

        # initialize all three discriminators
        self.list_discriminator = []
        for dis in range(n_discrimintors):
            self.list_discriminator.append(Discriminator(input_nc))  # instantiation???

        self.downsample = layers.AveragePooling2D(3, strides=2, padding='same')

    def call(self, inputs, training=True, mask=None):            # training = ???
        results = []

        for n, discriminator in enumerate(self.list_discriminator):
            if n != 0:
                inputs = self.downsample(inputs)

            results.append(discriminator(inputs))  # list which contains multi-scale discriminator outputs

        return results
