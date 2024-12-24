from typing import List, Tuple

import tensorflow as tf

def conv_block(filters, filters1=None):
    """
    Convolutional block composed of two convolutional layers with Batch Normalization and ReLU activation between them.
    
    Parameters
    ----------
    filters : int
        Number of filters of the first convolutional layer
    filters1 : int, optional
        Number of filters of the second convolutional layer. If None, the number of filters will be the same as the first layer.
    
    Returns
    -------
    _f : function
        Convolutional block
    """
    def _f(x):
        f = filters
        x = tf.keras.layers.Conv2D(f, kernel_size=3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        if filters1 is not None:
            f = filters1
        x = tf.keras.layers.Conv2D(f, kernel_size=3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        return x
    return _f

def dn_block(k=2):
    """
    Downsampling block composed of a Max Pooling 2D layer with a default kernel size of 2.
    
    Parameters
    ----------
    k : int, optional
        Kernel size of the Max Pooling layer. Default is 2.
    
    Returns
    -------
    _f : function
        Downsampling block
    """
    def _f(x):
        x = tf.keras.layers.MaxPool2D(k)(x)
        return x
    return _f

def up_block(k=2, filters=64):
    """
    Upsampling block composed of an UpSampling2D layer followed by a convolutional layer with Batch Normalization and ReLU activation.
    
    Parameters
    ----------
    k : int, optional
        Upsampling size for the UpSampling2D layer. Default is 2.
    filters : int
        Number of filters for the convolutional layer.
    
    Returns
    -------
    _f : function
        Upsampling block
    """
    def _f(x):
        x = tf.keras.layers.UpSampling2D(size=k, interpolation="bilinear")(x)
        x = tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        return x
    return _f

def generate_ds_encoder(
        x: tf.Tensor,
        init_filters: int = 128
    ) -> Tuple[tf.Tensor, List[tf.Tensor]]:
    """
    Generates a downsampling encoder path for a U-Net style architecture using convolutional and downsampling blocks.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor to the encoder network.
    init_filters : int, optional
        Initial number of filters for the convolutional blocks. Default is 128.

    Returns
    -------
    inputs : tf.Tensor
        The original input tensor.
    List[tf.Tensor]
        A list of tensors representing the output of each block in the encoder path.
    """
    inputs = x

    k = 1
    x = conv_block(init_filters * k)(x)
    b1 = x
    x = dn_block(2)(x)

    k = 2
    x = conv_block(init_filters * k)(x)
    b2 = x
    x = dn_block(2)(x)

    k = 4
    x = conv_block(init_filters * k)(x)
    b3 = x
    x = dn_block(2)(x)

    k = 8
    x = conv_block(init_filters * k)(x)
    b4 = x
    x = dn_block(2)(x)

    k = 8
    x = conv_block(init_filters * k)(x)
    outputs = x
    
    return inputs, [b1, b2, b3, b4, outputs]

def generate_ds_decoder(
        xinputs: List[tf.Tensor],
        init_filters: int = 128
    ) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Builds the decoder part of the U-Net with the given initial number of filters,
    and with the given list of input tensors from the encoder path.

    Parameters
    ----------
    xinputs : List[Tensor]
        A list of tensors representing the output of each block in the encoder path.
    init_filters : int, optional
        Initial number of filters for the convolutional blocks. Default is 128.

    Returns
    -------
    inputs : Tensor
        The original input tensor.
    outputs : Tensor
        The output tensor of the decoder.
    """
    x = xinputs.pop()
    inputs = x

    k = 8
    k1 = 8
    x = up_block(2, init_filters * k)(x)
    xs = xinputs.pop()
    x = tf.keras.layers.Concatenate(axis=-1)([x, xs])
    x = conv_block(init_filters * k, filters1=(init_filters * k1))(x)

    k = 4
    k1 = 4
    x = up_block(2, init_filters * k)(x)
    xs = xinputs.pop()
    x = tf.keras.layers.Concatenate(axis=-1)([x, xs])
    x = conv_block(init_filters * k, filters1=(init_filters * k1))(x)

    k = 2
    k1 = 2
    x = up_block(2, init_filters * k)(x)
    xs = xinputs.pop()
    x = tf.keras.layers.Concatenate(axis=-1)([x, xs])
    x = conv_block(init_filters * k, filters1=(init_filters * k1))(x)

    k = 1
    k1 = 1
    x = up_block(2, init_filters * k)(x)
    xs = xinputs.pop()
    x = tf.keras.layers.Concatenate(axis=-1)([x, xs])
    x = conv_block(init_filters * k, filters1=(init_filters * k1))(x)

    x = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same')(x)
    x = tf.keras.layers.Activation(tf.nn.sigmoid)(x)
    outputs = x

    return inputs, outputs

def build_segmentator(
        image_size: int,
        channels: int,
        init_filters: int = 128,
        verbose: bool = True
    ) -> tf.keras.models.Model:
    """
    Builds a U-Net style segmentation network.

    Parameters
    ----------
    image_size : int
        Size of the input images.
    channels : int
        Number of channels in the input images.
    init_filters : int, optional
        Initial number of filters for the convolutional blocks. Default is 128.
    verbose : bool, optional
        If True, prints the model summary. Default is True.

    Returns
    -------
    model : tf.keras.models.Model
        The U-Net style segmentation network model.
    """
    shape: Tuple[int, int, int] = (image_size, image_size, channels)
    inputs = tf.keras.layers.Input(shape=shape, name='input')
    x = inputs
    _, den0_outputs = generate_ds_encoder(x, init_filters=init_filters)
    _, dde0_outputs = generate_ds_decoder(den0_outputs, init_filters=init_filters)

    segmentator_model = tf.keras.models.Model(inputs=inputs, outputs=dde0_outputs, name='Segmentator')

    if verbose:
        # Print model summaries
        segmentator_model.summary()

        # Plot the autoencoder model architecture
        tf.keras.utils.plot_model(
            segmentator_model,
            to_file='/tmp/segmentator.png',
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            expand_nested=True,
            show_layer_activations=True,
            show_trainable=True,
        )

    return segmentator_model
