from typing import List, Tuple

import tensorflow as tf

def get_act(
    act: str,
    name: str
) -> tf.keras.layers.Layer:
    """
    Returns a Keras activation layer based on the provided activation type.

    Parameters
    ----------
    act : str
        The type of activation function to be applied, specified as a string.
        Supported values are 'lrelu', 'leakyrelu', 'relu', 'sigmoid', and 'tanh'.
    name : str
        The name to assign to the activation layer.

    Returns
    -------
    tf.keras.layers.Layer
        A Keras layer corresponding to the specified activation function.

    Raises
    ------
    ValueError
        If the provided activation type is not supported.
    """
    act = str(act).lower().strip()
    if act == 'lrelu' or act == 'leakyrelu':
        return tf.keras.layers.LeakyReLU(alpha=0.2, name=name)
    elif act == 'relu':
        return tf.keras.layers.ReLU(name=name)
    elif act == 'sigmoid':
        return tf.keras.layers.Activation(tf.nn.sigmoid, name=name)
    elif act == 'tanh':
        return tf.keras.layers.Activation(tf.nn.tanh, name=name)
    else:
        raise ValueError('act must be one of lrelu, leakyrelu, relu, sigmoid, tanh')

def res_block(
    inputs: tf.Tensor,
    filters: int,
    stage: int,
    block: int,
    strides: int,
    cut: str,
    encoder: bool,
    bias: bool,
    act: str,
    name: str,
    bn: bool,
    ks: int
) -> tf.Tensor:
    """
    Builds a residual block with optional Batch Normalization,
    activation functions, and a shortcut connection.

    Parameters
    ----------
    inputs : tf.Tensor
        The input tensor to the residual block.
    filters : int
        The number of filters for the convolutional layers.
    stage : int
        The stage index of the block.
    block : int
        The block index within the stage.
    strides : int
        The stride for the convolutional layers.
    cut : str
        Type of shortcut connection, either 'pre' or 'post'.
    encoder : bool
        If True, uses Conv2D layers; otherwise, uses Conv2DTranspose.
    bias : bool
        If True, adds a bias term to the convolutional layers.
    act : str
        The type of activation function to apply.
    name : str
        The name prefix for the layers.
    bn : bool
        If True, applies Batch Normalization.
    ks : int
        The kernel size for the convolutional layers.

    Returns
    -------
    tf.Tensor
        The output tensor of the residual block.
    """
    x = inputs

    # defining shortcut connection
    if cut == 'pre':
        shortcut = x
    elif cut == 'post':
        if encoder:
            shortcut = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=1,
                strides=strides,
                use_bias=bias,
                padding='same',
                name=f'{name}_stage{stage}_block{block}_scut_conv0'
            )(x)
        else:
            if strides == 1:
                shortcut = tf.keras.layers.Conv2DTranspose(
                    filters=filters,
                    kernel_size=1,
                    strides=strides,
                    use_bias=bias,
                    padding='same',
                    name=f'{name}_stage{stage}_block{block}_scut_conv0'
                )(x)
            else:
                shortcut = tf.keras.layers.Conv2DTranspose(
                    filters=filters // 2,
                    kernel_size=1,
                    strides=strides,
                    use_bias=bias,
                    padding='same',
                    name=f'{name}_stage{stage}_block{block}_scut_conv0'
                )(x)
        if bn:
            shortcut = tf.keras.layers.BatchNormalization(
                name=f'{name}_stage{stage}_block{block}_scut_bn1'
            )(shortcut)

        shortcut = get_act(
            act=act,
            name=f'{name}_stage{stage}_block{block}_scut_act_{act}1'
        )(shortcut)
    else:
        raise ValueError('Cut type not in ["pre", "post"]')

    # Sub-block 1
    if encoder:
        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=ks,
            strides=strides,
            use_bias=bias,
            padding='same',
            name=f'{name}_stage{stage}_block{block}_conv0'
        )(x)
    else:
        x = tf.keras.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=ks,
            strides=1,
            use_bias=bias,
            padding='same',
            name=f'{name}_stage{stage}_block{block}_conv0'
        )(x)

    if bn:
        x = tf.keras.layers.BatchNormalization(
            name=f'{name}_stage{stage}_block{block}_bn0'
        )(x)

    x = get_act(
        act=act,
        name=f'{name}_stage{stage}_block{block}_act_{act}0'
    )(x)

    # Sub-block 2
    if encoder:
        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=ks,
            strides=1,
            use_bias=bias,
            padding='same',
            name=f'{name}_stage{stage}_block{block}_conv1'
        )(x)
    else:
        if cut == 'post' and strides != 1:
            filters //= 2
        x = tf.keras.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=ks,
            strides=strides,
            use_bias=bias,
            padding='same',
            name=f'{name}_stage{stage}_block{block}_conv1'
        )(x)

    if bn:
        x = tf.keras.layers.BatchNormalization(
            name=f'{name}_stage{stage}_block{block}_bn1'
        )(x)

    x = get_act(
        act=act,
        name=f'{name}_stage{stage}_block{block}_act_{act}1'
    )(x)

    # Add
    x = tf.keras.layers.Add(
        name=f'{name}_stage{stage}_block{block}_add0'
    )([x, shortcut])

    return x

def gen_encoder(
    inputs: tf.Tensor,
    name: str,
    initial_filters: int,
    net_shape: List[int],
    flbn: bool,
    bias: bool,
    enc_act: str,
    iks: int,
    batch_norm: bool,
    kernel_size: int
) -> Tuple[tf.keras.models.Model, tf.Tensor, tf.Tensor]:
    """
    Constructs an encoder model using residual blocks.

    Parameters
    ----------
    inputs : tf.Tensor
        The input tensor to the encoder.
    name : str
        The name prefix for the encoder layers.
    initial_filters : int
        The number of filters for the initial convolutional layer.
    net_shape : List[int]
        A list indicating the number of residual blocks at each stage.
    flbn : bool
        If True, applies Batch Normalization after the initial convolution.
    bias : bool
        If True, adds a bias term to the convolutional layers.
    enc_act : str
        The activation function to use in the encoder.
    iks : int
        The kernel size for the initial convolutional layer.
    batch_norm : bool
        If True, applies Batch Normalization in the residual blocks.
    kernel_size : int
        The kernel size for the residual block convolutions.

    Returns
    -------
    model : tf.keras.models.Model
        The constructed encoder model.
    inputs : tf.Tensor
        The original input tensor.
    x : tf.Tensor
        The output tensor of the encoder.
    """
    name = str(name)
    
    # Input layers
    filters = initial_filters
    x = inputs
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=iks, strides=1, padding='same', use_bias=bias, name=name+'_in_conv0')(x)
    if flbn:
        x = tf.keras.layers.BatchNormalization(name=name+'_in_bn0')(x)
    x = get_act(act=enc_act, name=name+'_in_act_' + enc_act + '0')(x)

    for stage, blocks in enumerate(net_shape):
        for block in range(blocks):
            f = filters * (2 ** stage)

            if stage == 0 and block == 0:
                x = res_block(inputs=x, filters=f, stage=stage, block=block, strides=1, cut='post', encoder=True, bias=bias, act=enc_act, name=name, bn=batch_norm, ks=kernel_size)
            elif block == 0:
                x = res_block(inputs=x, filters=f, stage=stage, block=block, strides=2, cut='post', encoder=True, bias=bias, act=enc_act, name=name, bn=batch_norm, ks=kernel_size)
            else:
                x = res_block(inputs=x, filters=f, stage=stage, block=block, strides=1, cut='pre' , encoder=True, bias=bias, act=enc_act, name=name, bn=batch_norm, ks=kernel_size)
    
    return inputs, x

def gen_decoder(
    inputs: tf.Tensor,
    channels: int,
    last_act: bool,
    name: str,
    initial_filters: int,
    net_shape: List[int],
    flbn: bool,
    bias: bool,
    dec_act: str,
    iks: int,
    batch_norm: bool,
    kernel_size: int
) -> Tuple[tf.keras.models.Model, tf.Tensor, tf.Tensor]:
    """
    Constructs a decoder model using residual blocks and transposed convolutions.

    Parameters
    ----------
    inputs : tf.Tensor
        The input tensor to the decoder.
    channels : int
        The number of output channels for the final convolutional layer.
    last_act : str
        The activation function to use for the final layer.
    name : str
        The name prefix for the decoder layers.
    initial_filters : int
        The initial number of filters for the convolutional layers.
    net_shape : List[int]
        A list indicating the number of residual blocks at each stage.
    flbn : bool
        If True, applies Batch Normalization after the final convolution.
    bias : bool
        If True, includes a bias term in the convolutional layers.
    dec_act : str
        The activation function to use in the decoder.
    iks : int
        The kernel size for the initial convolutional layer.
    batch_norm : bool
        If True, applies Batch Normalization in the residual blocks.
    kernel_size : int
        The kernel size for the residual block convolutions.

    Returns
    -------
    model : tf.keras.models.Model
        The constructed decoder model.
    inputs : tf.Tensor
        The original input tensor.
    x : tf.Tensor
        The output tensor of the decoder.
    """
    name = str(name)
    
    filters = initial_filters
    x = inputs

    last_stage = len(net_shape)
    for stage, blocks in enumerate(net_shape):
        for block in range(blocks):

            f = filters * (2 ** ((last_stage - 1) - stage))

            if stage == (last_stage - 1) and block == (blocks - 1):
                x = res_block(inputs=x, filters=f, stage=stage, block=block, strides=1, cut='post', encoder=False, bias=bias, act=dec_act, name=name, bn=batch_norm, ks=kernel_size)
            elif block == (blocks - 1):
                x = res_block(inputs=x, filters=f, stage=stage, block=block, strides=2, cut='post', encoder=False, bias=bias, act=dec_act, name=name, bn=batch_norm, ks=kernel_size)
            else:
                x = res_block(inputs=x, filters=f, stage=stage, block=block, strides=1, cut='pre' , encoder=False, bias=bias, act=dec_act, name=name, bn=batch_norm, ks=kernel_size)

    # Output layers
    x = tf.keras.layers.Conv2DTranspose(filters=channels, kernel_size=iks, strides=1, padding='same', use_bias=bias, name=name+'_in_conv0')(x)
    if flbn:
        x = tf.keras.layers.BatchNormalization(name=name+'_in_bn0')(x)
    if last_act:
        x = get_act(act='sigmoid', name=name+'_in_act_sigmoid')(x)
    else:
        x = get_act(act=dec_act, name=name+'_in_act_' + 'last_lrelu' + '0')(x)

    return inputs, x

def build_generator(
    image_size: int,
    channels: int,
    latent_size: int,
    name: str,
    initial_filters: int,
    net_shape: List[int] = (2, 2, 2, 2, 2),
    flbn: bool = False,
    bias: bool = False,
    enc_act: str = 'lrelu',
    dec_act: str = 'relu',
    iks: int = 4,
    batch_norm: bool = False,
    kernel_size: int = 3,
    initial_padding: int = 0,
    initial_padding_filters: int = 0,
    dense_bottleneck: bool = False,
    verbose: bool = False,
) -> Tuple[tf.keras.models.Model, tf.keras.models.Model, tf.keras.models.Model]:
    """
    Builds a generator model using residual blocks and convolutional layers.

    Parameters
    ----------
    image_size : int
        The size of the input images (assumes square images).
    channels : int
        The number of channels in the input images.
    latent_size : int
        The size of the latent space (bottleneck).
    name : str
        The name prefix for the generator layers.
    initial_filters : int
        The initial number of filters for the convolutional layers.
    net_shape : List[int], optional
        A list indicating the number of residual blocks at each stage. Defaults to (2, 2, 2, 2, 2).
    flbn : bool, optional
        If True, applies Batch Normalization after the initial convolution. Defaults to False.
    bias : bool, optional
        If True, adds a bias term to the convolutional layers. Defaults to False.
    enc_act : str, optional
        The activation function to use in the encoder. Defaults to 'lrelu'.
    dec_act : str, optional
        The activation function to use in the decoder. Defaults to 'relu'.
    iks : int, optional
        The kernel size for the initial convolutional layer. Defaults to 4.
    batch_norm : bool, optional
        If True, applies Batch Normalization in the residual blocks. Defaults to False.
    kernel_size : int, optional
        The kernel size for the residual block convolutions. Defaults to 3.
    initial_padding : int, optional
        The amount of padding to add to the input images. Defaults to 0.
    initial_padding_filters : int, optional
        The number of filters for the initial padding convolutional layer. Defaults to 0.
    dense_bottleneck : bool, optional
        If True, adds a dense layer at the bottleneck. Defaults to False.
    verbose : bool, optional
        If True, prints the model summary and plots the model diagrams. Defaults to False.

    Returns
    -------
    Tuple[tf.keras.models.Model, tf.keras.models.Model, tf.keras.models.Model]
        The constructed encoder, autoencoder, and generator models.
    """
    shape: Tuple[int, int, int] = (image_size, image_size, channels)

    inputs = tf.keras.layers.Input(shape=shape, name='input')
    pre_x = inputs

    if initial_padding > 0:
        if initial_padding_filters <= 0:
            initial_padding_filters = channels
        pre_x = tf.keras.layers.ZeroPadding2D(padding=initial_padding, name='prePad')(pre_x)
        pre_x = tf.keras.layers.Conv2D(filters=initial_padding_filters, kernel_size=int((initial_padding * 2) + 1), strides=1, padding='valid', use_bias=bias, name='prePadConv')(pre_x)
        pre_x = tf.keras.layers.LeakyReLU(alpha=0.2, name='prePadAct')(pre_x)
        
        last_activation_flag = True
        dec_nc = initial_padding_filters
    else:
        last_activation_flag = False
        dec_nc = channels
    
    x = pre_x
    _, x = gen_encoder(
        inputs=x,
        name=f'encoder_{name}',
        initial_filters=initial_filters,
        net_shape=net_shape,
        flbn=flbn,
        bias=bias,
        enc_act=enc_act,
        iks=iks,
        batch_norm=batch_norm,
        kernel_size=kernel_size
    )

    if dense_bottleneck:
        _, h, w, c = tf.keras.backend.int_shape(x)
        x = tf.keras.layers.Flatten(name='encoder0_bottleneck_flat')(x)
        x = tf.keras.layers.Dense(latent_size, name=f'encoder0_dense_bottleneck_nz{latent_size}', use_bias=bias)(x)
        x = get_act(act=enc_act, name=name+'bottleneck_act')(x)
        latent_space = x
        x = tf.keras.layers.Dense(h * w * c, name=f'encoder0_dense_inv_bottleneck_nz{h * w * c}', use_bias=bias)(x)
        x = tf.keras.layers.Reshape((h, w, c), name='encoder0_dense_bottleneck_reshape')(x)
        x = get_act(act=dec_act, name=name+'inv_bottleneck_act')(x)
    else:
        _, _, _, pre_bottleneck_c = tf.keras.backend.int_shape(x)
        x = tf.keras.layers.Conv2D(latent_size, 4, strides=1, use_bias=bias, padding='same', name=f'encoder0_conv_bottleneck_nz{latent_size}')(x)
        if batch_norm:
            x = tf.keras.layers.BatchNormalization(name='encoder0_bn_bottleneck')(x)
        x = get_act(act=enc_act, name=name+'bottleneck_act')(x)
        _, h, w, c = tf.keras.backend.int_shape(x)
        x = tf.keras.layers.Flatten(name='encoder0_bottleneck_flat')(x)
        latent_space = x
        x = tf.keras.layers.Reshape((h, w, c), name='encoder0_dense_bottleneck_reshape')(x)
        x = tf.keras.layers.Conv2DTranspose(pre_bottleneck_c, 4, strides=1, use_bias=False, padding='same', name='decoder0_bottleneck_inv_conv')(x)

    encoder_model = tf.keras.models.Model(inputs=inputs, outputs=latent_space, name='encoder')

    _, x = gen_decoder(
        inputs=x,
        channels=channels,
        last_act=(initial_padding <= 0),
        name=f'decoder_{name}',
        initial_filters=initial_filters,
        net_shape=net_shape,
        flbn=flbn,
        bias=bias,
        dec_act=dec_act,
        iks=iks,
        batch_norm=batch_norm,
        kernel_size=kernel_size
    )

    post_x = x
    if initial_padding > 0:
        post_x = tf.keras.layers.Conv2DTranspose(filters=channels, kernel_size=int((initial_padding * 2) + 1), strides=1, padding='valid', use_bias=bias)(post_x)
        post_x = tf.keras.layers.Activation(activation=tf.nn.sigmoid)(post_x)
        post_x = tf.keras.layers.Cropping2D(cropping=initial_padding)(post_x)

    autoencoder_model = tf.keras.models.Model(inputs=inputs, outputs=post_x, name='autoencoder')

    latent_space_rebuilt = encoder_model(post_x)
    generator_model = tf.keras.models.Model(inputs=inputs, outputs=(latent_space, post_x, latent_space_rebuilt), name='generator')

    if verbose:
        # Print model summaries
        encoder_model.summary()
        autoencoder_model.summary()
        generator_model.summary()

        # Plot the autoencoder model architecture
        tf.keras.utils.plot_model(
            encoder_model,
            to_file='/tmp/encoder.png',
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            expand_nested=True,
            show_layer_activations=True,
            show_trainable=True,
        )

        # Plot the autoencoder model architecture
        tf.keras.utils.plot_model(
            autoencoder_model,
            to_file='/tmp/autoencoder.png',
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            expand_nested=True,
            show_layer_activations=True,
            show_trainable=True,
        )

        # Plot the autoencoder model architecture
        tf.keras.utils.plot_model(
            generator_model,
            to_file='/tmp/generator.png',
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            expand_nested=True,
            show_layer_activations=True,
            show_trainable=True,
        )

    return encoder_model, autoencoder_model, generator_model

def build_discriminator(
    image_size: int,
    channels: int,
    name: str,
    initial_filters: int,
    net_shape: List[int] = (2, 2, 2, 2, 2),
    flbn: bool = False,
    bias: bool = False,
    enc_act: str = 'lrelu',
    iks: int = 4,
    batch_norm: bool = False,
    kernel_size: int = 3,
    initial_padding: int = 0,
    initial_padding_filters: int = 0,
    verbose: bool = False,
) -> Tuple[tf.keras.models.Model, tf.keras.models.Model, tf.keras.models.Model]:
    """
    Builds a generator model using residual blocks and convolutional layers.

    Parameters
    ----------
    image_size : int
        The size of the input images (assumes square images).
    channels : int
        The number of channels in the input images.
    name : str
        The name prefix for the generator layers.
    initial_filters : int
        The initial number of filters for the convolutional layers.
    net_shape : List[int], optional
        A list indicating the number of residual blocks at each stage. Defaults to (2, 2, 2, 2, 2).
    flbn : bool, optional
        If True, applies Batch Normalization after the initial convolution. Defaults to False.
    bias : bool, optional
        If True, adds a bias term to the convolutional layers. Defaults to False.
    enc_act : str, optional
        The activation function to use in the encoder. Defaults to 'lrelu'.
    dec_act : str, optional
        The activation function to use in the decoder. Defaults to 'relu'.
    iks : int, optional
        The kernel size for the initial convolutional layer. Defaults to 4.
    batch_norm : bool, optional
        If True, applies Batch Normalization in the residual blocks. Defaults to False.
    kernel_size : int, optional
        The kernel size for the residual block convolutions. Defaults to 3.
    initial_padding : int, optional
        The amount of padding to add to the input images. Defaults to 0.
    initial_padding_filters : int, optional
        The number of filters for the initial padding convolutional layer. Defaults to 0.
    dense_bottleneck : bool, optional
        If True, adds a dense layer at the bottleneck. Defaults to False.
    verbose : bool, optional
        If True, prints the model summary and plots the model diagrams. Defaults to False.

    Returns
    -------
    Tuple[tf.keras.models.Model, tf.keras.models.Model, tf.keras.models.Model]
        The constructed encoder, autoencoder, and generator models.
    """
    shape: Tuple[int, int, int] = (image_size, image_size, channels)

    inputs = tf.keras.layers.Input(shape=shape, name='input')
    pre_x = inputs

    if initial_padding > 0:
        if initial_padding_filters <= 0:
            initial_padding_filters = channels
        pre_x = tf.keras.layers.ZeroPadding2D(padding=initial_padding, name='prePad')(pre_x)
        pre_x = tf.keras.layers.Conv2D(filters=initial_padding_filters, kernel_size=int((initial_padding * 2) + 1), strides=1, padding='valid', use_bias=bias, name='prePadConv')(pre_x)
        pre_x = tf.keras.layers.LeakyReLU(alpha=0.2, name='prePadAct')(pre_x)
        
        last_activation_flag = True
        dec_nc = initial_padding_filters
    else:
        last_activation_flag = False
        dec_nc = channels
    
    x = pre_x
    _, x = gen_encoder(
        inputs=x,
        name=f'encoder_{name}',
        initial_filters=initial_filters,
        net_shape=net_shape,
        flbn=flbn,
        bias=bias,
        enc_act=enc_act,
        iks=iks,
        batch_norm=batch_norm,
        kernel_size=kernel_size
    )

    feat_x = x

    x = tf.keras.layers.Flatten(name='encoder0_bottleneck_flat')(x)
    x = tf.keras.layers.Dense(1, name='encoder0_dense_bottleneck_out', activation='sigmoid')(x)

    discriminator_model = tf.keras.models.Model(inputs=inputs, outputs=(feat_x, x), name='discriminator')

    if verbose:
        # Print model summaries
        discriminator_model.summary()

        # Plot the autoencoder model architecture
        tf.keras.utils.plot_model(
            discriminator_model,
            to_file='/tmp/discriminator.png',
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            expand_nested=True,
            show_layer_activations=True,
            show_trainable=True,
        )

    return discriminator_model
