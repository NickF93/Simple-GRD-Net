from typing import Optional, Tuple, List

import tensorflow as tf

def build_residual_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: Tuple[int],
    strides: int,
    name: str = None,
    use_bn: bool = True,
    use_residual: bool = True,
    use_dropout: bool = False,
    use_bias: bool = False,
    use_conv_1x1: bool = False,
    max_pooling: bool = False,
    decoder: bool = False,
    add: bool = True,
    last_activation: bool = True,
) -> tf.Tensor:
    """
    Builds a residual block with optional Batch Normalization, convolutional layers, and dropout.

    Args:
        x (tf.Tensor): The input tensor.
        filters (int): The number of filters in the convolutional layers.
        kernel_size (int): The size of the convolutional kernel.
        strides (int): The stride of the convolutional layers.
        name (str, optional): The name of the block. Defaults to None.
        use_bn (bool, optional): If True, adds Batch Normalization to the model. Defaults to True.
        use_residual (bool, optional): If True, adds a skip connection to the output of the block. Defaults to True.
        use_dropout (bool, optional): If True, adds dropout to the model. Defaults to False.
        use_bias (bool, optional): If True, adds a bias term to the convolutional layers. Defaults to False.
        use_conv_1x1 (bool, optional): If True, adds a 1x1 convolutional layer to the model. Defaults to False.
        max_pooling (bool, optional): If True, uses max pooling instead of average pooling. Defaults to False.
        decoder (bool, optional): If True, the block is used for the decoder. Defaults to False.
        add (bool, optional): If True, adds the residual connection to the output of the block. Defaults to True.
        last_activation (bool, optional): If True, applies a LeakyReLU activation function to the output of the block. Defaults to True.

    Returns:
        tf.Tensor: The output tensor of the block.
    """
    residual = x

    for i, kernel in enumerate(kernel_size):
        if not decoder:
            # Convolutional layer with LeakyReLU activation
            x = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=kernel,
                strides=1 if i < len(kernel_size) - 1 else strides,
                use_bias=use_bias,
                padding="same",
                name=f"{name}_conv_{i + 1}"
            )(x)
            if use_bn:
                # Batch normalization
                x = tf.keras.layers.BatchNormalization(name=f"{name}_bn_{i + 1}")(x)
            if use_dropout:
                # Dropout
                x = tf.keras.layers.Dropout(0.1, name=f"{name}_dropout_{i + 1}")(x)
            if i < len(kernel_size) - 1:
                # LeakyReLU activation
                x = tf.keras.layers.LeakyReLU(negative_slope=0.2,name=f"{name}_leakyrelu_{i + 1}")(x)
        else:
            # Convolutional transpose layer with LeakyReLU activation
            x = tf.keras.layers.Conv2DTranspose(
                filters=filters,
                kernel_size=kernel,
                strides=strides if i == 0 else 1,
                use_bias=use_bias,
                padding="same",
                name=f"{name}_conv_transpose_{i + 1}"
            )(x)
            if use_bn:
                # Batch normalization
                x = tf.keras.layers.BatchNormalization(name=f"{name}_bn_{i + 1}")(x)
            if use_dropout:
                # Dropout
                x = tf.keras.layers.Dropout(0.1, name=f"{name}_dropout_{i + 1}")(x)
            if i < len(kernel_size) - 1:
                # LeakyReLU activation
                x = tf.keras.layers.LeakyReLU(negative_slope=0.2,name=f"{name}_leakyrelu_{i + 1}")(x)

    if use_conv_1x1:
        # 1x1 convolutional layer
        if not decoder:
            residual = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=1,
                strides=1,
                use_bias=use_bias,
                padding="same",
                name=f"{name}_conv_1x1"
            )(residual)
            if strides > 1:
                # Max pooling or average pooling
                pooling_layer_class = tf.keras.layers.MaxPooling2D if max_pooling else tf.keras.layers.AveragePooling2D
                residual = pooling_layer_class(
                    pool_size=strides,
                    strides=strides,
                    padding="same",
                    name=f"{name}_pool_1x1"
                )(residual)
        else:
            residual = tf.keras.layers.UpSampling2D(
                size=strides,
                interpolation="bilinear",
                name=f"{name}_upsample_1x1"
            )(residual)
            residual = tf.keras.layers.Conv2DTranspose(
                filters=filters,
                kernel_size=1,
                strides=1,
                use_bias=use_bias,
                padding="same",
                name=f"{name}_conv_transpose_1x1"
            )(residual)
        if use_bn:
            # Batch normalization
            residual = tf.keras.layers.BatchNormalization(name=f"{name}_bn_1x1")(residual)
        if use_dropout:
            # Dropout
            residual = tf.keras.layers.Dropout(0.1, name=f"{name}_dropout_1x1")(residual)

    if use_residual:
        # Residual connection
        if add:
            x = tf.keras.layers.Add(name=f"{name}_add")([x, residual])
        else:
            x = tf.keras.layers.Concatenate(name=f"{name}_concat")([x, residual])
            if not decoder:
                # Convolutional layer to reduce the number of filters
                x = tf.keras.layers.Conv2D(
                    filters=filters,
                    kernel_size=1,
                    strides=1,
                    use_bias=use_bias,
                    padding="same",
                    name=f"{name}_conv_out"
                )(x)
            else:
                # Convolutional transpose layer to increase the number of filters
                x = tf.keras.layers.Conv2DTranspose(
                    filters=filters,
                    kernel_size=1,
                    strides=1,
                    use_bias=use_bias,
                    padding="same",
                    name=f"{name}_conv_transpose_out"
                )(x)
            if use_bn:
                # Batch normalization
                x = tf.keras.layers.BatchNormalization(name=f"{name}_bn_out")(x)
            if use_dropout:
                # Dropout
                x = tf.keras.layers.Dropout(0.1, name=f"{name}_dropout_out")(x)

    if last_activation:
        # LeakyReLU activation
        x = tf.keras.layers.LeakyReLU(negative_slope=0.2,name=f"{name}_leakyrelu_out")(x)

    return x


def build_residual_stage(
    x: tf.Tensor,
    filters: int,
    name: str = None,
    use_bn: bool = True,
    use_residual: bool = True,
    use_dropout: bool = False,
    use_bias: bool = False,
    max_pooling: bool = False,
    decoder: bool = False,
    add: bool = True,
    residual_stage: bool = False,
) -> tf.Tensor:
    if residual_stage:
        residual = x

    x = build_residual_block(
        x=x,
        filters=filters,
        kernel_size=(3, 1, 1),
        strides=1 if (not decoder) else 2,
        name=f'{name}_block_1',
        use_bn=use_bn,
        use_residual=use_residual,
        use_dropout=use_dropout,
        use_bias=use_bias,
        use_conv_1x1=True,
        max_pooling=max_pooling,
        decoder=decoder,
        add=add
    )
    x = build_residual_block(
        x=x,
        filters=filters,
        kernel_size=(3, 1, 3),
        strides=1,
        name=f'{name}_block_2',
        use_bn=use_bn,
        use_residual=use_residual,
        use_dropout=use_dropout,
        use_bias=use_bias,
        use_conv_1x1=False,
        max_pooling=max_pooling,
        decoder=decoder,
        add=add
    )
    x = build_residual_block(
        x=x,
        filters=filters,
        kernel_size=(3, 1, 3),
        strides=2 if (not decoder) else 1,
        name=f'{name}_block_3',
        use_bn=use_bn,
        use_residual=use_residual,
        use_dropout=use_dropout,
        use_bias=use_bias,
        use_conv_1x1=True,
        max_pooling=max_pooling,
        decoder=decoder,
        add=add,
        last_activation=(not residual_stage)
    )

    if residual_stage:
        if not decoder:
            residual = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=1,
                strides=1,
                use_bias=use_bias,
                padding="same",
                name=f'{name}_conv_stage_residual'
            )(residual)
            pooling_layer_class = tf.keras.layers.MaxPooling2D if max_pooling else tf.keras.layers.AveragePooling2D
            residual = pooling_layer_class(
                pool_size=2,
                strides=2,
                padding="same",
                name=f'{name}_pool_stage_residual'
            )(residual)
        else:
            residual = tf.keras.layers.UpSampling2D(
                size=2,
                interpolation="bilinear",
                name=f'{name}_upsample_stage_residual'
            )(residual)
            residual = tf.keras.layers.Conv2DTranspose(
                filters=filters,
                kernel_size=1,
                strides=1,
                use_bias=use_bias,
                padding="same",
                name=f'{name}_conv_transpose_stage_residual'
            )(residual)
        if use_bn:
            residual = tf.keras.layers.BatchNormalization(name=f'{name}_bn_stage_residual')(residual)
        if use_dropout:
            residual = tf.keras.layers.Dropout(0.1, name=f'{name}_dropout_stage_residual')(residual)
        x = tf.keras.layers.Add(name=f'{name}_add')([x, residual])

        x = tf.keras.layers.LeakyReLU(negative_slope=0.2, name=f'{name}_leakyrelu_stage_out')(x)

    return x
    

def build_residual_generator(
    image_size: int,
    channels: int,
    init_features: int = 64,
    latent_space_size: int = 32,
    use_bn: bool = True,
    deep_feature_size: int = 4,
    dense_bottleneck: bool = False,
    build_unet: bool = False,
    skip_depth: int = 2,
    skip_dense: bool = False,
    override_output_channels: Optional[int] = None,
    use_dropout: bool = False,
    verbose: bool = True
) -> tf.keras.models.Model:
    """
    Builds a symmetric autoencoder model with optional dense bottleneck and Batch Normalization.

    Parameters:
    - image_size (int): Size of the input images (assumes square images).
    - channels (int): Number of channels in the input images.
    - init_features (int): Initial number of features in the encoder.
    - latent_space_size (int): Size of the latent space (bottleneck).
    - use_bn (bool): If True, adds Batch Normalization to the model.
    - deep_feature_size (int): Minimum feature map size before the bottleneck.
    - dense_bottleneck (bool): If True, adds a dense layer at the bottleneck.
    - verbose (bool): If True, prints the model summary and plots the model diagrams.

    Returns:
    - Tuple of Keras Models: Encoder, Autoencoder, and Generator models.
    """
    shape: Tuple[int, int, int] = (image_size, image_size, channels)
    inputs = tf.keras.layers.Input(shape=shape, name='input')
    x = inputs

    feat: int = init_features
    h = image_size
    encoder_layers = []

    skips: List[tf.Tensor] = []

    if build_unet and not skip_dense:
        dense_bottleneck = False
    elif build_unet and skip_dense:
        dense_bottleneck = True

    x = tf.keras.layers.Conv2D(
        filters=feat,
        kernel_size=(7, 7),
        strides=(1, 1),
        padding='same',
        use_bias=False,
        name='encoder_conv2d_stem'
    )(x)

    if use_bn:
        x = tf.keras.layers.BatchNormalization(name='encoder_conv2d_stem_bn')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2, name='encoder_conv2d_stem_leakyrelu')(x)

    # ENCODER
    level = 0
    while h > deep_feature_size:
        # Add a convolutional layer with a stride of 2 to reduce the spatial dimensions by half
        x = build_residual_stage(
            x = x,
            filters = feat,
            name = f'encoder_stage_{level}',
            use_bn = False,
            use_residual = True,
            use_dropout = use_dropout,
            use_bias = False,
            max_pooling = True,
            decoder = False,
            add = True,
            residual_stage = True,
        )

        skips.append(x)

        # Save the layer configuration for the decoder
        encoder_layers.append(feat)
        h //= 2
        feat *= 2  # Double the features after each encoder block
        level += 1

    # BOTTLENECK
    x = tf.keras.layers.Conv2D(
        filters=latent_space_size,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        use_bias=False,
        name='conv_bottleneck'
    )(x)

    # Optionally add Batch Normalization to the bottleneck
    if use_bn:
        x = tf.keras.layers.BatchNormalization(name='conv_bottleneck_bn')(x)

    # Apply LeakyReLU activation to the bottleneck
    x = tf.keras.layers.LeakyReLU(alpha=0.2, name='conv_bottleneck_activation')(x)

    if dense_bottleneck:
        # Dense bottleneck configuration
        _, h, w, c = tf.keras.backend.int_shape(x)
        flattened_tensor_size = int(h * w * c)
        x = tf.keras.layers.Flatten(name='dense_bottleneck_flatten')(x)
        x = tf.keras.layers.Dense(
            units=latent_space_size,
            name='dense_bottleneck'
        )(x)
        if use_bn:
            x = tf.keras.layers.BatchNormalization(name='dense_bottleneck_bn')(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2, name='dense_bottleneck_activation')(x)

    latent_space = x

    # Define the encoder model
    if not build_unet:
        encoder_model = tf.keras.models.Model(inputs=inputs, outputs=latent_space, name='encoder_model')
    else:
        encoder_model = None

    if dense_bottleneck:
        # Inverse dense bottleneck for the decoder
        x = tf.keras.layers.Dense(
            units=flattened_tensor_size,
            name='dense_bottleneck_inverse'
        )(x)
        if use_bn:
            x = tf.keras.layers.BatchNormalization(name='dense_bottleneck_inverse_bn')(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2, name='dense_bottleneck_inverse_activation')(x)
        x = tf.keras.layers.Reshape(target_shape=(h, w, c), name='dense_bottleneck_inverse_reshape')(x)

    # DECODER
    encoder_layers = encoder_layers[::-1]  # Reverse encoder layers for the decoder

    # Initial Conv2DTranspose to expand dimensions
    x = tf.keras.layers.Conv2DTranspose(
        filters=encoder_layers[0],
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        use_bias=False,
        name='decoder_conv2dtranspose_initial'
    )(x)

    # Optionally add Batch Normalization
    if use_bn:
        x = tf.keras.layers.BatchNormalization(name='decoder_bn_initial')(x)

    # Apply LeakyReLU activation
    x = tf.keras.layers.LeakyReLU(alpha=0.2, name='decoder_activation_initial')(x)

    for idx, feat in enumerate(encoder_layers):
        if build_unet and skip_depth is not None and skip_depth > 0:
            skip_x = skips.pop()

            if skip_dense:
                # Dense bottleneck configuration
                _, h, w, c = tf.keras.backend.int_shape(skip_x)
                flattened_tensor_size = int(h * w * c)
                skip_x = tf.keras.layers.Flatten(name='dense_bottleneck_flatten')(skip_x)
                skip_x = tf.keras.layers.Dense(
                    units=latent_space_size,
                    name='dense_bottleneck'
                )(skip_x)
                if use_bn:
                    skip_x = tf.keras.layers.BatchNormalization(name='dense_bottleneck_bn')(skip_x)
                skip_x = tf.keras.layers.LeakyReLU(alpha=0.2, name='dense_bottleneck_activation')(skip_x)

                # Inverse dense bottleneck for the decoder
                skip_x = tf.keras.layers.Dense(
                    units=flattened_tensor_size,
                    name='dense_bottleneck_inverse'
                )(skip_x)
                if use_bn:
                    skip_x = tf.keras.layers.BatchNormalization(name='dense_bottleneck_inverse_bn')(skip_x)
                skip_x = tf.keras.layers.LeakyReLU(alpha=0.2, name='dense_bottleneck_inverse_activation')(skip_x)
                skip_x = tf.keras.layers.Reshape(target_shape=(h, w, c), name='dense_bottleneck_inverse_reshape')(skip_x)


            x = tf.keras.layers.Concatenate(name=f'decoder_concat_skip_{idx}')([x, skip_x])
            skip_depth -= 1

        # Add a Conv2DTranspose layer with a stride of 2 to double the spatial dimensions
        x = build_residual_stage(
            x = x,
            filters = feat,
            name = f'decoder_stage_{idx}',
            use_bn = False,
            use_residual = True,
            use_dropout = use_dropout,
            use_bias = False,
            max_pooling = True,
            decoder = True,
            add = True,
            residual_stage = True,
        )

    # OUTPUT LAYER
    if override_output_channels is not None and override_output_channels > 0:
        channels = override_output_channels
    outputs = tf.keras.layers.Conv2DTranspose(
        filters=channels,
        kernel_size=(7, 7),
        strides=(1, 1),
        padding='same',
        use_bias=False,
        activation='sigmoid',
        name='output_layer'
    )(x)

    # Define the autoencoder model
    autoencoder_model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name=('symmetric_autoencoder_model' if not build_unet else 'unet_model'))

    if not build_unet:
        # Define the generator model
        inputs = tf.keras.layers.Input(shape=shape, name='input_generator')
        z = encoder_model(inputs)
        xf = autoencoder_model(inputs)
        zf = encoder_model(xf)
        generator_model = tf.keras.models.Model(inputs=inputs, outputs=(z, xf, zf), name='generator_model')
    else:
        generator_model = None

    if verbose:
        # Print model summaries
        autoencoder_model.summary()
        if not build_unet:
            encoder_model.summary()
            generator_model.summary()

        # Plot the autoencoder model architecture
        tf.keras.utils.plot_model(
            autoencoder_model,
            to_file='/tmp/autoencoder.png' if not build_unet else '/tmp/unet.png',
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            expand_nested=True,
            show_layer_activations=True,
            show_trainable=True,
        )
        if not build_unet:
            # Plot the encoder model architecture
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
            # Plot the generator model architecture
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
