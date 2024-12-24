from typing import Optional, Tuple, List

import tensorflow as tf

def build_simple_generator(
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


    # ENCODER
    while h > deep_feature_size:
        # Add a convolutional layer with a stride of 2 to reduce the spatial dimensions by half
        x = tf.keras.layers.Conv2D(
            filters=feat,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            use_bias=False,
            name=f'encoder_conv2d_{len(encoder_layers)}'
        )(x)

        # Optionally add Batch Normalization
        if use_bn:
            x = tf.keras.layers.BatchNormalization(name=f'encoder_bn_{len(encoder_layers)}')(x)

        # Apply LeakyReLU activation
        x = tf.keras.layers.LeakyReLU(alpha=0.2, name=f'encoder_activation_{len(encoder_layers)}')(x)

        skips.append(x)

        # Save the layer configuration for the decoder
        encoder_layers.append(feat)
        h //= 2
        feat *= 2  # Double the features after each encoder block

    # Dropout layer before bottleneck
    x = tf.keras.layers.Dropout(0.1, name='dropout_bottleneck')(x)

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
        x = tf.keras.layers.Conv2DTranspose(
            filters=feat,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            use_bias=False,
            name=f'decoder_conv2dtranspose_{idx}'
        )(x)

        # Optionally add Batch Normalization
        if use_bn:
            x = tf.keras.layers.BatchNormalization(name=f'decoder_bn_{idx}')(x)

        # Apply LeakyReLU activation
        x = tf.keras.layers.LeakyReLU(alpha=0.2, name=f'decoder_activation_{idx}')(x)

    # OUTPUT LAYER
    if override_output_channels is not None and override_output_channels > 0:
        channels = override_output_channels
    outputs = tf.keras.layers.Conv2DTranspose(
        filters=channels,
        kernel_size=(3, 3),
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


def build_simple_discriminator(
    image_size: int,
    channels: int,
    init_features: int = 64,
    use_bn: bool = True,
    deep_feature_size: int = 4,
    verbose: bool = True
) -> tf.keras.models.Model:
    """
    Builds a simple discriminator model with a convolutional encoder and a fully connected layer.

    The architecture consists of a convolutional encoder, followed by a fully connected layer with a
    sigmoid activation function. The output of the model is a probability (a value in the range [0, 1]).

    Parameters:
    - image_size: Size of the input images (assumes square images).
    - channels: Number of channels in the input images.
    - init_features: Initial number of features in the encoder.
    - use_bn: If True, adds Batch Normalization to the model.
    - deep_feature_size: Size of the deepest feature map.
    - verbose: If True, prints the model summary.

    Returns:
    - A Keras Model representing the discriminator.
    """
    shape: Tuple[int, int, int] = (image_size, image_size, channels)
    inputs = tf.keras.layers.Input(shape=shape, name='input')
    x = inputs

    # ENCODER
    feat: int = init_features
    h = image_size
    encoder_layers = []

    while h > deep_feature_size:
        # Add a convolutional layer with a stride of 2 to reduce the
        # spatial dimensions of the input data by half
        x = tf.keras.layers.Conv2D(
            filters=feat,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            use_bias=False,
            name=f'encoder_conv2d_{len(encoder_layers)}'
        )(x)

        # Add Batch Normalization to the output of the convolutional layer
        if use_bn:
            x = tf.keras.layers.BatchNormalization(name=f'encoder_bn_{len(encoder_layers)}')(x)

        # Add a LeakyReLU activation function to the output of the convolutional layer
        x = tf.keras.layers.LeakyReLU(alpha=0.2, name=f'encoder_activation_{len(encoder_layers)}')(x)

        # Save the output of the convolutional layer for the decoder
        encoder_layers.append(feat)
        h = h // 2
        feat *= 2  # Double the features after each layer

    feat_out = x  # f(x)

    # Flatten the tensor output of the convolutional layers
    x = tf.keras.layers.Flatten()(x)

    # Dropout layer before bottleneck
    x = tf.keras.layers.Dropout(0.1)(x)

    # Output layer
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=(feat_out, output), name='discriminator_model')

    if verbose:
        model.summary()
        tf.keras.utils.plot_model(
            model,
            to_file='/tmp/discriminator.png',
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            expand_nested=True,
            show_layer_activations=True,
            show_trainable=True,
        )

    return model
