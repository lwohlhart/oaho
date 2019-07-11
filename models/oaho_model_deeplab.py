import tensorflow as tf
from tensorflow.keras import layers as kl
from tensorflow.keras import applications as ka
from models.oaho_model import OAHOModel
from typing import Dict, Tuple


class OAHOModelDeeplab(OAHOModel):
    def __init__(self, config: dict) -> None:
        """
        :param config: global configuration
        """
        super().__init__(config)

    def _create_model(self, x: tf.Tensor, is_training: bool) -> tf.Tensor:
        """
        Implement the architecture of your model
        :param x: input data
        :param is_training: flag if currently training
        :return: completely constructed model
        """

        tf.logging.info("Constructing OAHO Model Deeplab")

        def decoder_block(input_tensor: tf.Tensor, feature_maps: int, kernel_size: Tuple[int, int],
                          strides: Tuple[int, int], is_training: bool,
                          skip_connection: tf.Tensor = None) -> tf.Tensor:
            x = kl.Conv2DTranspose(feature_maps, kernel_size=kernel_size, strides=strides, padding='same')(input_tensor)
            x = kl.BatchNormalization()(x)
            x = kl.ReLU()(x)
            if skip_connection is not None:
                x = tf.keras.layers.concatenate([x, skip_connection])
            return x

        no_filters = [10, 32, 128, 512]
        filter_sizes= [(3, 3), (3, 3), (3, 3), (2, 2)]

        input_tensor = x
        x = tf.keras.layers.concatenate([x,x,x])

        pretrained_weights = 'imagenet' if is_training else None

        resnet = ka.ResNet50(weights=pretrained_weights, include_top=False, input_tensor=x, input_shape=(480,640,3))

        enc1 = resnet.get_layer('activation').output # resnet.layers[4].output
        enc2 = resnet.get_layer('activation_9').output # resnet.layers[38].output
        enc3 = resnet.get_layer('activation_21').output # resnet.layers[80].output
        enc4 = resnet.get_layer('activation_39').output # resnet.layers[142].output
        # enc5 = resnet.get_layer('activation_48').output # resnet.layers[174].output

        dec = enc4

        # dec = decoder_block(dec, no_filters[3], kernel_size=filter_sizes[3], strides=(2, 2), skip_connection=enc4, is_training=is_training)
        dec = decoder_block(dec, no_filters[3], kernel_size=filter_sizes[3], strides=(2, 2), skip_connection=enc3, is_training=is_training)
        dec = decoder_block(dec, no_filters[2], kernel_size=filter_sizes[2], strides=(2, 2), skip_connection=enc2, is_training=is_training)
        dec = decoder_block(dec, no_filters[1], kernel_size=filter_sizes[1], strides=(2, 2), skip_connection=enc1, is_training=is_training)
        dec = decoder_block(dec, no_filters[0], kernel_size=filter_sizes[0], strides=(2, 2), skip_connection=input_tensor, is_training=is_training)

        x = dec
        # ===================================================================================================
        # Output layers
        seg_head = kl.Conv2D(32, kernel_size=2, padding='same', name='seg_head_1', activation='relu')(x)
        seg_output = kl.Conv2D(4, kernel_size=2, padding='same', name='seg_out')(seg_head)

        grasp_head = kl.Conv2D(32, kernel_size=2, padding='same', name='grasp_head_1', activation='relu')(x)
        pos_output = kl.Conv2D(1, kernel_size=2, padding='same', name='pos_out')(grasp_head)
        cos_output = kl.Conv2D(1, kernel_size=2, padding='same', name='cos_out')(grasp_head)
        sin_output = kl.Conv2D(1, kernel_size=2, padding='same', name='sin_out')(grasp_head)
        width_output = kl.Conv2D(1, kernel_size=2, padding='same', name='width_out')(grasp_head)
        return seg_output, pos_output, cos_output, sin_output, width_output