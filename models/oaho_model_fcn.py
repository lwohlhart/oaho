import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, ReLU
from models.oaho_model import OAHOModel
from typing import Dict, Tuple


class OAHOModelFCN(OAHOModel):
    def __init__(self, config: dict) -> None:
        """
        :param config: global configuration
        """
        super().__init__(config)

    @staticmethod
    def encoder_block(input_tensor: tf.Tensor, feature_maps: int, kernel_size: Tuple[int, int], strides: Tuple[int, int]) -> tf.Tensor:
        x = Conv2D(feature_maps, kernel_size=kernel_size, strides=strides, padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x
    
    @staticmethod
    def decoder_block(input_tensor: tf.Tensor, feature_maps: int, kernel_size: Tuple[int, int], strides: Tuple[int, int], skip_connection: tf.Tensor = None) -> tf.Tensor:
        x = Conv2DTranspose(feature_maps, kernel_size=kernel_size, strides=strides, padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        if skip_connection is not None:
            x = tf.keras.layers.concatenate([x, skip_connection])
        return x

    def _create_model(self, x: tf.Tensor, is_training: bool) -> tf.Tensor:
        """
        Implement the architecture of your model
        :param x: input data
        :param is_training: flag if currently training
        :return: completely constructed model
        """
        tf.logging.info("Constructing OAHO Model FCN")

        no_filters = [10, 32, 128, 512]
        filter_sizes= [(3, 3), (3, 3), (3, 3), (2, 2)]
        input_tensor = x

        enc1 = OAHOModelFCN.encoder_block(input_tensor, no_filters[0], kernel_size=filter_sizes[0], strides=(2, 2))
        enc2 = OAHOModelFCN.encoder_block(enc1, no_filters[1], kernel_size=filter_sizes[1], strides=(2, 2))
        enc3 = OAHOModelFCN.encoder_block(enc2, no_filters[2], kernel_size=filter_sizes[2], strides=(2, 2))
        enc4 = OAHOModelFCN.encoder_block(enc3, no_filters[3], kernel_size=filter_sizes[3], strides=(2, 2))

        dec0 = OAHOModelFCN.decoder_block(enc4, no_filters[3], kernel_size=filter_sizes[3], strides=(2, 2), skip_connection=enc3)
        dec1 = OAHOModelFCN.decoder_block(dec0, no_filters[2], kernel_size=filter_sizes[2], strides=(2, 2), skip_connection=enc2)
        dec2 = OAHOModelFCN.decoder_block(dec1, no_filters[1], kernel_size=filter_sizes[1], strides=(2, 2), skip_connection=enc1)
        dec3 = OAHOModelFCN.decoder_block(dec2, no_filters[0], kernel_size=filter_sizes[0], strides=(2, 2), skip_connection=input_tensor)
        
        x = dec3
        # ===================================================================================================
        # Output layers
        seg_head = Conv2D(10, kernel_size=2, padding='same', name='seg_out')(x)
        seg_output = Conv2D(4, kernel_size=2, padding='same', name='seg_out')(seg_head)

        grasp_head = Conv2D(10, kernel_size=2, padding='same', name='seg_out')(x)
        pos_output = Conv2D(1, kernel_size=2, padding='same', name='pos_out')(grasp_head)
        cos_output = Conv2D(1, kernel_size=2, padding='same', name='cos_out')(grasp_head)
        sin_output = Conv2D(1, kernel_size=2, padding='same', name='sin_out')(grasp_head)
        width_output = Conv2D(1, kernel_size=2, padding='same', name='width_out')(grasp_head)
        return seg_output, pos_output, cos_output, sin_output, width_output
