import tensorflow as tf
from tensorflow.keras import layers as kl
from models.oaho_model import OAHOModel
from typing import Dict, Tuple, List


class OAHOModelResidualUnet(OAHOModel):
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

        def encoder_block(input_tensor: tf.Tensor, feature_maps: int, kernel_size: Tuple[int, int],
                          strides: Tuple[int, int], is_training: bool, scope: str) -> tf.Tensor:
            with tf.name_scope(scope):
                x = input_tensor
                x = kl.Conv2D(feature_maps, kernel_size=kernel_size, strides=strides, padding='same')(x)
                x = kl.ReLU()(x)
                x = kl.BatchNormalization()(x)
                x = kl.Conv2D(feature_maps, kernel_size=kernel_size, strides=strides, padding='same')(x)
                x = kl.ReLU()(x)
                x = kl.BatchNormalization()(x)

                x = kl.Concatenate()([x, input_tensor])
                pre_pooling = x
                x = kl.MaxPooling2D(strides=(2, 2), padding='same')(x)
            return x, pre_pooling

        def decoder_block(input_tensor: tf.Tensor, feature_maps: int, kernel_size: Tuple[int, int],
                          strides: Tuple[int, int], is_training: bool, scope: str,
                          skip_connection: tf.Tensor = None) -> tf.Tensor:
            with tf.name_scope(scope):                          
                x = input_tensor
                x = kl.UpSampling2D(interpolation='bilinear')(x)
                if skip_connection is not None:
                    x = kl.Concatenate()([x, skip_connection])
                x = kl.Conv2D(feature_maps, kernel_size=kernel_size, strides=strides, padding='same')(x)
                x = kl.ReLU()(x)
                x = kl.BatchNormalization()(x)
                x = kl.Conv2D(feature_maps, kernel_size=kernel_size, strides=strides, padding='same')(x)
                x = kl.ReLU()(x)
                x = kl.BatchNormalization()(x)
            return x

        def dilated_bottleneck(input_tensor: tf.Tensor, feature_maps: int, kernel_size: Tuple[int, int], 
                                dilation_rates: List, is_training: bool, scope: str):
            with tf.name_scope(scope):
                x = input_tensor
                dilated_features = [x] # or initialize without image pooling []
                for rate in dilation_rates:
                    f = kl.Conv2D(feature_maps, kernel_size=kernel_size, padding='same', dilation_rate=(rate, rate))(x)
                    dilated_features.append(f)

                x = kl.Concatenate()(dilated_features)
                x = kl.Conv2D(feature_maps, kernel_size=(1, 1), padding='same')(x)
            return x

        tf.logging.info('Constructing OAHO Model Residual Unet')

        x, enc_skip_1 = encoder_block(x, 32, (3, 3), (1, 1), is_training=is_training, scope='encoder_block1')
        x, enc_skip_2 = encoder_block(x, 64, (3, 3), (1, 1), is_training=is_training, scope='encoder_block2')
        x, enc_skip_3 = encoder_block(x, 128, (3, 3), (1, 1), is_training=is_training, scope='encoder_block3')
        x, enc_skip_4 = encoder_block(x, 256, (3, 3), (1, 1), is_training=is_training, scope='encoder_block4')

        x = bottleneck = dilated_bottleneck(x, 512, (3, 3), [1, 2, 4], is_training=is_training, scope='aspp')

        x = dec4 = decoder_block(x, 256, (3, 3), (1, 1), is_training=is_training, scope='decoder_block4', skip_connection=enc_skip_4)
        x = dec3 = decoder_block(x, 128, (3, 3), (1, 1), is_training=is_training, scope='decoder_block3', skip_connection=enc_skip_3)
        x = dec2 = decoder_block(x, 64, (3, 3), (1, 1), is_training=is_training, scope='decoder_block2', skip_connection=enc_skip_2)
        x = dec1 = decoder_block(x, 32, (3, 3), (1, 1), is_training=is_training, scope='decoder_block1', skip_connection=enc_skip_1)
        
        
        # x = dec
        # ===================================================================================================
        # Output layers
        # seg_head = kl.Conv2D(32, kernel_size=2, padding='same', name='seg_head_1', activation='relu')(features_decoder)
        seg_output = kl.Conv2D(4, kernel_size=1, padding='same', name='seg_out')(x)
        
        # grasp_head = kl.Conv2D(32, kernel_size=1, padding='same', name='grasp_head_1', activation='relu')(features_decoder)
        grasp_head = tf.keras.layers.concatenate([x, seg_output])
        pos_output = kl.Conv2D(1, kernel_size=1, padding='same', name='pos_out')(grasp_head)
        cos_output = kl.Conv2D(1, kernel_size=1, padding='same', name='cos_out')(grasp_head)
        sin_output = kl.Conv2D(1, kernel_size=1, padding='same', name='sin_out')(grasp_head)
        width_output = kl.Conv2D(1, kernel_size=1, padding='same', name='width_out')(grasp_head)
        return seg_output, pos_output, cos_output, sin_output, width_output