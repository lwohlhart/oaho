import tensorflow as tf
from tensorflow.keras import layers as kl
from tensorflow.keras import applications as ka
from models.oaho_model import OAHOModel
from typing import Dict, Tuple

import sys
sys.path.append('tensorflow_models/research')
sys.path.append('tensorflow_models/research/slim')

import deeplab.common as dlc
import deeplab.model as dlm
from deeplab.core import utils as dlutils
import deeplab.core.feature_extractor as dl_feature_extractor

class OAHOModelDeeplabDeconv(OAHOModel):
    def __init__(self, config: dict) -> None:
        """
        :param config: global configuration
        """
        super().__init__(config)

        # config['model_variant'] = 'mobilenet_v2'
        # config['model_variant'] = 'xception_65'
        config['model_variant'] = 'resnet_v1_50'

        tf.flags.FLAGS.model_variant = config['model_variant']

        if 'resnet' in config['model_variant'] or 'xception' in config['model_variant']:
            tf.flags.FLAGS.decoder_output_stride = [4]
            atrous_rates = [6, 12, 18]
        elif 'mobilenet' in config['model_variant']: 
            tf.flags.FLAGS.decoder_output_stride = None
            atrous_rates = None

        outputs_to_num_classes = {'semantic': 4, 'pos': 1, 'angle_sin': 1, 'angle_cos': 1, 'width': 1}

        self.model_options = dlc.ModelOptions(outputs_to_num_classes, atrous_rates=atrous_rates, crop_size=[480,640])


    def _create_model(self, x: tf.Tensor, is_training: bool) -> tf.Tensor:
        """
        Implement the architecture of your model
        :param x: input data
        :param is_training: flag if currently training
        :return: completely constructed model
        """
        def decoder_block(input_tensor: tf.Tensor, feature_maps: int, kernel_size: Tuple[int, int],
                          strides: Tuple[int, int], is_training: bool,
                          skip_connection: tf.Tensor = None) -> tf.Tensor:
            x = kl.Conv2DTranspose(feature_maps, kernel_size=kernel_size, strides=strides, padding='same')(input_tensor)
            x = kl.BatchNormalization()(x)
            x = kl.ReLU()(x)
            if skip_connection is not None:
                x = tf.keras.layers.concatenate([x, skip_connection])
            return x

        tf.logging.info('Constructing OAHO Model Deeplab Deconv, variant: {}'.format(self.model_options.model_variant))

        reuse = None
        weight_decay=0.0001
        fine_tune_batch_norm=is_training # False
        nas_training_hyper_parameters = None

        input_tensor = x
        # x = tf.keras.layers.concatenate([x,x,x])
        dl_feature_extractor._PREPROCESS_FN[self.model_options.model_variant] = OAHOModel._preprocess_robot_reach_zero_mean_unit_range
        features_encoder, endpoints = dlm.extract_features(x, self.model_options,
            weight_decay=weight_decay,
            reuse=reuse,
            is_training=is_training,
            fine_tune_batch_norm=fine_tune_batch_norm,
            nas_training_hyper_parameters=nas_training_hyper_parameters)
        features_decoder = dlm.refine_by_decoder(features_encoder,
            endpoints,
            crop_size=self.model_options.crop_size,
            decoder_output_stride=self.model_options.decoder_output_stride,
            model_variant=self.model_options.model_variant,
            weight_decay=weight_decay,
            reuse=reuse,
            is_training=is_training,
            fine_tune_batch_norm=fine_tune_batch_norm,
            use_bounded_activation=self.model_options.use_bounded_activation
            )

        # endpoints
        
        features_decoder = decoder_block(features_decoder, 256, kernel_size=3, strides=(2, 2), skip_connection=endpoints['root_block'], is_training=is_training)
        features_decoder = decoder_block(features_decoder, 128, kernel_size=3, strides=(2, 2), skip_connection=input_tensor, is_training=is_training)
        
        # x = dec
        # ===================================================================================================
        # Output layers
        # seg_head = kl.Conv2D(32, kernel_size=2, padding='same', name='seg_head_1', activation='relu')(features_decoder)
        seg_output = kl.Conv2D(4, kernel_size=2, padding='same', name='seg_out')(features_decoder)
        
        # grasp_head = kl.Conv2D(32, kernel_size=2, padding='same', name='grasp_head_1', activation='relu')(features_decoder)
        grasp_head = tf.keras.layers.concatenate([features_decoder, seg_output])
        pos_output = kl.Conv2D(1, kernel_size=2, padding='same', name='pos_out')(grasp_head)
        cos_output = kl.Conv2D(1, kernel_size=2, padding='same', name='cos_out')(grasp_head)
        sin_output = kl.Conv2D(1, kernel_size=2, padding='same', name='sin_out')(grasp_head)
        width_output = kl.Conv2D(1, kernel_size=2, padding='same', name='width_out')(grasp_head)
        return seg_output, pos_output, cos_output, sin_output, width_output