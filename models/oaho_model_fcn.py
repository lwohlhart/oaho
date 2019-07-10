import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, ReLU
from base.model import BaseModel
from typing import Dict, Tuple


class OAHOModel(BaseModel):
    def __init__(self, config: dict) -> None:
        """
        :param config: global configuration
        """
        super().__init__(config)

    def model(
        self, features: Dict[str, tf.Tensor], labels: Dict[str, tf.Tensor], mode: str
    ) -> tf.Tensor:
        """
        Define your model metrics and architecture, the logic is dependent on the mode.
        :param features: A dictionary of potential inputs for your model
        :param labels: Input label set
        :param mode: Current training mode (train, test, predict)
        :return: An estimator spec used by the higher level API
        """
        # set flag if the model is currently training
        is_training = mode == tf.estimator.ModeKeys.TRAIN

        # get input data
        x = features['input']

        # TODO: create graph
        # initialise model architecture
        seg_output, pos_output, cos_output, sin_output, width_output = self._create_model(x, is_training)

        segmentation_classes = tf.argmax(input=seg_output, axis=3, output_type=tf.int32)
        # TODO: update model predictions
        predictions = {
            'segmentation': tf.expand_dims(segmentation_classes,-1),
            'segmentation_probabilities': tf.nn.softmax(seg_output),
        }
#
        if mode == tf.estimator.ModeKeys.PREDICT:
            # TODO: update output during serving
            export_outputs = {
                'segmentation': tf.estimator.export.ClassificationOutput(
		    scores=predictions['segmentation_probabilities'],
		    classes=tf.cast(predictions['segmentation'], tf.string)	
                )
            }
            return tf.estimator.EstimatorSpec(
                mode, predictions=predictions, export_outputs=export_outputs
            )

        # calculate loss
        # specify some class weightings
        segmentation_class_weights = tf.constant([1, 5, 1, 5])

        # specify the weights for each sample in the batch (without having to compute the onehot label matrix)
        segmentation_weights = tf.gather(segmentation_class_weights, labels['seg'])

        seg_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels['seg'], logits=seg_output, weights=segmentation_weights)
        # seg_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels['seg'], logits=seg_output)
        
        # grasp_loss_mask = labels['quality']
        grasp_loss_mask = tf.to_float(tf.greater(labels['quality'], tf.zeros_like(labels['quality'])))
        # quality_loss = tf.losses.mean_squared_error(labels=labels['quality'], predictions=pos_output)
        quality_loss = tf.losses.sigmoid_cross_entropy(labels['quality'], logits=pos_output)

        sin_loss = tf.losses.mean_squared_error(labels=labels['angle_sin'], predictions=sin_output, weights=grasp_loss_mask)
        cos_loss = tf.losses.mean_squared_error(labels=labels['angle_cos'], predictions=cos_output, weights=grasp_loss_mask)
        width_loss = tf.losses.mean_squared_error(labels=labels['gripper_width'], predictions=width_output, weights=grasp_loss_mask)

        loss = seg_loss + quality_loss + sin_loss + cos_loss # + width_loss

        # TODO: update summaries for tensorboard
        segmentations_class_colors = tf.convert_to_tensor([[0,0,0],[255,0,0],[0,255,0],[0,0,255]], dtype=tf.uint8)
        
        tf.summary.scalar('seg_loss', seg_loss)
        tf.summary.scalar('quality_loss', quality_loss)
        tf.summary.scalar('angle_sin_loss', sin_loss)
        tf.summary.scalar('angle_cos_loss', cos_loss)
        tf.summary.scalar('width_loss', width_loss)
        tf.summary.scalar('loss', loss)
        tf.summary.image('input', x)
	    #tf.summary.image('segmentation', tf.image.hsv_to_rgb(predictions['segmentation'] / 4.0))
        # tf.summary.image('segmentation', tf.cast(predictions['segmentation'], tf.float32))
        tf.summary.image('segmentation', tf.gather(segmentations_class_colors, segmentation_classes))
        tf.summary.image('quality', tf.sigmoid(pos_output))
        tf.summary.image('angle_sin', sin_output)
        tf.summary.image('angle_cos', cos_output)

        if mode == tf.estimator.ModeKeys.EVAL:
            # TODO: update evaluation metrics
            summaries_dict = {
                'val_mean_iou': tf.metrics.mean_iou(
                    labels['seg'], predictions=predictions['segmentation'], num_classes=4
                )
            }
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, eval_metric_ops=summaries_dict
            )

        # assert only reach this point during training
        assert mode == tf.estimator.ModeKeys.TRAIN

        # create learning rate variable for hyper param tuning
        lr = tf.Variable(
            initial_value=self.config['learning_rate'], name='learning-rate'
        )

        # TODO: update optimiser
        optimizer = tf.train.AdamOptimizer(lr)

        train_op = optimizer.minimize(
            loss,
            global_step=tf.train.get_global_step(),
            colocate_gradients_with_ops=True,
        )

        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

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

    @staticmethod
    def _create_model(x: tf.Tensor, is_training: bool) -> tf.Tensor:
        """
        Implement the architecture of your model
        :param x: input data
        :param is_training: flag if currently training
        :return: completely constructed model
        """

        no_filters = [10, 32, 128, 512]
        filter_sizes= [(3, 3), (3, 3), (3, 3), (2, 2)]
        input_tensor = x

        enc1 = OAHOModel.encoder_block(input_tensor, no_filters[0], kernel_size=filter_sizes[0], strides=(2, 2))
        enc2 = OAHOModel.encoder_block(enc1, no_filters[1], kernel_size=filter_sizes[1], strides=(2, 2))
        enc3 = OAHOModel.encoder_block(enc2, no_filters[2], kernel_size=filter_sizes[2], strides=(2, 2))
        enc4 = OAHOModel.encoder_block(enc3, no_filters[3], kernel_size=filter_sizes[3], strides=(2, 2))

        dec0 = OAHOModel.decoder_block(enc4, no_filters[3], kernel_size=filter_sizes[3], strides=(2, 2), skip_connection=enc3)
        dec1 = OAHOModel.decoder_block(dec0, no_filters[2], kernel_size=filter_sizes[2], strides=(2, 2), skip_connection=enc2)
        dec2 = OAHOModel.decoder_block(dec1, no_filters[1], kernel_size=filter_sizes[1], strides=(2, 2), skip_connection=enc1)
        dec3 = OAHOModel.decoder_block(dec2, no_filters[0], kernel_size=filter_sizes[0], strides=(2, 2), skip_connection=input_tensor)
        
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
