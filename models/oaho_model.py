import tensorflow as tf
from tensorflow.keras import layers as kl
from tensorflow.keras import applications as ka
from base.model import BaseModel
from typing import Dict, Tuple
import sys
from metrics.oaho_evaluation import OAHODetectionEvaluator
from utils.oaho_visualization import OAHODetectionVisualizer


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

        # initialise model architecture
        seg_output, pos_output, cos_output, sin_output, width_output = self._create_model(features['input'], is_training)
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

        # tf.print(labels['grasps'], output_stream=sys.stdout)
        # print(labels['grasps'])
        # grasp_loss_mask = labels['quality']
        grasp_loss_mask = tf.to_float(tf.greater(labels['quality'], tf.zeros_like(labels['quality'])))
        # quality_loss = tf.losses.mean_squared_error(labels=labels['quality'], predictions=pos_output)
        quality_loss = tf.losses.sigmoid_cross_entropy(labels['quality'], logits=pos_output)

        sin_loss = tf.losses.mean_squared_error(labels=labels['angle_sin'], predictions=sin_output, weights=grasp_loss_mask)
        cos_loss = tf.losses.mean_squared_error(labels=labels['angle_cos'], predictions=cos_output, weights=grasp_loss_mask)
        width_loss = tf.losses.mean_squared_error(labels=(labels['gripper_width'] / 150.0), predictions=width_output, weights=grasp_loss_mask)

        angle = 0.5 * tf.atan2(sin_output, cos_output)

        loss = seg_loss + quality_loss + sin_loss + cos_loss + width_loss

        # TODO: update summaries for tensorboard
        segmentations_class_colors = tf.convert_to_tensor([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=tf.uint8)
        segmentation_image = tf.gather(segmentations_class_colors, segmentation_classes)

        tf.summary.scalar('seg_loss', seg_loss)
        tf.summary.scalar('quality_loss', quality_loss)
        tf.summary.scalar('angle_sin_loss', sin_loss)
        tf.summary.scalar('angle_cos_loss', cos_loss)
        tf.summary.scalar('width_loss', width_loss)
        tf.summary.scalar('loss', loss)

        gaussian_blur_kernel = gaussian_kernel(5,0.0,1.0)
        gaussian_blur_kernel = gaussian_blur_kernel[:,:,tf.newaxis, tf.newaxis]
        quality = tf.sigmoid(pos_output)
        quality = tf.nn.conv2d(quality, gaussian_blur_kernel, [1,1,1,1], 'SAME')
        

        images = {
            'input': tf.summary.image('input', features['input']),
            'segmentation': tf.summary.image('segmentation', segmentation_image),
            'quality': tf.summary.image('quality', quality),
            'angle_sin': tf.summary.image('angle_sin', sin_output),
            'angle_cos': tf.summary.image('angle_cos', cos_output),
            'width': tf.summary.image('width', width_output),
            'angle': tf.summary.image('angle', angle)
        }

	    # tf.summary.image('segmentation', tf.image.hsv_to_rgb(predictions['segmentation'] / 4.0))
        # tf.summary.image('segmentation', tf.cast(predictions['segmentation'], tf.float32))



        if mode == tf.estimator.ModeKeys.EVAL:
            # TODO: update evaluation metrics
            # output eval images
            # eval_summary_hook = tf.train.SummarySaverHook(summary_op=images, save_secs=120)
            summaries_dict = {
                'val_mean_iou': tf.metrics.mean_iou(
                    labels['seg'], predictions=predictions['segmentation'], num_classes=4
                )
            }
            # summaries_dict.update(images)
            # summaries_dict.update(get_estimator_eval_metric_ops)
            b = tf.shape(quality)[0]

            detection_grasps = self._create_detection_head(quality, angle, width_output)

            detection_evaluator = OAHODetectionEvaluator()
            detection_visualizer = OAHODetectionVisualizer()
            groundtruth_grasps = tf.reshape(tf.sparse_tensor_to_dense (labels['grasps'], -1), (b, -1, 4))
            groundtruth_segmentation_image = tf.gather(segmentations_class_colors, tf.squeeze(labels['seg'], -1))

            summaries_dict.update(detection_evaluator.get_estimator_eval_metric_ops({
                                                                                    'image_id': labels['id'],
                                                                                    'groundtruth_grasps': groundtruth_grasps,
                                                                                    'detection_grasps' : detection_grasps
                                                                                    }))

            normalized_depth = tf.image.convert_image_dtype(features['input'] / tf.reduce_max(features['input'], axis=[1,2], keepdims=True) , dtype=tf.uint8)
            summaries_dict.update(detection_visualizer.get_estimator_eval_metric_ops({
                                                                                'image_id': labels['id'],
                                                                                'depth': normalized_depth,
                                                                                'groundtruth_grasps': groundtruth_grasps,
                                                                                'detection_grasps' : detection_grasps,
                                                                                'groundtruth_segmentation': groundtruth_segmentation_image,
                                                                                'detection_segmentation': segmentation_image
            }))
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, eval_metric_ops=summaries_dict#, evaluation_hooks=[eval_summary_hook]
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
    def encoder_block(input_tensor: tf.Tensor, feature_maps: int, kernel_size: Tuple[int, int], strides: Tuple[int, int], is_training: bool) -> tf.Tensor:
        x = kl.Conv2D(feature_maps, kernel_size=kernel_size, strides=strides, padding='same')(input_tensor)
        x = kl.BatchNormalization()(x, is_training)
        x = kl.ReLU()(x)
        return x


    @staticmethod
    def _create_detection_head(quality: tf.Tensor, angle: tf.Tensor, width: tf.Tensor) -> tf.Tensor:        

        # b,h,w,d = pos_output.shape
        # quality = 
        out_shape = tf.shape(quality)
        b, h, w, d = out_shape[0], out_shape[1], out_shape[2], out_shape[3]

        # flatten the Tensor along the height and width axes
        flat_tensor = tf.reshape(quality, (b, -1, d))

        # argmax of the flat tensor
        argmax = tf.cast(tf.argmax(flat_tensor, axis=1), tf.int32)

        # convert indexes into 2D coordinates
        argmax_x = argmax % w # // w
        argmax_y = argmax // w # % w

        angle_avg = kl.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(angle)
        width_avg = kl.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(width)

        # stack and return 2D coordinates
        grasp_center = tf.stack((argmax_x, argmax_y), axis=1)
        grasp_angle = tf.batch_gather(tf.reshape(angle_avg, (b, -1,d)), argmax)#, tf.expand_dims(argmax,-1))   #tf.gather_nd(tf.transpose(angle, [1,2,0,3]), grasp_center)
        grasp_width = 150.0 * tf.batch_gather(tf.reshape(width_avg, (b, -1,d)), argmax)#, tf.expand_dims(argmax,-1))

        detection_grasps = tf.concat([tf.cast(grasp_center,tf.float32), grasp_angle, grasp_width], axis=1)
        detection_grasps = tf.reshape(detection_grasps, (b, -1, 4))
        return detection_grasps


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


        tf.logging.info("Constructing OAHO Model")

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



def gaussian_kernel(size: int, mean: float, std: float):
    """Makes 2D gaussian Kernel for convolution."""
    d = tf.distributions.Normal(mean, std)
    vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))
    gauss_kernel = tf.einsum('i,j->ij', vals, vals)
    return gauss_kernel / tf.reduce_sum(gauss_kernel)