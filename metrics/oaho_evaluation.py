
"""Class for evaluating object detections with COCO metrics."""
import numpy as np
import tensorflow as tf
from utils.grasp import Grasp

# from object_detection.core import standard_fields
# from object_detection.metrics import coco_tools
# from object_detection.utils import object_detection_evaluation


# object_detection_evaluation.DetectionEvaluator):
class OAHODetectionEvaluator(object):
    """Class to evaluate COCO detection metrics."""

    def __init__(self ):
        """Constructor.

        Args:
          categories: A list of dicts, each of which has the following keys -
            'id': (required) an integer id uniquely identifying this category.
            'name': (required) string representing category name e.g., 'cat', 'dog'.
          include_metrics_per_category: If True, include metrics for each category.
          all_metrics_per_category: Whether to include all the summary metrics for
            each category in per_category_ap. Be careful with setting it to true if
            you have more than handful of categories, because it will pollute
            your mldash.
        """
        # super(CocoDetectionEvaluator, self).__init__(categories)
        # _image_ids is a dictionary that maps unique image ids to Booleans which
        # indicate whether a corresponding detection has been added.
        self._image_ids = {}
        self._groundtruth_list = []
        self._detection_grasps_list = []
        # self._category_id_set = set([cat['id'] for cat in self._categories])
        self._annotation_id = 1
        self._metrics = None
        # self._include_metrics_per_category = include_metrics_per_category
        # self._all_metrics_per_category = all_metrics_per_category

    def clear(self):
        """Clears the state to prepare for a fresh evaluation."""
        self._image_ids.clear()
        self._groundtruth_list = []
        self._detection_grasps_list = []

    def add_single_ground_truth_image_info(self, image_id, groundtruth_dict):
        """Adds groundtruth for a single image to be used for evaluation.

        If the image has already been added, a warning is logged, and groundtruth is
        ignored.

        Args:
          image_id: A unique string/integer identifier for the image.
          groundtruth_dict: A dictionary containing -
            InputDataFields.groundtruth_grasps: float32 numpy array of shape
              [num_grasps, 4] containing `num_grasps` groundtruth boxes of the format
              [ymin, xmin, ymax, xmax] in absolute image coordinates.
            InputDataFields.groundtruth_classes: integer numpy array of shape
              [num_grasps] containing 1-indexed groundtruth classes for the boxes.
        """
        if image_id in self._image_ids:
            tf.logging.warning('Ignoring ground truth with image id %s since it was '
                               'previously added', image_id)
            return

        self._groundtruth_list.extend(
            ExportSingleImageGroundtruth(
                image_id=image_id,
                next_annotation_id=self._annotation_id,
                groundtruth_grasps=groundtruth_dict['groundtruth_grasps'])
        )
        self._annotation_id += groundtruth_dict['groundtruth_grasps'].shape[0]
        self._image_ids[image_id] = False

    def add_single_detected_image_info(self, image_id, detections_dict):
        """Adds detections for a single image to be used for evaluation.

        If a detection has already been added for this image id, a warning is
        logged, and the detection is skipped.

        Args:
          image_id: A unique string/integer identifier for the image.
          detections_dict: A dictionary containing -
            DetectionResultFields.detection_grasps: float32 numpy array of shape
              [num_grasps, 4] containing `num_grasps` detection boxes of the format
              [ymin, xmin, ymax, xmax] in absolute image coordinates.
            DetectionResultFields.detection_scores: float32 numpy array of shape
              [num_grasps] containing detection scores for the boxes.
            DetectionResultFields.detection_classes: integer numpy array of shape
              [num_grasps] containing 1-indexed detection classes for the boxes.
            DetectionResultFields.detection_masks: optional uint8 numpy array of
              shape [num_grasps, image_height, image_width] containing instance
              masks for the boxes.

        Raises:
          ValueError: If groundtruth for the image_id is not available.
        """
        if image_id not in self._image_ids:
            raise ValueError(
                'Missing groundtruth for image id: {}'.format(image_id))

        if self._image_ids[image_id]:
            tf.logging.warning('Ignoring detection with image id %s since it was '
                               'previously added', image_id)
            return

        self._detection_grasps_list.extend(
            ExportSingleImageDetectionBoxesToCoco(
                image_id=image_id,
                detection_grasps=detections_dict['detection_grasps'])
            # detection_scores=detections_dict[]
        )
        self._image_ids[image_id] = True

    def evaluate(self):
        """Evaluates the detection boxes and returns a dictionary of coco metrics.

        Returns:
          A dictionary holding -

          1. summary_metrics:
          'DetectionBoxes_Precision/mAP': mean average precision over classes
            averaged over IOU thresholds ranging from .5 to .95 with .05
            increments.
          'DetectionBoxes_Precision/mAP@.50IOU': mean average precision at 50% IOU
          'DetectionBoxes_Precision/mAP@.75IOU': mean average precision at 75% IOU
          'DetectionBoxes_Precision/mAP (small)': mean average precision for small
            objects (area < 32^2 pixels).
          'DetectionBoxes_Precision/mAP (medium)': mean average precision for
            medium sized objects (32^2 pixels < area < 96^2 pixels).
          'DetectionBoxes_Precision/mAP (large)': mean average precision for large
            objects (96^2 pixels < area < 10000^2 pixels).
          'DetectionBoxes_Recall/AR@1': average recall with 1 detection.
          'DetectionBoxes_Recall/AR@10': average recall with 10 detections.
          'DetectionBoxes_Recall/AR@100': average recall with 100 detections.
          'DetectionBoxes_Recall/AR@100 (small)': average recall for small objects
            with 100.
          'DetectionBoxes_Recall/AR@100 (medium)': average recall for medium objects
            with 100.
          'DetectionBoxes_Recall/AR@100 (large)': average recall for large objects
            with 100 detections.

          2. per_category_ap: if include_metrics_per_category is True, category
          specific results with keys of the form:
          'Precision mAP ByCategory/category' (without the supercategory part if
          no supercategories exist). For backward compatibility
          'PerformanceByCategory' is included in the output regardless of
          all_metrics_per_category.
        """
        # groundtruth_dict = {
        #     'annotations': self._groundtruth_list,
        #     'images': [{'id': image_id} for image_id in self._image_ids]
        # }
        eval_dict = {image_id:{'id': image_id, 'groundtruth_grasps':[], 'detected_grasps':[]} for image_id in self._image_ids}

        for groundtruth in self._groundtruth_list:
          eval_dict[groundtruth['image_id']]['groundtruth_grasps'].append(groundtruth['grasp'])

        for detected in self._detection_grasps_list:
          eval_dict[detected['image_id']]['detected_grasps'].append(detected['grasp'])
        
        # detections_dict = {detection.image_id : detection for detection in self._detection_grasps_list}

        detections = []
        for image_id, eval_pack in eval_dict.items():
          if not eval_pack['groundtruth_grasps']:
            tf.logging.warning('Skipping evaluation of image id %s groundtruth missing', image_id)
            continue
          if not eval_pack['detected_grasps']:
            tf.logging.warning('Skipping evaluation of image id %s detection missing', image_id)
            continue
          
          groundtruth_grasps = [Grasp((g[1], g[0]), g[2], g[3], g[3]/2.0) for g in eval_pack['groundtruth_grasps']]
          groundtruth_bbs = [g.as_bb for g in groundtruth_grasps]

          detected_grasps = [Grasp((g[1], g[0]), g[2], g[3], g[3]/2.0) for g in eval_pack['detected_grasps']]

          max_iou_per_detected_grasp = [g.max_iou(groundtruth_bbs) for g in detected_grasps]
          if (np.array(max_iou_per_detected_grasp) > 0.25).any():
            detections.append(1.0)
            tf.logging.info('Eval image id %s SUCCESS', image_id)
          else:
            detections.append(0.0)
            tf.logging.info('Eval image id %s FAILURE', image_id)            
            tf.logging.info(max_iou_per_detected_grasp)
            tf.logging.info([(g.center, g.angle) for g in groundtruth_grasps])
            tf.logging.info('---- detected')
            tf.logging.info([(g.center, g.angle) for g in detected_grasps])
          

        box_metrics = {'mAP@0.25IOU': np.mean(detections)}
        return box_metrics

    @staticmethod
    def get_max_iou(groundtruth, detections):
        groundtruth_grasps = [Grasp((g[1], g[0]), g[2], g[3], g[3]/2.0) for g in groundtruth]]
        groundtruth_bbs = [g.as_bb for g in groundtruth_grasps]

        detected_grasps = [Grasp((g[1], g[0]), g[2], g[3], g[3]/2.0) for g in detections]]

        max_iou_per_detected_grasp = [g.max_iou(groundtruth_bbs) for g in detected_grasps]
        return max_iou_per_detected_grasp

    def get_estimator_eval_metric_ops(self, eval_dict):
        """Returns a dictionary of eval metric ops to use with `tf.EstimatorSpec`.

        Note that once value_op is called, the detections and groundtruth added via
        update_op are cleared.

        Args:
          image_id: Unique string/integer identifier for the image.
          groundtruth_grasps: float32 tensor of shape [num_grasps, 4] containing
            `num_grasps` groundtruth boxes of the format
            [ymin, xmin, ymax, xmax] in absolute image coordinates.
          groundtruth_classes: int32 tensor of shape [num_grasps] containing
            1-indexed groundtruth classes for the boxes.
          detection_grasps: float32 tensor of shape [num_grasps, 4] containing
            `num_grasps` detection boxes of the format [ymin, xmin, ymax, xmax]
            in absolute image coordinates.
          detection_scores: float32 tensor of shape [num_grasps] containing
            detection scores for the boxes.
          detection_classes: int32 tensor of shape [num_grasps] containing
            1-indexed detection classes for the boxes.

        Returns:
          a dictionary of metric names to tuple of value_op and update_op that can
          be used as eval metric ops in tf.EstimatorSpec. Note that all update ops
          must be run together and similarly all value ops must be run together to
          guarantee correct behaviour.
        """
        def update_op( image_id_batched, groundtruth_grasps_batched, detection_grasps_batched ):
            for (image_id, groundtruth_grasps, detection_grasps) in zip(
               image_id_batched, groundtruth_grasps_batched, detection_grasps_batched):
              self.add_single_ground_truth_image_info( image_id, {'groundtruth_grasps': groundtruth_grasps} )
              self.add_single_detected_image_info( image_id, {'detection_grasps': detection_grasps}) 

        image_id = eval_dict['image_id']
        groundtruth_grasps = eval_dict['groundtruth_grasps']
        detection_grasps = eval_dict['detection_grasps']

        update_op = tf.py_func(update_op, [image_id, groundtruth_grasps, detection_grasps], [])
        metric_names = ['mAP@0.25IOU']

        def first_value_func():
            self._metrics = self.evaluate()
            self.clear()
            return np.float32(self._metrics[metric_names[0]])

        def value_func_factory(metric_name):
            def value_func():
                return np.float32(self._metrics[metric_name])
            return value_func

        first_value_op = tf.py_func(first_value_func, [], tf.float32)
        eval_metric_ops = {metric_names[0]: (first_value_op, update_op)}
        with tf.control_dependencies([first_value_op]):
            for metric_name in metric_names[1:]:
                eval_metric_ops[metric_name] = (tf.py_func(
                    value_func_factory(metric_name), [], np.float32), update_op)
        return eval_metric_ops


def ExportSingleImageGroundtruth(image_id, next_annotation_id, groundtruth_grasps):
    """Export groundtruth of a single image to COCO format.

    This function converts groundtruth detection annotations represented as numpy
    arrays to dictionaries that can be ingested by the COCO evaluation API. Note
    that the image_ids provided here must match the ones given to
    ExportSingleImageDetectionsToCoco. We assume that boxes and classes are in
    correspondence - that is: groundtruth_grasps[i, :], and
    groundtruth_classes[i] are associated with the same groundtruth annotation.

    In the exported result, "area" fields are always set to the area of the
    groundtruth bounding box and "iscrowd" fields are always set to 0.
    TODO: pass in "iscrowd" array for evaluating on COCO dataset.

    Args:
      image_id: a unique image identifier either of type integer or string.
      next_annotation_id: integer specifying the first id to use for the
        groundtruth annotations. All annotations are assigned a continuous integer
        id starting from this value.
      category_id_set: A set of valid class ids. Groundtruth with classes not in
        category_id_set are dropped.
      groundtruth_grasps: numpy array (float32) with shape [num_gt_boxes, 4]
      groundtruth_classes: numpy array (int) with shape [num_gt_boxes]
      groundtruth_masks: optional uint8 numpy array of shape [num_detections,
        image_height, image_width] containing detection_masks.

    Returns:
      a list of groundtruth annotations for a single image in the COCO format.

    Raises:
      ValueError: if (1) groundtruth_grasps and groundtruth_classes do not have the
        right lengths or (2) if each of the elements inside these lists do not
        have the correct shapes or (3) if image_ids are not integers
    """
    
    if len(groundtruth_grasps.shape) != 2:
        raise ValueError('groundtruth_grasps is expected to be of '
                         'rank 2.')
    if groundtruth_grasps.shape[1] != 4:
        raise ValueError('groundtruth_grasps should have '
                         'shape[1] == 4.')
    num_grasps = groundtruth_grasps.shape[0]

    groundtruth_list = []
    for i in range(num_grasps):
        export_dict = {
            'id': next_annotation_id + i,
            'image_id': image_id,
            'grasp': groundtruth_grasps[i, :]
        }
        if (groundtruth_grasps[i, 0:2] < 0).any():
            continue
        groundtruth_list.append(export_dict)
    return groundtruth_list


def ExportSingleImageDetectionBoxesToCoco(image_id, detection_grasps, detection_scores=None ):
    """Export detections of a single image to COCO format.

    This function converts detections represented as numpy arrays to dictionaries
    that can be ingested by the COCO evaluation API. Note that the image_ids
    provided here must match the ones given to the
    ExporSingleImageDetectionBoxesToCoco. We assume that boxes, and classes are in
    correspondence - that is: boxes[i, :], and classes[i]
    are associated with the same groundtruth annotation.

    Args:
      image_id: unique image identifier either of type integer or string.
      category_id_set: A set of valid class ids. Detections with classes not in
        category_id_set are dropped.
      detection_grasps: float numpy array of shape [num_detections, 4] containing
        detection boxes.
      detection_scores: float numpy array of shape [num_detections] containing
        scored for the detection boxes.
      detection_classes: integer numpy array of shape [num_detections] containing
        the classes for detection boxes.

    Returns:
      a list of detection annotations for a single image in the COCO format.

    Raises:
      ValueError: if (1) detection_grasps, detection_scores and detection_classes
        do not have the right lengths or (2) if each of the elements inside these
        lists do not have the correct shapes or (3) if image_ids are not integers.
    """

    if len(detection_grasps.shape) != 2:
        raise ValueError('All entries in detection_grasps expected to be of '
                         'rank 2.')
    if detection_grasps.shape[1] != 4:
        raise ValueError('All entries in detection_grasps should have '
                         'shape[1] == 4.')
    num_grasps = detection_grasps.shape[0]

    detections_list = []
    for i in range(num_grasps):
        detections_list.append({
            'image_id': image_id,
            'grasp': detection_grasps[i, :],
            'score': float(detection_scores[i]) if detection_scores is not None else 1
        })
    return detections_list
