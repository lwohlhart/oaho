# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A set of functions that are used for visualization.

These functions often receive an image, perform some visualization on the image.
The functions do not return a value, instead they modify the image itself.

"""
import abc
import collections
# Set headless-friendly backend.
import matplotlib; matplotlib.use('Agg')  # pylint: disable=multiple-statements
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
import numpy as np
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import six
import tensorflow as tf
from utils.grasp import Grasp

# from object_detection.core import standard_fields as fields
# from object_detection.utils import shape_utils


def save_image_array_as_png(image, output_path):
  """Saves an image (represented as a numpy array) to PNG.

  Args:
    image: a numpy array with shape [height, width, 3].
    output_path: path to which image should be written.
  """
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  with tf.gfile.Open(output_path, 'w') as fid:
    image_pil.save(fid, 'PNG')


def encode_image_array_as_png_str(image):
  """Encodes a numpy array into a PNG string.

  Args:
    image: a numpy array with shape [height, width, 3].

  Returns:
    PNG encoded image string.
  """
  image_pil = Image.fromarray(np.uint8(image))
  output = six.BytesIO()
  image_pil.save(output, format='PNG')
  png_string = output.getvalue()
  output.close()
  return png_string


def draw_bounding_box_on_image_array(image,
                                     center_x,
                                     center_y,
                                     angle,
                                     width,
                                     color='red',
                                     thickness=4,
                                     display_str_list=(),
                                     use_normalized_coordinates=True):
  """Adds a bounding box to an image (numpy array).

  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.

  Args:
    image: a numpy array with shape [height, width, 3].
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  draw_bounding_box_on_image(image_pil, center_x, center_y, angle, width, color,
                             thickness, display_str_list,
                             use_normalized_coordinates)
  np.copyto(image, np.array(image_pil))


def draw_bounding_box_on_image(image,
                               center_x,
                               center_y,
                               angle,
                               width,
                               color='red',
                               thickness=4,
                               display_str_list=(),
                               use_normalized_coordinates=True):
  """Adds a bounding box to an image.

  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.

  Each string in display_str_list is displayed on a separate line above the
  bounding box in black text on a rectangle filled with the input 'color'.
  If the top of the bounding box extends to the edge of the image, the strings
  are displayed below the bounding box.

  Args:
    image: a PIL.Image object.
    center_x: center_x of grasp recangle.
    center_y: center_y of grasp recangle.
    angle: angle of grasp recangle.
    width: width of grasp recangle.
    color: color to draw grasp rectangle. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
  if center_x < 0 or center_y < 0:
      return # invalid grasp definition (usually from sparse to dense annotation conversion)

  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  center = (center_y, center_x)
  bb = Grasp(center, angle, width, width/2.0).as_bb
#   if use_normalized_coordinates:
#     (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
#                                   ymin * im_height, ymax * im_height)
#   else:
#     (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
  points = tuple((t[1], t[0]) for t in bb.points)
  points = points + (points[0])

  draw.line(points, width=thickness, fill=color)

"""
  try:
    font = ImageFont.truetype('arial.ttf', 24)
  except IOError:
    font = ImageFont.load_default()
  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = bottom + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle(
        [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                          text_bottom)],
        fill=color)
    draw.text(
        (left + margin, text_bottom - text_height - margin),
        display_str,
        fill='black',
        font=font)
    text_bottom -= text_height - 2 * margin"""


def draw_bounding_boxes_on_image_array(image,
                                       boxes,
                                       color='red',
                                       thickness=4,
                                       display_str_list_list=()):
  """Draws bounding boxes on image (numpy array).

  Args:
    image: a numpy array object.
    boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
           The coordinates are in normalized format between [0, 1].
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list_list: list of list of strings.
                           a list of strings for each bounding box.
                           The reason to pass a list of strings for a
                           bounding box is that it might contain
                           multiple labels.

  Raises:
    ValueError: if boxes is not a [N, 4] array
  """
  image_pil = Image.fromarray(image)
  draw_bounding_boxes_on_image(image_pil, boxes, color, thickness,
                               display_str_list_list)
  np.copyto(image, np.array(image_pil))


def draw_bounding_boxes_on_image(image,
                                 boxes,
                                 color='red',
                                 thickness=4,
                                 display_str_list_list=()):
  """Draws bounding boxes on image.

  Args:
    image: a PIL.Image object.
    boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
           The coordinates are in normalized format between [0, 1].
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list_list: list of list of strings.
                           a list of strings for each bounding box.
                           The reason to pass a list of strings for a
                           bounding box is that it might contain
                           multiple labels.

  Raises:
    ValueError: if boxes is not a [N, 4] array
  """
  boxes_shape = boxes.shape
  if not boxes_shape:
    return
  if len(boxes_shape) != 2 or boxes_shape[1] != 4:
    raise ValueError('Input must be of size [N, 4]')
  for i in range(boxes_shape[0]):
    display_str_list = ()
    if display_str_list_list:
      display_str_list = display_str_list_list[i]
    draw_bounding_box_on_image(image, boxes[i, 0], boxes[i, 1], boxes[i, 2],
                               boxes[i, 3], color, thickness, display_str_list)


def create_visualization_fn(include_segmentation=False, **kwargs):
  """Constructs a visualization function that can be wrapped in a py_func.

  py_funcs only accept positional arguments. This function returns a suitable
  function with the correct positional argument mapping. The positional
  arguments in order are:
  0: image
  1: boxes
  2: classes
  3: scores
  [4-6]: masks (optional)
  [4-6]: keypoints (optional)
  [4-6]: track_ids (optional)

  -- Example 1 --
  vis_only_masks_fn = create_visualization_fn(category_index,
    include_masks=True, include_keypoints=False, include_track_ids=False,
    **kwargs)
  image = tf.py_func(vis_only_masks_fn,
                     inp=[image, boxes, classes, scores, masks],
                     Tout=tf.uint8)

  -- Example 2 --
  vis_masks_and_track_ids_fn = create_visualization_fn(category_index,
    include_masks=True, include_keypoints=False, include_track_ids=True,
    **kwargs)
  image = tf.py_func(vis_masks_and_track_ids_fn,
                     inp=[image, boxes, classes, scores, masks, track_ids],
                     Tout=tf.uint8)

  Args:
    category_index: a dict that maps integer ids to category dicts. e.g.
      {1: {1: 'dog'}, 2: {2: 'cat'}, ...}
    include_masks: Whether masks should be expected as a positional argument in
      the returned function.
    include_keypoints: Whether keypoints should be expected as a positional
      argument in the returned function.
    include_track_ids: Whether track ids should be expected as a positional
      argument in the returned function.
    **kwargs: Additional kwargs that will be passed to
      visualize_boxes_and_labels_on_image_array.

  Returns:
    Returns a function that only takes tensors as positional arguments.
  """

  def visualization_py_func_fn(*args):
    """Visualization function that can be wrapped in a tf.py_func.

    Args:
      *args: First 4 positional arguments must be:
        image - uint8 numpy array with shape (img_height, img_width, 3).
        boxes - a numpy array of shape [N, 4].
        classes - a numpy array of shape [N].
        scores - a numpy array of shape [N] or None.
        -- Optional positional arguments --
        instance_masks - a numpy array of shape [N, image_height, image_width].
        keypoints - a numpy array of shape [N, num_keypoints, 2].
        track_ids - a numpy array of shape [N] with unique track ids.

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3) with overlaid
      boxes.
    """
    image = args[0]
    boxes = args[1]
    segmentation = None
    pos_arg_ptr = 2  # Positional argument for first optional tensor (masks).
    if include_segmentation:
      segmentation = args[pos_arg_ptr]
      pos_arg_ptr += 1

    return visualize_boxes_and_labels_on_image_array(
        image,
        boxes,
        segmentation=segmentation,
        **kwargs)
  return visualization_py_func_fn


def _resize_original_image(image, image_shape):
  image = tf.expand_dims(image, 0)
  image = tf.image.resize_images(
      image,
      image_shape,
      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
      align_corners=True)
  return tf.cast(tf.squeeze(image, 0), tf.uint8)


# images_with_groundtruth = draw_bounding_boxes_on_image_tensors(
#         tf.expand_dims(
#             eval_dict['depth'][indx], axis=0),
#         tf.expand_dims(
#             eval_dict['detection_grasps'][indx], axis=0),
#         segmentation=groundtruth_segmentation,
#         max_boxes_to_draw=None,
#         min_score_thresh=0.0
#         )



def draw_bounding_boxes_on_image_tensors(images,
                                         boxes,
                                         segmentation=None,
                                         max_boxes_to_draw=20,
                                         min_score_thresh=0.2,
                                         ):
  """Draws bounding boxes, masks, and keypoints on batch of image tensors.

  Args:
    images: A 4D uint8 image tensor of shape [N, H, W, C]. If C > 3, additional
      channels will be ignored. If C = 1, then we convert the images to RGB
      images.
    boxes: [N, max_detections, 4] float32 tensor of detection boxes.
    classes: [N, max_detections] int tensor of detection classes. Note that
      classes are 1-indexed.
    scores: [N, max_detections] float32 tensor of detection scores.
    category_index: a dict that maps integer ids to category dicts. e.g.
      {1: {1: 'dog'}, 2: {2: 'cat'}, ...}
    original_image_spatial_shape: [N, 2] tensor containing the spatial size of
      the original image.
    true_image_shape: [N, 3] tensor containing the spatial size of unpadded
      original_image.
    segmentation: A 4D uint8 tensor of shape [N, max_detection, H, W] with
      instance masks.
    keypoints: A 4D float32 tensor of shape [N, max_detection, num_keypoints, 2]
      with keypoints.
    track_ids: [N, max_detections] int32 tensor of unique tracks ids (i.e.
      instance ids for each object). If provided, the color-coding of boxes is
      dictated by these ids, and not classes.
    max_boxes_to_draw: Maximum number of boxes to draw on an image. Default 20.
    min_score_thresh: Minimum score threshold for visualization. Default 0.2.
    use_normalized_coordinates: Whether to assume boxes and kepoints are in
      normalized coordinates (as opposed to absolute coordiantes).
      Default is True.

  Returns:
    4D image tensor of type uint8, with boxes drawn on top.
  """
  # Additional channels are being ignored.
  if images.shape[3] > 3:
    images = images[:, :, :, 0:3]
  elif images.shape[3] == 1:
    images = tf.image.grayscale_to_rgb(images)
  visualization_keyword_args = {
      'use_normalized_coordinates': False,
      'max_boxes_to_draw': max_boxes_to_draw,
      'min_score_thresh': min_score_thresh,
      'agnostic_mode': False,
      'line_thickness': 4
  }
  visualize_boxes_fn = create_visualization_fn(
      include_segmentation=segmentation is not None,
      **visualization_keyword_args)

  elems = [images, boxes]
  if segmentation is not None:
    elems.append(segmentation)

  def draw_boxes(image_and_detections):
    """Draws boxes on image."""
    image_with_boxes = tf.py_func(visualize_boxes_fn, image_and_detections,
                                  tf.uint8)
    return image_with_boxes

  images = tf.map_fn(draw_boxes, elems, dtype=tf.uint8, back_prop=False)
  return images


def draw_side_by_side_evaluation_image(eval_dict,
                                       max_boxes_to_draw=20,
                                       min_score_thresh=0.2,
                                       use_normalized_coordinates=True):
  """Creates a side-by-side image with detections and groundtruth.

  Bounding boxes (and instance masks, if available) are visualized on both
  subimages.

  Args:
    eval_dict: The evaluation dictionary returned by
      eval_util.result_dict_for_batched_example() or
      eval_util.result_dict_for_single_example().
    category_index: A category index (dictionary) produced from a labelmap.
    max_boxes_to_draw: The maximum number of boxes to draw for detections.
    min_score_thresh: The minimum score threshold for showing detections.
    use_normalized_coordinates: Whether to assume boxes and kepoints are in
      normalized coordinates (as opposed to absolute coordiantes).
      Default is True.

  Returns:
    A list of [1, H, 2 * W, C] uint8 tensor. The subimage on the left
      corresponds to detections, while the subimage on the right corresponds to
      groundtruth.
  """
#   detection_fields = fields.DetectionResultFields()
#   input_data_fields = fields.InputDataFields()

  images_with_detections_list = []

  # Add the batch dimension if the eval_dict is for single example.
  # TODO check which tensor to use for batch evaluation
#   if len(eval_dict[detection_fields.detection_classes].shape) == 1:
#     for key in eval_dict:
#       if key != 'depth':
#         eval_dict[key] = tf.expand_dims(eval_dict[key], 0)

  for indx in range(eval_dict['depth'].shape[0]):
    detection_segmentation = None
    if 'detection_segmentation' in eval_dict:
      detection_segmentation = tf.cast(
          tf.expand_dims(eval_dict['detection_segmentation'][indx], axis=0), tf.uint8)
    groundtruth_segmentation = None
    if 'groundtruth_segmentation' in eval_dict:
      groundtruth_segmentation = tf.cast(
          tf.expand_dims(eval_dict['groundtruth_segmentation'][indx], axis=0), tf.uint8)

    images_with_detections = draw_bounding_boxes_on_image_tensors(
        tf.expand_dims(eval_dict['depth'][indx], axis=0),
        tf.expand_dims(eval_dict['detection_grasps'][indx], axis=0),
        segmentation=detection_segmentation,
        max_boxes_to_draw=max_boxes_to_draw,
        min_score_thresh=min_score_thresh
        )
    images_with_groundtruth = draw_bounding_boxes_on_image_tensors(
        tf.expand_dims(eval_dict['depth'][indx], axis=0),
        tf.expand_dims(eval_dict['groundtruth_grasps'][indx], axis=0),
        segmentation=groundtruth_segmentation,
        max_boxes_to_draw=None,
        min_score_thresh=0.0
        )
    images_with_detections_list.append(
        tf.concat([images_with_detections, images_with_groundtruth], axis=2))
  return images_with_detections_list


def draw_mask_on_image_array(image, mask, color='red', alpha=0.4):
  """Draws mask on an image.

  Args:
    image: uint8 numpy array with shape (img_height, img_height, 3)
    mask: a uint8 numpy array of shape (img_height, img_height) with
      values between either 0 or 1.
    color: color to draw the keypoints with. Default is red.
    alpha: transparency value between 0 and 1. (default: 0.4)

  Raises:
    ValueError: On incorrect data type for image or masks.
  """
  if image.dtype != np.uint8:
    raise ValueError('`image` not of type np.uint8')
  if mask.dtype != np.uint8:
    raise ValueError('`mask` not of type np.uint8')
  if np.any(np.logical_and(mask != 1, mask != 0)):
    raise ValueError('`mask` elements should be in [0, 1]')
  if image.shape[:2] != mask.shape:
    raise ValueError('The image has spatial dimensions %s but the mask has '
                     'dimensions %s' % (image.shape[:2], mask.shape))
  rgb = ImageColor.getrgb(color)
  pil_image = Image.fromarray(image)

  solid_color = np.expand_dims(
      np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
  pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert('RGBA')
  pil_mask = Image.fromarray(np.uint8(255.0*alpha*mask)).convert('L')
  pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
  np.copyto(image, np.array(pil_image.convert('RGB')))

def draw_segmentation_on_image_array(image, segmentation, alpha=0.4):
  """Draws mask on an image.
  Args:
    image: uint8 numpy array with shape (img_height, img_height, 3)
    mask: a uint8 numpy array of shape (img_height, img_height) with
      values between either 0 or 1.
    color: color to draw the keypoints with. Default is red.
    alpha: transparency value between 0 and 1. (default: 0.4)

  Raises:
    ValueError: On incorrect data type for image or masks.
  """
  if image.dtype != np.uint8:
    raise ValueError('`image` not of type np.uint8')
  if segmentation.dtype != np.uint8:
    raise ValueError('`mask` not of type np.uint8')

  # rgb = ImageColor.getrgb(color)
  """pil_image = Image.fromarray(image)
  pil_segmentation = Image.fromarray(segmentation)
  pil_mask = Image.fromarray(np.uint8(255.0*alpha * (np.max(segmentation, axis=2) > 0.0)))
  pil_image = Image.composite(pil_image, pil_segmentation, pil_mask)"""

  pil_image = Image.fromarray(image)
  pil_segmentation = Image.fromarray(segmentation).convert('RGBA')
  pil_mask = Image.fromarray(np.uint8(255.0*alpha * (np.max(segmentation, axis=2) > 0.0)))
  pil_image = Image.composite(pil_segmentation, pil_image, pil_mask)

  np.copyto(image, np.array(pil_image.convert('RGB')))


def visualize_boxes_and_labels_on_image_array(
    image,
    boxes,
    scores=None,
    segmentation=None,
    instance_boundaries=None,
    keypoints=None,
    track_ids=None,
    use_normalized_coordinates=False,
    max_boxes_to_draw=20,
    min_score_thresh=.5,
    agnostic_mode=False,
    line_thickness=4,
    groundtruth_box_visualization_color='yellow'):
  """Overlay labeled boxes on an image with formatted scores and label names.

  This function groups boxes that correspond to the same location
  and creates a display string for each detection and overlays these
  on the image. Note that this function modifies the image in place, and returns
  that same image.

  Args:
    image: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    instance_masks: a numpy array of shape [N, image_height, image_width] with
      values ranging between 0 and 1, can be None.
    use_normalized_coordinates: whether boxes is to be interpreted as
      normalized coordinates or not.
    max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
      all boxes.
    min_score_thresh: minimum score threshold for a box to be visualized
    agnostic_mode: boolean (default: False) controlling whether to evaluate in
      class-agnostic mode or not.  This mode will display scores but ignore
      classes.
    line_thickness: integer (default: 4) controlling line width of the boxes.
    groundtruth_box_visualization_color: box color for visualizing groundtruth
      boxes
    skip_scores: whether to skip score when drawing a single detection
    skip_labels: whether to skip label when drawing a single detection
    skip_track_ids: whether to skip track id when drawing a single detection

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
  """
  # Create a display string (and color) for every box location, group any boxes
  # that correspond to the same location.
  box_to_display_str_map = collections.defaultdict(list)
  box_to_color_map = collections.defaultdict(str)
  if not max_boxes_to_draw:
    max_boxes_to_draw = boxes.shape[0]
  for i in range(min(max_boxes_to_draw, boxes.shape[0])):
    if scores is None or scores[i] > min_score_thresh:
      box = tuple(boxes[i].tolist())
      if scores is None:
        box_to_color_map[box] = groundtruth_box_visualization_color
    #   else:
        # display_str = ''
        # if not skip_labels:
        #   if not agnostic_mode:
        #     if classes[i] in category_index.keys():
        #       class_name = category_index[classes[i]]['name']
        #     else:
        #       class_name = 'N/A'
        #     display_str = str(class_name)
        # if not skip_scores:
        #   if not display_str:
        #     display_str = '{}%'.format(int(100*scores[i]))
        #   else:
        #     display_str = '{}: {}%'.format(display_str, int(100*scores[i]))
        # if not skip_track_ids and track_ids is not None:
        #   if not display_str:
        #     display_str = 'ID {}'.format(track_ids[i])
        #   else:
        #     display_str = '{}: ID {}'.format(display_str, track_ids[i])
        # box_to_display_str_map[box].append(display_str)
        # if agnostic_mode:
        #   box_to_color_map[box] = 'DarkOrange'
        # elif track_ids is not None:
        #   prime_multipler = _get_multiplier_for_color_randomness()
        #   box_to_color_map[box] = STANDARD_COLORS[
        #       (prime_multipler * track_ids[i]) % len(STANDARD_COLORS)]
        # else:
        #   box_to_color_map[box] = STANDARD_COLORS[
        #       classes[i] % len(STANDARD_COLORS)]

  # Draw all boxes onto image.
  if segmentation is not None:
    draw_segmentation_on_image_array(
      image,
      segmentation,
      alpha=0.4
    )
  for box, color in box_to_color_map.items():
    center_x, center_y, angle, width = box
    # if instance_masks is not None:
    #   draw_mask_on_image_array(
    #       image,
    #       box_to_instance_masks_map[box],
    #       color=color
    #   )
    # if instance_boundaries is not None:
    #   draw_mask_on_image_array(
    #       image,
    #       box_to_instance_boundaries_map[box],
    #       color='red',
    #       alpha=1.0
    #   )
    draw_bounding_box_on_image_array(
        image,
        center_x,
        center_y,
        angle,
        width,
        color=color,
        thickness=line_thickness,
        # display_str_list=box_to_display_str_map[box],
        use_normalized_coordinates=use_normalized_coordinates)

  return image


class EvalMetricOpsVisualization(object):
  """Abstract base class responsible for visualizations during evaluation.

  Currently, summary images are not run during evaluation. One way to produce
  evaluation images in Tensorboard is to provide tf.summary.image strings as
  `value_ops` in tf.estimator.EstimatorSpec's `eval_metric_ops`. This class is
  responsible for accruing images (with overlaid detections and groundtruth)
  and returning a dictionary that can be passed to `eval_metric_ops`.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self,
               max_examples_to_draw=5,
               max_boxes_to_draw=20,
               min_score_thresh=0.2,
               use_normalized_coordinates=True,
               summary_name_prefix='evaluation_image'):
    """Creates an EvalMetricOpsVisualization.

    Args:
      max_examples_to_draw: The maximum number of example summaries to produce.
      max_boxes_to_draw: The maximum number of boxes to draw for detections.
      min_score_thresh: The minimum score threshold for showing detections.
      use_normalized_coordinates: Whether to assume boxes and kepoints are in
        normalized coordinates (as opposed to absolute coordiantes).
        Default is True.
      summary_name_prefix: A string prefix for each image summary.
    """

    self._max_examples_to_draw = max_examples_to_draw
    self._max_boxes_to_draw = max_boxes_to_draw
    self._min_score_thresh = min_score_thresh
    self._use_normalized_coordinates = use_normalized_coordinates
    self._summary_name_prefix = summary_name_prefix
    self._images = []

  def clear(self):
    self._images = []

  def add_images(self, images):
    """Store a list of images, each with shape [1, H, W, C]."""
    if len(self._images) >= self._max_examples_to_draw:
      return

    # Store images and clip list if necessary.
    self._images.extend(images)
    if len(self._images) > self._max_examples_to_draw:
      self._images[self._max_examples_to_draw:] = []

  def get_estimator_eval_metric_ops(self, eval_dict):
    """Returns metric ops for use in tf.estimator.EstimatorSpec.

    Args:
      eval_dict: A dictionary that holds an image, groundtruth, and detections
        for a batched example. Note that, we use only the first example for
        visualization. See eval_util.result_dict_for_batched_example() for a
        convenient method for constructing such a dictionary. The dictionary
        contains
        fields.InputDataFields.original_image: [batch_size, H, W, 3] image.
        fields.InputDataFields.original_image_spatial_shape: [batch_size, 2]
          tensor containing the size of the original image.
        fields.InputDataFields.true_image_shape: [batch_size, 3]
          tensor containing the spatial size of the upadded original image.
        fields.InputDataFields.groundtruth_boxes - [batch_size, num_boxes, 4]
          float32 tensor with groundtruth boxes in range [0.0, 1.0].
        fields.InputDataFields.groundtruth_classes - [batch_size, num_boxes]
          int64 tensor with 1-indexed groundtruth classes.
        fields.InputDataFields.groundtruth_instance_masks - (optional)
          [batch_size, num_boxes, H, W] int64 tensor with instance masks.
        fields.DetectionResultFields.detection_boxes - [batch_size,
          max_num_boxes, 4] float32 tensor with detection boxes in range [0.0,
          1.0].
        fields.DetectionResultFields.detection_classes - [batch_size,
          max_num_boxes] int64 tensor with 1-indexed detection classes.
        fields.DetectionResultFields.detection_scores - [batch_size,
          max_num_boxes] float32 tensor with detection scores.
        fields.DetectionResultFields.detection_masks - (optional) [batch_size,
          max_num_boxes, H, W] float32 tensor of binarized masks.
        fields.DetectionResultFields.detection_keypoints - (optional)
          [batch_size, max_num_boxes, num_keypoints, 2] float32 tensor with
          keypoints.

    Returns:
      A dictionary of image summary names to tuple of (value_op, update_op). The
      `update_op` is the same for all items in the dictionary, and is
      responsible for saving a single side-by-side image with detections and
      groundtruth. Each `value_op` holds the tf.summary.image string for a given
      image.
    """
    if self._max_examples_to_draw == 0:
      return {}
    images = self.images_from_evaluation_dict(eval_dict)

    def get_images():
      """Returns a list of images, padded to self._max_images_to_draw."""
      images = self._images
      while len(images) < self._max_examples_to_draw:
        images.append(np.array(0, dtype=np.uint8))
      self.clear()
      return images

    def image_summary_or_default_string(summary_name, image):
      """Returns image summaries for non-padded elements."""
      return tf.cond(
          tf.equal(tf.size(tf.shape(image)), 4),
          lambda: tf.summary.image(summary_name, image),
          lambda: tf.constant(''))

    if tf.executing_eagerly():
      update_op = self.add_images([[images[0]]])
      image_tensors = get_images()
    else:
      update_op = tf.py_func(self.add_images, [[images[0]]], [])
      image_tensors = tf.py_func(
          get_images, [], [tf.uint8] * self._max_examples_to_draw)
    eval_metric_ops = {}
    for i, image in enumerate(image_tensors):
      summary_name = self._summary_name_prefix + '/' + str(i)
      value_op = image_summary_or_default_string(summary_name, image)
      eval_metric_ops[summary_name] = (value_op, update_op)
    return eval_metric_ops

  @abc.abstractmethod
  def images_from_evaluation_dict(self, eval_dict):
    """Converts evaluation dictionary into a list of image tensors.

    To be overridden by implementations.

    Args:
      eval_dict: A dictionary with all the necessary information for producing
        visualizations.

    Returns:
      A list of [1, H, W, C] uint8 tensors.
    """
    raise NotImplementedError


class OAHODetectionVisualizer(EvalMetricOpsVisualization):
  """Class responsible for single-frame object detection visualizations."""

  def __init__(self,
               max_examples_to_draw=5,
               max_boxes_to_draw=20,
               min_score_thresh=0.2,
               use_normalized_coordinates=True,
               summary_name_prefix='Detections_Left_Groundtruth_Right'):
    super(OAHODetectionVisualizer, self).__init__(
        max_examples_to_draw=max_examples_to_draw,
        max_boxes_to_draw=max_boxes_to_draw,
        min_score_thresh=min_score_thresh,
        use_normalized_coordinates=use_normalized_coordinates,
        summary_name_prefix=summary_name_prefix)

  def images_from_evaluation_dict(self, eval_dict):
    return draw_side_by_side_evaluation_image(
        eval_dict, self._max_boxes_to_draw,
        self._min_score_thresh, self._use_normalized_coordinates)
