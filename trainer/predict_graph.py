import os.path
from typing import Any, Dict, List, Text

import numpy as np
from PIL import Image
import tensorflow as tf
import PIL

# OSS: removed unused atomic file imports.
from deeplab2 import common
from deeplab2.data import ade20k_constants
from deeplab2.data import coco_constants
from deeplab2.data import dataset
from deeplab2.trainer import vis_utils

# The format of the labels.
_IMAGE_FORMAT = '%06d_image'
_CENTER_LABEL_FORMAT = '%06d_center_label'
_OFFSET_LABEL_FORMAT = '%06d_offset_label'
_PANOPTIC_LABEL_FORMAT = '%06d_panoptic_label'
_SEMANTIC_LABEL_FORMAT = '%06d_semantic_label'

# The format of the predictions.
_INSTANCE_PREDICTION_FORMAT = '%06d_instance_prediction'
_CENTER_HEATMAP_PREDICTION_FORMAT = '%06d_center_prediction'
_OFFSET_PREDICTION_RGB_FORMAT = '%06d_offset_prediction_rgb'
_PANOPTIC_PREDICTION_FORMAT = '%06d_panoptic_prediction'
_SEMANTIC_PREDICTION_FORMAT = '%06d_semantic_prediction'

# The format of others.
_ANALYSIS_FORMAT = '%06d_semantic_error'

def get_annotation(label,
                    add_colormap=True,
                    normalize_to_unit_values=False,
                    scale_factor=None,
                    colormap_name='cityscapes',
                    output_dtype=np.uint8):
  """Saves the given label to image on disk.

  Args:
    label: The numpy array to be saved. The data will be converted
      to uint8 and saved as png image.
    save_dir: String, the directory to which the results will be saved.
    filename: String, the image filename.
    add_colormap: Boolean, add color map to the label or not.
    normalize_to_unit_values: Boolean, normalize the input values to [0, 1].
    scale_factor: Float or None, the factor to scale the input values.
    colormap_name: A string specifying the dataset to choose the corresponding
      color map. Currently supported: 'cityscapes', 'motchallenge'. (Default:
      'cityscapes').
    output_dtype: The numpy dtype of output before converting to PIL image.
  """
  # Add colormap for visualizing the prediction.
  if add_colormap:
    colored_label = vis_utils.label_to_color_image(label, colormap_name)
  else:
    colored_label = label
    if normalize_to_unit_values:
      min_value = np.amin(colored_label)
      max_value = np.amax(colored_label)
      range_value = max_value - min_value
      if range_value != 0:
        colored_label = (colored_label - min_value) / range_value

    if scale_factor:
      colored_label = scale_factor * colored_label

  return colored_label.astype(dtype=output_dtype)


def get_parsing_result(parsing_result,
                        label_divisor,
                        thing_list,
                        id_to_colormap=None,
                        colormap_name='cityscapes'):
  """Saves the parsing results.

  The parsing result encodes both semantic segmentation and instance
  segmentation results. In order to visualize the parsing result with only
  one png file, we adopt the following procedures, similar to the
  `visualization.py` provided in the COCO panoptic segmentation evaluation
  codes.

  1. Pixels predicted as `stuff` will take the same semantic color defined
    in the colormap.
  2. Pixels of a predicted `thing` instance will take similar semantic color
    defined in the colormap. For example, `car` class takes blue color in
    the colormap. Predicted car instance 1 will then be colored with the
    blue color perturbed with a small amount of RGB noise.

  Args:
    parsing_result: The numpy array to be saved. The data will be converted
      to uint8 and saved as png image.
    label_divisor: Integer, encoding the semantic segmentation and instance
      segmentation results as value = semantic_label * label_divisor +
      instance_label.
    thing_list: A list containing the semantic indices of the thing classes.
    save_dir: String, the directory to which the results will be saved.
    filename: String, the image filename.
    id_to_colormap: An optional mapping from track ID to color.
    colormap_name: A string specifying the dataset to choose the corresponding
      color map. Currently supported: 'cityscapes', 'motchallenge'. (Default:
      'cityscapes').

  Raises:
    ValueError: If parsing_result is not of rank 2 or its value in semantic
      segmentation result is larger than color map maximum entry.
    ValueError: If provided colormap_name is not supported.

  Returns:
    If id_to_colormap is passed, the updated id_to_colormap will be returned.
  """
  if parsing_result.ndim != 2:
    raise ValueError('Expect 2-D parsing result. Got {}'.format(
        parsing_result.shape))
  semantic_result = parsing_result // label_divisor
  instance_result = parsing_result % label_divisor
  colormap_max_value = 256
  if np.max(semantic_result) >= colormap_max_value:
    raise ValueError('Predicted semantic value too large: {} >= {}.'.format(
        np.max(semantic_result), colormap_max_value))
  height, width = parsing_result.shape
  colored_output = np.zeros((height, width, 3), dtype=np.uint8)
  if colormap_name == 'cityscapes':
    colormap = vis_utils.create_cityscapes_label_colormap()
  elif colormap_name == 'motchallenge':
    colormap = vis_utils.create_motchallenge_label_colormap()
  elif colormap_name == 'coco':
    colormap = vis_utils.create_coco_label_colormap()
  elif colormap_name == 'ade20k':
    colormap = vis_utils.create_ade20k_label_colormap()
  elif colormap_name == 'waymo':
    colormap = vis_utils.create_waymo_label_colormap()
  elif colormap_name == 'graph':
    colormap =vis_utils.create_graph_label_colormap()
  else:
    raise ValueError('Could not find a colormap for dataset %s.' %
                     colormap_name)
  # Keep track of used colors.
  used_colors = set()
  if id_to_colormap is not None:
    used_colors = set([tuple(val) for val in id_to_colormap.values()])
    np_state = None
  else:
    # Use random seed 0 in order to reproduce the same visualization.
    np_state = np.random.RandomState(0)

  unique_semantic_values = np.unique(semantic_result)
  for semantic_value in unique_semantic_values:
    semantic_mask = semantic_result == semantic_value
    if semantic_value in thing_list:
      # For `thing` class, we will add a small amount of random noise to its
      # correspondingly predefined semantic segmentation colormap.
      unique_instance_values = np.unique(instance_result[semantic_mask])
      for instance_value in unique_instance_values:
        instance_mask = np.logical_and(semantic_mask,
                                       instance_result == instance_value)
        if id_to_colormap is not None:
          if instance_value in id_to_colormap:
            colored_output[instance_mask] = id_to_colormap[instance_value]
            continue
        random_color = vis_utils.perturb_color(
            colormap[semantic_value],
            vis_utils._COLOR_PERTURBATION,
            used_colors,
            random_state=np_state)
        colored_output[instance_mask] = random_color
        if id_to_colormap is not None:
          id_to_colormap[instance_value] = random_color
    else:
      # For `stuff` class, we use the defined semantic color.
      colored_output[semantic_mask] = colormap[semantic_value]
      used_colors.add(tuple(colormap[semantic_value]))

  return colored_output.astype(dtype=np.uint8)


    
def get_predictions(predictions: Dict[str, Any], inputs: Dict[str, Any], dataset_info: dataset.DatasetDescriptor):
  """Returns numpy image version of the predictions and labels"""
  predictions = {key: predictions[key][0] for key in predictions}
  predictions = vis_utils.squeeze_batch_dim_and_convert_to_numpy(predictions)
  inputs = {key: inputs[key][0] for key in inputs}
  del inputs[common.IMAGE_NAME]
  inputs = vis_utils.squeeze_batch_dim_and_convert_to_numpy(inputs)

  thing_list = dataset_info.class_has_instances_list
  label_divisor = dataset_info.panoptic_label_divisor
  colormap_name = dataset_info.colormap

  preds_vis = {}

  # 1. Save image.
  image = inputs[common.IMAGE]
  preds_vis[common.IMAGE] = get_annotation(
      image,
      add_colormap=False)

  # 2. Save semantic predictions and semantic labels.
  preds_vis[common.PRED_SEMANTIC_KEY] = get_annotation(
      predictions[common.PRED_SEMANTIC_KEY],
      add_colormap=True,
      colormap_name=colormap_name)
  # vis_utils.save_annotation(
  #     inputs[common.GT_SEMANTIC_RAW],
  #     add_colormap=True,
  #     colormap_name=colormap_name)

  if common.PRED_CENTER_HEATMAP_KEY in predictions:
    # 3. Save center heatmap.
    heatmap_pred = predictions[common.PRED_CENTER_HEATMAP_KEY]
    heat_map_gt = inputs[common.GT_INSTANCE_CENTER_KEY]
    preds_vis[common.PRED_CENTER_HEATMAP_KEY] = get_annotation(
        vis_utils.overlay_heatmap_on_image(
            heatmap_pred,
            image),
        add_colormap=False)
    preds_vis[common.GT_INSTANCE_CENTER_KEY] = get_annotation(
        vis_utils.overlay_heatmap_on_image(
            heat_map_gt,
            image),
        add_colormap=False)

  if common.PRED_OFFSET_MAP_KEY in predictions:
    # 4. Save center offsets.
    center_offset_prediction = predictions[common.PRED_OFFSET_MAP_KEY]
    center_offset_prediction_rgb = vis_utils.flow_to_color(
        center_offset_prediction)
    semantic_prediction = predictions[common.PRED_SEMANTIC_KEY]
    pred_fg_mask = vis_utils._get_fg_mask(semantic_prediction, thing_list)
    center_offset_prediction_rgb = (
        center_offset_prediction_rgb * pred_fg_mask)
    preds_vis[common.PRED_OFFSET_MAP_KEY] = get_annotation(
        center_offset_prediction_rgb,
        add_colormap=False)

    center_offset_label = inputs[common.GT_INSTANCE_REGRESSION_KEY]
    center_offset_label_rgb = vis_utils.flow_to_color(center_offset_label)
    gt_fg_mask = vis_utils._get_fg_mask(inputs[common.GT_SEMANTIC_RAW], thing_list)
    center_offset_label_rgb = center_offset_label_rgb * gt_fg_mask

    preds_vis[common.GT_INSTANCE_REGRESSION_KEY] = get_annotation(
        center_offset_label_rgb,
        add_colormap=False)

  if common.PRED_INSTANCE_KEY in predictions:
    # 5. Save instance map.
    preds_vis[common.PRED_INSTANCE_KEY] = get_annotation(
        vis_utils.create_rgb_from_instance_map(
            predictions[common.PRED_INSTANCE_KEY]),
        add_colormap=False)

  if common.PRED_PANOPTIC_KEY in predictions:
    # 6. Save panoptic segmentation.
    preds_vis[common.PRED_PANOPTIC_KEY] = get_parsing_result(
        predictions[common.PRED_PANOPTIC_KEY],
        label_divisor=label_divisor,
        thing_list=thing_list,
        colormap_name=colormap_name)
    preds_vis[common.GT_PANOPTIC_RAW] = get_parsing_result(
        parsing_result=inputs[common.GT_PANOPTIC_RAW],
        label_divisor=label_divisor,
        thing_list=thing_list,
        colormap_name=colormap_name)

  # 7. Save error of semantic prediction.
  label = inputs[common.GT_SEMANTIC_RAW].astype(np.uint8)
  error_prediction = (
      (predictions[common.PRED_SEMANTIC_KEY] != label) &
      (label != dataset_info.ignore_label)).astype(np.uint8) * 255
  preds_vis['semantic_error'] = get_annotation(
      error_prediction,
      add_colormap=False)
  return preds_vis