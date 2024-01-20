# Required packages

import os

import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

# Variables and Filenames


pt_model_name = "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8"
pt_model_path = os.path.join('Tensorflow', 'pre_trained_model')
config_path = os.path.join('pipeline', 'pipeline.config')
exp_model_path = os.path.join('exported_model', 'costume_trained_model')



# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(config_path)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(exp_model_path, 'ckpt-3')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections