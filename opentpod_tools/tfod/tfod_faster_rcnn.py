from .base import TFODDetector
from mako import template

GENERIC_FASTER_RCNN_RESNET_TEMPLATE = """
# Faster R-CNN with Resnet-101 (v1) configured for the Oxford-IIIT Pet Dataset.
# Users should configure the fine_tune_checkpoint field in the train config as
# well as the label_map_path and input_path fields in the train_input_reader and
# eval_input_reader. Search for "PATH_TO_BE_CONFIGURED" to find the fields that
# should be configured.

model {
faster_rcnn {
    num_classes: ${num_classes}
    image_resizer {
    keep_aspect_ratio_resizer {
        min_dimension: 600
        max_dimension: 1024
    }
    }
    feature_extractor {
    type: "${feature_extractor_type}"
    first_stage_features_stride: 16
    }
    first_stage_anchor_generator {
    grid_anchor_generator {
        scales: [0.25, 0.5, 1.0, 2.0]
        aspect_ratios: [0.5, 1.0, 2.0]
        height_stride: 16
        width_stride: 16
    }
    }
    first_stage_box_predictor_conv_hyperparams {
    op: CONV
    regularizer {
        l2_regularizer {
        weight: 0.0
        }
    }
    initializer {
        truncated_normal_initializer {
        stddev: 0.01
        }
    }
    }
    first_stage_nms_score_threshold: 0.0
    first_stage_nms_iou_threshold: 0.7
    first_stage_max_proposals: 300
    first_stage_localization_loss_weight: 2.0
    first_stage_objectness_loss_weight: 1.0
    initial_crop_size: 14
    maxpool_kernel_size: 2
    maxpool_stride: 2
    second_stage_box_predictor {
    mask_rcnn_box_predictor {
        use_dropout: false
        dropout_keep_probability: 1.0
        fc_hyperparams {
        op: FC
        regularizer {
            l2_regularizer {
            weight: 0.0
            }
        }
        initializer {
            variance_scaling_initializer {
            factor: 1.0
            uniform: true
            mode: FAN_AVG
            }
        }
        }
    }
    }
    second_stage_post_processing {
    batch_non_max_suppression {
        score_threshold: 0.0
        iou_threshold: 0.6
        max_detections_per_class: 100
        max_total_detections: 300
    }
    score_converter: SOFTMAX
    }
    second_stage_localization_loss_weight: 2.0
    second_stage_classification_loss_weight: 1.0
}
}

train_config: {
batch_size: ${batch_size}
optimizer {
    momentum_optimizer: {
    learning_rate: {
        manual_step_learning_rate {
        initial_learning_rate: 0.0003
        schedule {
            step: 900000
            learning_rate: .00003
        }
        schedule {
            step: 1200000
            learning_rate: .000003
        }
        }
    }
    momentum_optimizer_value: 0.9
    }
    use_moving_average: false
}
gradient_clipping_by_norm: 10.0
fine_tune_checkpoint: "${fine_tune_checkpoint}"
from_detection_checkpoint: true
load_all_detection_checkpoint_vars: true
# Note: The below line limits the training process to 200K steps, which we
# empirically found to be sufficient enough to train the pets dataset. This
# effectively bypasses the learning rate schedule (the learning rate will
# never decay). Remove the below line to train indefinitely.
num_steps: ${num_steps}
data_augmentation_options {
    random_horizontal_flip {
    }
}
}

train_input_reader: {
tf_record_input_reader {
    input_path: "${train_input_path}"
}
label_map_path: "${label_map_path}"
}

eval_config: {
metrics_set: "coco_detection_metrics"
}

eval_input_reader: {
tf_record_input_reader {
    input_path: "${eval_input_path}"
}
label_map_path: "${label_map_path}"
shuffle: false
num_readers: 1
}
"""


class TFODFasterRCNNResNetGeneric(TFODDetector):
    TRAINING_PARAMETERS = {'batch_size': 2, 'num_steps': 20000}

    def __init__(self, config):
        super().__init__(config)
        self._config['feature_extractor_type'] = self.feature_extractor_type

    @property
    def feature_extractor_type(self):
        raise NotImplementedError()

    @property
    def pipeline_config_template(self):
        return GENERIC_FASTER_RCNN_RESNET_TEMPLATE


class TFODFasterRCNNResNet101(TFODFasterRCNNResNetGeneric):
    def __init__(self, config):
        super().__init__(config)

    @property
    def pretrained_model_url(self):
        return 'http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz'

    @property
    def feature_extractor_type(self):
        return 'faster_rcnn_resnet101'


class TFODFasterRCNNResNet50(TFODFasterRCNNResNetGeneric):
    def __init__(self, config):
        super().__init__(config)

    @property
    def pretrained_model_url(self):
        return 'http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz'

    @property
    def feature_extractor_type(self):
        return 'faster_rcnn_resnet50'
