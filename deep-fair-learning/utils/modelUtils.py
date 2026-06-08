import keras_hub
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet50_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

# If testing other model architectures, add its preprocessing here

from models.model_builder import (
    build_model_resnet50,
    build_model_resnet18,
    build_model_MobileNetV2,
    build_model_vit_small
)

def get_builder_and_preproc(arch):
    if arch=='resnet50':
        return build_model_resnet50, resnet50_preprocess
    elif arch == "resnet18":
        resnet18_preprocess = keras_hub.models.ImageClassifierPreprocessor.from_preset("resnet_18_imagenet")
        return build_model_resnet18, resnet18_preprocess
    elif arch == "vit-small":
        vit_small_preprocess = keras_hub.models.ImageClassifierPreprocessor.from_preset("deit_small_distilled_patch16_224_imagenet")
        return build_model_vit_small, vit_small_preprocess
    else:
        return build_model_MobileNetV2, mobilenet_preprocess
