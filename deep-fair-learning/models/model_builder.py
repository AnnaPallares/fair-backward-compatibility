import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, MaxPooling2D, Flatten, Dense
import keras_hub

def build_model_resnet50(input_shape, fine_tune_at=None):
    """
    ResNet50V2 for FBC. 
    fine_tune_at: Int. If provided, unfreezes layers from this index onwards.
    """
    base = ResNet50V2(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg' # Directly gives a 2048-d vector
    )
    
    # Standard Transfer Learning: Freeze the base initially
    base.trainable = False
    
    if fine_tune_at is not None:
        base.trainable = True
        for layer in base.layers[:fine_tune_at]:
            # Keep early layers frozen (generic features)
            # UNLESS they are BatchNormalization (best practice to keep BN frozen during fine-tuning)
            layer.trainable = False

    x = base.output
    # The 'pred' name is used by your trainer to identify the output head
    outputs = layers.Dense(1, activation='sigmoid', name='pred')(x)

    return Model(inputs=base.input, outputs=outputs, name="ResNet50_FBC")

def build_model_resnet18(input_shape):
    """
    ResNet18 via KerasHub. 
    Note: ResNet18 is often better for smaller datasets (like Fitzpatrick17k).
    """
    base = keras_hub.models.Backbone.from_preset(
        "resnet_18_imagenet",
        input_shape=input_shape,
        include_top=False,
    )
    base.trainable = False

    # KerasHub backbones return a dictionary of feature maps; we take the last one
    x = base.output
    if isinstance(x, dict):
        x = list(x.values())[-1]
        
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="pred")(x)
    
    return Model(inputs=base.input, outputs=outputs, name="ResNet18_FBC")

def build_model_MobileNetV2(input_shape):
    """
    MobileNetV2: Optimized for speed and edge deployment.
    """
    base = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )
    base.trainable = False

    x = base.output
    outputs = layers.Dense(1, activation='sigmoid', name='pred')(x)
    return Model(inputs=base.input, outputs=outputs, name="MobileNetV2_FBC")

def build_model_CNN(input_shape):
    """
    A lightweight custom CNN for baseline comparisons or small-scale testing.
    """
    inputs = Input(shape=input_shape)

    x = Conv2D(32, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D()(x)

    x = Conv2D(64, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D()(x)

    x = Flatten()(x)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid", name='pred')(x)
    
    return Model(inputs=inputs, outputs=outputs, name="CustomCNN_FBC")