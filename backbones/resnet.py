from typing import Optional

from official.vision.modeling.backbones.resnet import ResNet
import tensorflow as tf


def get_resnet(
    num_classes: int,
    model_id: int,
    input_shape,
    depth_multiplier: float = 1.0,
    stem_type: str = "v0",
    resnetd_shortcut: bool = False,
    replace_stem_max_pool: bool = False,
    se_ratio: Optional[float] = None,
    init_stochastic_depth_rate: float = 0.0,
    scale_stem: bool = True,
    activation: str = "relu",
    use_sync_bn: bool = False,
    norm_momentum: float = 0.99,
    norm_epsilon: float = 0.001,
    kernel_initializer: str = "VarianceScaling",
    l2_reg: float = None,
    l2_bias_reg: float = None,
    bn_trainable: bool = True,
):
    input_spec = tf.keras.layers.InputSpec(shape=(None,) + input_shape)
    if l2_reg is not None:
        l2_reg = tf.keras.regularizers.L2(l2=l2_reg)
    if l2_bias_reg is not None:
        l2_bias_reg = tf.keras.regularizers.L2(l2=l2_bias_reg)
    backbone = ResNet(
        model_id,
        input_specs=input_spec,
        depth_multiplier=depth_multiplier,
        stem_type=stem_type,
        resnetd_shortcut=resnetd_shortcut,
        replace_stem_max_pool=replace_stem_max_pool,
        se_ratio=se_ratio,
        init_stochastic_depth_rate=init_stochastic_depth_rate,
        scale_stem=scale_stem,
        activation=activation,
        use_sync_bn=use_sync_bn,
        norm_momentum=norm_momentum,
        norm_epsilon=norm_epsilon,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=l2_reg,
        bias_regularizer=l2_bias_reg,
        bn_trainable=bn_trainable,
    )
    x = tf.keras.layers.Input(shape=input_shape)
    x_init = x
    x = backbone(x)
    x = x["5"]
    if num_classes is not None:
        x = tf.keras.layers.GlobalAvgPool2D()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(
            units=num_classes,
            use_bias=True,
            kernel_regularizer=tf.keras.regularizers.L2(l2=0.0001),
        )(x)
    return tf.keras.Model(inputs=x_init, outputs=x)
