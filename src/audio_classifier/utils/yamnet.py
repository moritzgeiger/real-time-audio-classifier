"""load and serve pretrained model"""
from typing import Callable, Tuple, Union, Optional, List
from tensorflow.keras import Model, layers

from audio_classifier.data import params

LayerFunction = Callable[..., Union[layers.Layer, Model]]
class YAMNet():
    """Define the core YAMNet model in Keras."""
    def __init__(
        self,
        include_top: bool = True,
        weights: Optional[str] = None,
        input_tensor: Optional[layers.Input] = None,
        input_shape: Optional[Tuple[int, int]] = None,
        pooling: str = "avg",
        classes: int = 521,
        classifier_activation: str = "sigmoid",
        ):

        self.include_top = include_top
        self.weights = weights
        self.input_tensor = input_tensor
        self.input_shape = input_shape
        self.pooling = pooling
        self.classes = classes
        self.classifier_activation = classifier_activation

    def get_model(self,) -> Model:
        """Define the core YAMNet model in Keras."""

        if self.input_tensor is None:
            self.input_shape = (
                self.input_shape
                if self.input_shape is not None
                else (params.PATCH_FRAMES, params.PATCH_BANDS)
            )
            input_tensor = layers.Input(shape=self.input_shape)

        net = layers.Reshape(self.input_shape + (1,))(input_tensor)

        for i, (layer_fun, kernel, stride, filters) in enumerate(self.yamnet_layer_defs()):
            net = layer_fun(f"layer{i + 1}", kernel, stride, filters)(net)

        if self.include_top:
            if self.weights is not None and self.classes != params.NUM_CLASSES:
                model_temp = Model(inputs=input_tensor, outputs=net)

                if self.weights is not None:
                    model_temp.load_weights(self.weights)

                net = model_temp.output

            net = layers.GlobalAveragePooling2D()(net)
            logits = layers.Dense(units=self.classes, use_bias=True)(net)
            predictions = layers.Activation(
                name=params.EXAMPLE_PREDICTIONS_LAYER_NAME, activation=self.classifier_activation
            )(logits)

        else:
            if self.weights is not None:
                model_temp = Model(inputs=input_tensor, outputs=net)

                if self.weights is not None:
                    model_temp.load_weights(self.weights)

                net = model_temp.output

            if self.pooling == "avg":
                predictions = layers.GlobalAveragePooling2D()(net)
            elif self.pooling == "max":
                predictions = layers.GlobalMaxPooling2D()(net)
            else:
                predictions = net

        model = Model(inputs=input_tensor, outputs=predictions)

        if self.weights is not None and self.classes == params.NUM_CLASSES:
            model.load_weights(self.weights)

        return model

    def yamnet_layer_defs(self,) -> List:
        """(layer_function, kernel, stride, num_filters)"""
        return  [
            (self._conv, [3, 3], 2, 32),
            (self._separable_conv, [3, 3], 1, 64),
            (self._separable_conv, [3, 3], 2, 128),
            (self._separable_conv, [3, 3], 1, 128),
            (self._separable_conv, [3, 3], 2, 256),
            (self._separable_conv, [3, 3], 1, 256),
            (self._separable_conv, [3, 3], 2, 512),
            (self._separable_conv, [3, 3], 1, 512),
            (self._separable_conv, [3, 3], 1, 512),
            (self._separable_conv, [3, 3], 1, 512),
            (self._separable_conv, [3, 3], 1, 512),
            (self._separable_conv, [3, 3], 1, 512),
            (self._separable_conv, [3, 3], 2, 1024),
            (self._separable_conv, [3, 3], 1, 1024),
        ]

    def _batch_norm(self, name: str) -> Callable[[layers.Layer], layers.BatchNormalization]:
        """Create a batch normalization layer."""
        def _bn_layer(layer_input: layers.Layer) -> layers.BatchNormalization:
            return layers.BatchNormalization(
                name=name,
                center=params.BATCHNORM_CENTER,
                scale=params.BATCHNORM_SCALE,
                epsilon=params.BATCHNORM_EPSILON,
            )(layer_input)
        return _bn_layer

    def _conv(
            self, 
            name: str, 
            kernel: List[int], 
            stride: int, 
            filters: int
            ) -> Callable[[layers.Layer], layers.Layer]:
        """Create a convolutional layer."""
        def _conv_layer(layer_input: layers.Layer) -> layers.Layer:
            output = layers.Conv2D(
                name=f"{name}/conv",
                filters=filters,
                kernel_size=kernel,
                strides=stride,
                padding=params.CONV_PADDING,
                use_bias=False,
                activation=None,
            )(layer_input)
            output = self._batch_norm(name=f"{name}/conv/bn")(output)
            output = layers.ReLU(name=f"{name}/relu")(output)
            return output
        return _conv_layer

    def _separable_conv(
            self, 
            name: str, 
            kernel: List[int], 
            stride: int, 
            filters: int
            ) -> Callable[[layers.Layer], layers.Layer]:
        """Create a separable convolutional layer."""
        def _separable_conv_layer(layer_input: layers.Layer) -> layers.Layer:
            output = layers.DepthwiseConv2D(
                name=f"{name}/depthwise_conv",
                kernel_size=kernel,
                strides=stride,
                depth_multiplier=1,
                padding=params.CONV_PADDING,
                use_bias=False,
                activation=None,
            )(layer_input)
            output = self._batch_norm(name=f"{name}/depthwise_conv/bn")(output)
            output = layers.ReLU(name=f"{name}/depthwise_conv/relu")(output)
            output = layers.Conv2D(
                name=f"{name}/pointwise_conv",
                filters=filters,
                kernel_size=(1, 1),
                strides=1,
                padding=params.CONV_PADDING,
                use_bias=False,
                activation=None,
            )(output)
            output = self._batch_norm(name=f"{name}/pointwise_conv/bn")(output)
            output = layers.ReLU(name=f"{name}/pointwise_conv/relu")(output)
            return output
        return _separable_conv_layer
