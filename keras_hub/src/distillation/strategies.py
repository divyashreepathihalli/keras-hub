import keras


class BaseDistillationStrategy:
    """Base class for all distillation strategies.

    A distillation strategy defines a specific way a student model learns from a
    teacher model. This could involve matching logits, intermediate features,
    attention maps, or other custom comparisons.

    Each strategy is responsible for computing a loss component based on the
    teacher and student outputs (or internal states). These losses are then
    combined by the `Distiller` model.

    Args:
        loss_fn: A callable (e.g., a `keras.losses.Loss` instance) used to
            compute the difference between teacher and student representations.
        weight: A float specifying the contribution of this strategy's loss
            to the total distillation loss.
    """

    def __init__(self, loss_fn, weight):
        self.loss_fn = loss_fn
        self.weight = weight

    def compute_loss(self, teacher_outputs, student_outputs):
        raise NotImplementedError("Subclasses must implement this method.")


class LogitsDistillation(BaseDistillationStrategy):
    """Distillation strategy for matching softened logits.

    This strategy encourages the student model's output logits to match the
    teacher model's output logits after both are softened by a temperature T.
    The loss is typically a Kullback-Leibler divergence between the softened
    probability distributions. The final loss is scaled by T^2 as is common
    practice.

    Args:
        temperature: A float, the temperature T used for softening the logits.
            Higher temperatures result in softer probability distributions.
        loss_fn: A callable (e.g., `keras.losses.KLDivergence`) to compute the
            loss between the teacher's and student's softened probability
            distributions.
        weight: A float, weight for this strategy's loss.
    """

    def __init__(self, temperature, loss_fn, weight):
        super().__init__(loss_fn, weight)
        self.temperature = temperature

    def compute_loss(self, teacher_logits, student_logits):
        teacher_probs = keras.ops.softmax(
            teacher_logits / self.temperature, axis=-1
        )
        student_probs = keras.ops.softmax(
            student_logits / self.temperature, axis=-1
        )
        # Apply T^2 scaling to distillation loss
        return self.loss_fn(teacher_probs, student_probs) * (
            self.temperature**2
        )


class AttentionDistillation(BaseDistillationStrategy):
    """Distillation strategy for matching attention maps between models.

    This strategy extracts attention maps (or any intermediate feature map that
    can be interpreted as attention) from specified layers in the teacher and
    student models. It then computes a loss based on the difference between
    these maps, encouraging the student to learn similar attention patterns as
    the teacher.

    Submodels are created internally to extract outputs from the specified
    layers.

    Args:
        teacher_layer: Name or index of the layer in the teacher model from
            which to extract the attention map.
        student_layer: Name or index of the layer in the student model from
            which to extract the attention map.
        loss_fn: A callable (e.g., `keras.losses.MeanSquaredError`) to compute
            the loss between the teacher's and student's attention maps.
        weight: A float, weight for this strategy's loss.
    """

    def __init__(self, teacher_layer, student_layer, loss_fn, weight):
        super().__init__(loss_fn, weight)
        self.teacher_layer_name = teacher_layer
        self.student_layer_name = student_layer
        self.teacher_submodel = None
        self.student_submodel = None

    def _build_submodels(self, teacher_model, student_model):
        try:
            teacher_layer_output = teacher_model.get_layer(
                self.teacher_layer_name
            ).output
        except ValueError:
            raise ValueError(
                f"Teacher layer '{self.teacher_layer_name}' not found in teacher model. Available layers: {[layer.name for layer in teacher_model.layers]}"  # noqa: E501
            )
        self.teacher_submodel = keras.Model(
            inputs=teacher_model.input, outputs=teacher_layer_output
        )

        try:
            student_layer_output = student_model.get_layer(
                self.student_layer_name
            ).output
        except ValueError:
            raise ValueError(
                f"Student layer '{self.student_layer_name}' not found in student model. Available layers: {[layer.name for layer in student_model.layers]}"  # noqa: E501
            )
        self.student_submodel = keras.Model(
            inputs=student_model.input, outputs=student_layer_output
        )

    def compute_loss(self, teacher_model, student_model, inputs):
        if self.teacher_submodel is None or self.student_submodel is None:
            self._build_submodels(teacher_model, student_model)

        teacher_attention = self.teacher_submodel(inputs, training=False)
        student_attention = self.student_submodel(inputs, training=True)
        return self.loss_fn(teacher_attention, student_attention)


class FeatureDistillation(BaseDistillationStrategy):
    def __init__(
        self,
        teacher_layer,
        student_layer,
        loss_fn,
        weight,
        projection=False,
        student_output_dim_for_projection=None,
    ):
        super().__init__(loss_fn, weight)
        self.teacher_layer_name = teacher_layer
        self.student_layer_name = student_layer
        self.projection = projection
        self.projection_layer = None
        self.teacher_submodel = None
        self.student_submodel = None
        self.student_output_dim_for_projection = (
            student_output_dim_for_projection
        )

    def _build_submodels(self, teacher_model, student_model):
        try:
            teacher_layer_output = teacher_model.get_layer(
                self.teacher_layer_name
            ).output
        except ValueError:
            raise ValueError(
                f"Teacher layer '{self.teacher_layer_name}' not found in teacher model. Available layers: {[layer.name for layer in teacher_model.layers]}"  # noqa: E501
            )
        self.teacher_submodel = keras.Model(
            inputs=teacher_model.input, outputs=teacher_layer_output
        )

        try:
            student_layer_output = student_model.get_layer(
                self.student_layer_name
            ).output
        except ValueError:
            raise ValueError(
                f"Student layer '{self.student_layer_name}' not found in student model. Available layers: {[layer.name for layer in student_model.layers]}"  # noqa: E501
            )
        self.student_submodel = keras.Model(
            inputs=student_model.input, outputs=student_layer_output
        )

        if self.projection:
            student_output_dim = self.student_output_dim_for_projection
            if student_output_dim is None:
                # Try to infer from the student layer's output shape.
                student_output_dim = student_layer_output.shape[-1]

            if student_output_dim is None:
                raise ValueError(
                    "Cannot automatically infer `student_output_dim` for "
                    "projection in FeatureDistillation. "
                    "The student layer's output has an unknown feature "
                    "dimension. "
                    "Please specify `student_output_dim_for_projection` "
                    "explicitly."
                )

            self.projection_layer = keras.layers.Dense(
                student_output_dim,
                name=f"projection_{self.teacher_layer_name}_to_{self.student_layer_name}",  # noqa: E501
            )

    def compute_loss(self, teacher_model, student_model, inputs):
        if self.teacher_submodel is None or self.student_submodel is None:
            self._build_submodels(teacher_model, student_model)

        teacher_features = self.teacher_submodel(inputs, training=False)
        student_features = self.student_submodel(inputs, training=True)

        if self.projection and self.projection_layer:
            teacher_features = self.projection_layer(teacher_features)

        return self.loss_fn(teacher_features, student_features)
