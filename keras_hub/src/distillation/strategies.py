"""Distillation strategies for knowledge distillation."""

import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export


@keras_hub_export("keras_hub.distillation.BaseDistillationStrategy")
class BaseDistillationStrategy:
    """Base class for distillation strategies.

    Distillation strategies define how to compute the distillation loss
    between teacher and student outputs.
    """

    def compute_loss(self, teacher_outputs, student_outputs):
        """Compute distillation loss between teacher and student outputs.

        Args:
            teacher_outputs: Outputs from the teacher model.
            student_outputs: Outputs from the student model.

        Returns:
            Distillation loss tensor.
        """
        raise NotImplementedError("Subclasses must implement compute_loss")


@keras_hub_export("keras_hub.distillation.LogitsDistillation")
class LogitsDistillation(BaseDistillationStrategy):
    """Logits distillation with customizable loss functions using Keras
    built-ins.

    This strategy supports multiple loss functions for logits distillation,
    using Keras's built-in loss functions from the losses API.

    Args:
        temperature: Temperature for softening logits. Higher values
            make the distribution softer. Defaults to 2.0.
        loss_type: Type of loss function to use. Options:
            - "kl_divergence": KL divergence using keras.losses.KLDivergence
            - "mse": Mean squared error using keras.losses.MeanSquaredError
            - "cross_entropy": Cross entropy using
              keras.losses.CategoricalCrossentropy
            - "custom": Use custom_loss_fn
        custom_loss_fn: Custom loss function. Used when loss_type="custom".
            Should take (teacher_outputs, student_outputs) as arguments.
    """

    def __init__(
        self, temperature=2.0, loss_type="kl_divergence", custom_loss_fn=None
    ):
        self.temperature = temperature
        self.loss_type = loss_type
        self.custom_loss_fn = custom_loss_fn

        # Validate loss_type
        valid_loss_types = ["kl_divergence", "mse", "cross_entropy", "custom"]
        if loss_type not in valid_loss_types:
            raise ValueError(f"loss_type must be one of {valid_loss_types}")

        if loss_type == "custom" and custom_loss_fn is None:
            raise ValueError(
                "custom_loss_fn must be provided when loss_type='custom'"
            )

        # Initialize Keras loss functions
        if loss_type == "kl_divergence":
            self.loss_fn = keras.losses.KLDivergence(
                reduction="sum_over_batch_size"
            )
        elif loss_type == "mse":
            self.loss_fn = keras.losses.MeanSquaredError(
                reduction="sum_over_batch_size"
            )
        elif loss_type == "cross_entropy":
            self.loss_fn = keras.losses.CategoricalCrossentropy(
                from_logits=False, reduction="sum_over_batch_size"
            )

    def compute_loss(self, teacher_outputs, student_outputs):
        """Compute distillation loss using Keras built-in loss functions.

        Args:
            teacher_outputs: Logits from teacher model.
            student_outputs: Logits from student model.

        Returns:
            Distillation loss tensor.
        """
        if self.loss_type == "custom":
            return self.custom_loss_fn(teacher_outputs, student_outputs)

        # Apply temperature scaling
        teacher_logits = teacher_outputs / self.temperature
        student_logits = student_outputs / self.temperature

        if self.loss_type == "kl_divergence":
            # Convert to probabilities for KL divergence
            teacher_probs = ops.softmax(teacher_logits, axis=-1)
            student_probs = ops.softmax(student_logits, axis=-1)

            # Use Keras KLDivergence
            loss = self.loss_fn(teacher_probs, student_probs)

        elif self.loss_type == "mse":
            # Use Keras MeanSquaredError directly on logits
            loss = self.loss_fn(teacher_logits, student_logits)

        elif self.loss_type == "cross_entropy":
            # Convert teacher to probabilities, keep student as logits
            teacher_probs = ops.softmax(teacher_logits, axis=-1)

            # Use Keras CategoricalCrossentropy
            loss = self.loss_fn(teacher_probs, student_logits)

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        # Scale by temperature^2 for consistency with literature
        return loss * (self.temperature**2)

    def get_config(self):
        """Get configuration for serialization."""
        return {
            "temperature": self.temperature,
            "loss_type": self.loss_type,
            "custom_loss_fn": self.custom_loss_fn,
        }


@keras_hub_export("keras_hub.distillation.FeatureDistillation")
class FeatureDistillation(BaseDistillationStrategy):
    """Feature distillation strategy using Keras built-in loss functions.

    This strategy distills intermediate features from teacher to student,
    not just the final outputs.

    Args:
        loss_type: Type of loss function to use. Options:
            - "mse": Mean squared error using keras.losses.MeanSquaredError
            - "cosine": Cosine similarity using keras.losses.CosineSimilarity
            - "custom": Use custom_loss_fn
        custom_loss_fn: Custom loss function for features.
    """

    def __init__(self, loss_type="mse", custom_loss_fn=None):
        self.loss_type = loss_type
        self.custom_loss_fn = custom_loss_fn

        # Validate loss_type
        valid_loss_types = ["mse", "cosine", "custom"]
        if loss_type not in valid_loss_types:
            raise ValueError(f"loss_type must be one of {valid_loss_types}")

        if loss_type == "custom" and custom_loss_fn is None:
            raise ValueError(
                "custom_loss_fn must be provided when loss_type='custom'"
            )

        # Initialize Keras loss functions
        if loss_type == "mse":
            self.loss_fn = keras.losses.MeanSquaredError(
                reduction="sum_over_batch_size"
            )
        elif loss_type == "cosine":
            self.loss_fn = keras.losses.CosineSimilarity(
                reduction="sum_over_batch_size"
            )

    def compute_loss(self, teacher_features, student_features):
        """Compute feature distillation loss using Keras built-in loss
        functions.

        Args:
            teacher_features: Intermediate features from teacher model.
            student_features: Intermediate features from student model.

        Returns:
            Feature distillation loss tensor.
        """
        if self.loss_type == "custom":
            return self.custom_loss_fn(teacher_features, student_features)

        if self.loss_type == "mse":
            # Use Keras MeanSquaredError
            return self.loss_fn(teacher_features, student_features)

        elif self.loss_type == "cosine":
            # Use Keras CosineSimilarity (returns similarity, convert to
            # distance)
            similarity = self.loss_fn(teacher_features, student_features)
            # Convert similarity to distance: distance = 1 - similarity
            return 1.0 - similarity

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

    def get_config(self):
        """Get configuration for serialization."""
        return {
            "loss_type": self.loss_type,
            "custom_loss_fn": self.custom_loss_fn,
        }
