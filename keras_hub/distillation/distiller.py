# Copyright 2024 The KerasHub Authors. All Rights Reserved.
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
"""Distiller class for KerasHub."""
import keras
# Placeholder for keras_hub import, assuming it will be available in the environment.
# import keras_hub
import os

from .strategies import LogitsDistillation, AttentionDistillation, FeatureDistillation

class Distiller(keras.Model):
    """A Keras Model for knowledge distillation.

    The `Distiller` class orchestrates the distillation process between a larger
    teacher model and a smaller student model. It wraps both models and manages
    the computation of losses, including the student's standard loss on ground
    truth (if provided) and various distillation losses that encourage the
    student to mimic the teacher's behavior.

    The Distiller can be used with "classic" logits distillation (matching softened
    output distributions) or with a list of modular distillation `strategies`
    that can target intermediate features, attention maps, or other custom aspects.

    Key functionalities:
    - Freezes the teacher model during training.
    - Computes a combined loss from student's hard-target loss and distillation losses.
    - Updates only the student model's weights.
    - Integrates with standard Keras `compile()` and `fit()` workflows.
    - Tracks individual loss components (student loss, distillation loss, total loss)
      as Keras metrics.

    Args:
        student: A `keras.Model` instance representing the student model.
            Its weights will be updated during distillation.
        teacher: A `keras.Model` instance representing the teacher model.
            It will be frozen (not trained) and used for generating targets.
        strategies: Optional list of distillation strategy instances (e.g.,
            `LogitsDistillation`, `AttentionDistillation`). If None or empty,
            "classic" logits distillation can be configured via `compile()`.
        name: Optional string, name for the Distiller model.
        **kwargs: Additional keyword arguments passed to `keras.Model.__init__()`.

    Raises:
        ValueError: If `student` or `teacher` are not `keras.Model` instances,
            or if `strategies` is not a list.
    """
    def __init__(self, student, teacher, strategies=None, name="distiller", **kwargs):
        super().__init__(name=name, **kwargs)
        if not isinstance(student, keras.Model):
            raise ValueError(f"`student` must be a `keras.Model` instance, got: {student}")
        if not isinstance(teacher, keras.Model):
            raise ValueError(f"`teacher` must be a `keras.Model` instance, got: {teacher}")

        self.student = student
        self.teacher = teacher
        self.teacher.trainable = False  # Ensure teacher is frozen

        self.strategies = strategies if strategies is not None else []
        if not isinstance(self.strategies, list):
            raise ValueError(f"`strategies` must be a list of strategy instances, got: {self.strategies}")

        self.student_loss_fn = None
        self.distillation_loss_fn = None # For "classic" distillation if no strategies.
        self.alpha = 0.0 # Weight for student_loss_fn
        self.temperature = 1.0 # For "classic" distillation

        # Metrics for tracking individual loss components
        self.student_loss_tracker = keras.metrics.Mean(name="student_loss")
        self.distillation_loss_tracker = keras.metrics.Mean(name="distillation_loss") # Tracks sum of strategy losses or classic distillation loss
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss") # Tracks the final weighted sum loss

    def compile(self, optimizer, student_loss_fn=None, distillation_loss_fn=None, metrics=None, alpha=0.5, temperature=1.0, **kwargs):
        """ Configure the distiller for training.

        Args:
            optimizer: Optimizer instance for the student model.
            student_loss_fn: Optional. Loss function for the student model's direct predictions against true labels.
                             Used if `y` is provided in `fit()`.
            distillation_loss_fn: Optional. Loss function for "classic" logits distillation if no `strategies`
                                  are provided to the Distiller constructor. Typically `keras.losses.KLDivergence()`.
                                  This is ignored if `strategies` list is non-empty.
            metrics: Optional. List of metrics to be evaluated by the model during training and testing.
                     These are applied to the student's direct output (`y_pred`).
            alpha: Float between 0 and 1. Weight for the `student_loss_fn`. The weight for the
                   `distillation_loss_fn` (if used) will be `(1 - alpha)`.
                   If `strategies` are used, their individual weights apply, and `alpha` applies to `student_loss_fn`
                   while the sum of strategy losses gets a weight of 1.0 (as their internal weights define proportions).
                   More precisely, total_loss = alpha * student_loss + sum(strategy.compute_loss for strategy in strategies).
                   If using classic distillation (no strategies): total_loss = alpha * student_loss + (1-alpha) * distillation_loss.
            temperature: Float. Temperature for softening probability distributions in "classic" logits distillation.
                         This is ignored if `strategies` list is non-empty (as strategies like `LogitsDistillation`
                         have their own temperature).
            **kwargs: Additional keyword arguments for `keras.Model.compile()`.
        """
        super().compile(optimizer=optimizer, metrics=metrics, **kwargs)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn # Used only if self.strategies is empty
        self.alpha = alpha
        self.temperature = temperature

        if not self.strategies and (self.student_loss_fn is None and self.distillation_loss_fn is None):
            raise ValueError(
                "If no specific distillation `strategies` are provided to the Distiller, "
                "either `student_loss_fn` or `distillation_loss_fn` (or both) must be specified in `compile()`."
            )
        if self.strategies and self.distillation_loss_fn is not None:
            print(
                "Warning: `distillation_loss_fn` was provided to `compile()` but `strategies` "
                "are also configured for the Distiller. The `distillation_loss_fn` and `temperature` "
                "compile arguments will be ignored in favor of the configured strategies."
            )
        if not self.strategies and self.distillation_loss_fn is None and self.alpha < 1.0:
             print(
                "Warning: `alpha` is less than 1.0 but `distillation_loss_fn` is not provided and no strategies are set. "
                "The distillation component of the loss will be zero."
            )
        if self.student_loss_fn is None and self.alpha > 0.0:
            print(
                "Warning: `alpha` is greater than 0.0 but `student_loss_fn` is not provided. "
                "The student's hard target loss component will be zero."
            )

    def call(self, inputs, training=False):
        # When Distiller is called directly (e.g., for inference or evaluation),
        # it behaves like the student model.
        return self.student(inputs, training=training)

    def compute_loss(self, x, y, y_pred, sample_weight=None, training=True):
        """ Computes the total loss for distillation.

        This method is called by `train_step` and `test_step`.

        Args:
            x: Input data.
            y: True labels. Can be None if only distillation is performed.
            y_pred: Predictions from the student model.
            sample_weight: Optional sample weights. Currently not explicitly used by custom
                           loss calculations but passed for compatibility.
            training: Boolean, whether the model is in training mode. This affects
                      how teacher/student models are called (e.g. for dropout).

        Returns:
            The total computed loss.
        """
        del sample_weight # sample_weight is not used by this custom logic but is part of Keras API.

        # Initialize losses to zero tensors in a way that works with the current backend and device.
        # We get a sample tensor from y_pred if available, or make a scalar 0.
        # This ensures operations like addition work correctly across backends.
        if y_pred is not None and hasattr(y_pred, 'shape') and y_pred.shape.rank > 0 : # Check if y_pred is a tensor with shape
            zero_loss_ref = keras.ops.zeros_like(y_pred[0]) # Use a slice if y_pred is batched
            if zero_loss_ref.shape.rank > 0: # If y_pred[0] is still a tensor, sum it to get a scalar
                 zero_loss_ref = keras.ops.sum(zero_loss_ref)
        else: # Fallback for scalar y_pred or if y_pred is None (though less likely here)
            zero_loss_ref = keras.ops.convert_to_tensor(0.0)


        student_loss = zero_loss_ref
        distillation_loss_sum = zero_loss_ref

        # 1. Compute student's hard target loss (if applicable)
        if self.student_loss_fn is not None and y is not None and self.alpha > 0:
            # Ensure y and y_pred are in a compatible format for the loss function
            # This might involve ensuring they are on the same device, have same dtype, etc.
            # Keras ops should handle this generally.
            current_student_loss = self.student_loss_fn(y, y_pred)
            student_loss = self.alpha * current_student_loss
            self.student_loss_tracker.update_state(current_student_loss) # Track unweighted student loss
        else:
            self.student_loss_tracker.update_state(zero_loss_ref) # Update with 0 if not used or alpha is 0

        # 2. Compute distillation loss(es)
        # This section runs whether training or not, as teacher outputs are needed for eval too.
        # The `training` flag is passed to teacher/student calls appropriately.

        if self.strategies:
            # Teacher outputs for strategies that need main model logits (like LogitsDistillation)
            # Other strategies (Attention, Feature) call teacher/student internally with inputs `x`.
            # We call the teacher once here if any LogitsDistillation strategy is present,
            # or if it's needed by a custom strategy.
            # A more optimized approach might involve checking strategy types.
            # For now, assume teacher_outputs_main might be needed by some strategy.
            teacher_outputs_main = self.teacher(x, training=False) # Teacher always in inference mode

            current_distillation_loss_total_for_strategies = zero_loss_ref
            for strategy in self.strategies:
                if isinstance(strategy, LogitsDistillation):
                    # LogitsDistillation uses the main teacher/student outputs (y_pred is student_outputs_main)
                    loss_val = strategy.compute_loss(teacher_outputs_main, y_pred) # Weight is applied inside strategy.compute_loss
                    current_distillation_loss_total_for_strategies += loss_val
                elif isinstance(strategy, (AttentionDistillation, FeatureDistillation)):
                    # These strategies manage their own submodel calls using `x`
                    loss_val = strategy.compute_loss(self.teacher, self.student, x) # Weight is applied inside strategy.compute_loss
                    current_distillation_loss_total_for_strategies += loss_val
                else:
                    # Fallback for custom strategies - they might need pre-computed teacher/student main outputs
                    # Or they might operate like Attention/Feature. For now, assume they take main outputs.
                    # This part can be refined if we define a stricter API for custom strategies.
                    # Assuming custom strategy's compute_loss takes (teacher_main_output, student_main_output)
                    # and handles its own weight.
                    loss_val = strategy.compute_loss(teacher_outputs_main, y_pred)
                    current_distillation_loss_total_for_strategies += loss_val

            distillation_loss_sum = current_distillation_loss_total_for_strategies
            self.distillation_loss_tracker.update_state(distillation_loss_sum) # Sum of all weighted strategy losses

        elif self.distillation_loss_fn is not None and (1.0 - self.alpha) > 0:
            # "Classic" logit distillation (if no strategies are provided and distillation is weighted)
            teacher_logits = self.teacher(x, training=False) # Teacher always in inference mode
            # student_logits are y_pred, already computed

            # Soften probabilities with temperature
            # Ensure teacher_logits and y_pred (student_logits) are compatible for softmax and loss_fn
            teacher_probs = keras.ops.softmax(teacher_logits / self.temperature, axis=-1)
            student_probs = keras.ops.softmax(y_pred / self.temperature, axis=-1)

            # Apply T^2 scaling to distillation loss as per original Hinton paper and common practice
            classic_distill_loss = self.distillation_loss_fn(teacher_probs, student_probs) * (self.temperature**2)
            distillation_loss_sum = (1.0 - self.alpha) * classic_distill_loss
            self.distillation_loss_tracker.update_state(classic_distill_loss) # Track unweighted classic distillation loss
        else:
            self.distillation_loss_tracker.update_state(zero_loss_ref) # Update with 0 if not used

        # Total loss is sum of student loss and distillation loss(es)
        # If strategies are used, their individual weights are already applied within their compute_loss.
        # If classic distillation, (1-alpha) is applied.
        # Student loss already has alpha applied.
        total_loss = student_loss + distillation_loss_sum
        self.total_loss_tracker.update_state(total_loss)

        return total_loss

    def train_step(self, data):
        # Unpack the data. `y_true` can be None if pure distillation (unlabeled data)
        # or if student_loss_fn is not provided.
        if len(data) == 1: # Assume data is x only
            x, y_true = data[0], None
        elif len(data) == 2:
            x, y_true = data
        else: # len(data) == 3 could be (x, y, sample_weight), but we ignore sample_weight for now
             x, y_true = data[0], data[1]


        # Keras 3 automatically handles data unpacking if it's a dict, tf.data.Dataset, etc.
        # The above is a simplified version assuming tuple/list input.
        # More robust unpacking might be needed if directly using this train_step
        # outside of model.fit() which handles data adapter logic.
        # However, model.fit() will typically provide data as (x,) or (x,y) or (x,y,sw).

        # Forward pass through student and compute all losses under GradientTape
        with keras.backend.GradientTape() as tape:
            y_pred = self.student(x, training=True)
            # compute_loss needs to be called with training=True context
            loss = self.compute_loss(x=x, y=y_true, y_pred=y_pred, sample_weight=None, training=True)

        # Compute gradients and update student weights
        # Only student's trainable variables should be updated.
        trainable_vars = self.student.trainable_variables
        if not trainable_vars:
            raise ValueError(
                "The student model does not have any trainable variables. "
                "Ensure your student model is built correctly and not frozen."
            )

        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update compiled metrics (these are for student's direct output vs y_true)
        # The compiled_metrics are passed in model.compile(metrics=...)
        # They are only updated if y_true is available.
        if y_true is not None and self.compiled_metrics is not None:
             self.compiled_metrics.update_state(y_true, y_pred) # sample_weight can be added if supported

        # Return a dict of metrics including our custom loss trackers
        # The `loss` key here will be the total_loss from total_loss_tracker.
        results = {m.name: m.result() for m in self.metrics} # self.metrics includes the loss trackers
        return results

    def test_step(self, data):
        # Unpack the data, similar to train_step
        if len(data) == 1:
            x, y_true = data[0], None
        elif len(data) == 2:
            x, y_true = data
        else:
            x, y_true = data[0], data[1]

        # Forward pass through student (in inference mode)
        y_pred = self.student(x, training=False)

        # Compute losses (in inference mode for loss calculation)
        # This will update our loss trackers (student_loss, distillation_loss, total_loss)
        self.compute_loss(x=x, y=y_true, y_pred=y_pred, sample_weight=None, training=False)

        # Update compiled metrics (if y_true is available)
        if y_true is not None and self.compiled_metrics is not None:
            self.compiled_metrics.update_state(y_true, y_pred)

        results = {m.name: m.result() for m in self.metrics}
        return results

    @property
    def metrics(self):
        # Extend the model's metrics to include our custom loss trackers
        # This ensures they are reset correctly and displayed during training/evaluation.
        # `compiled_metrics` are those passed by the user to `compile()`.
        # `loss_trackers` are our internal ones.
        metrics_list = [
            self.total_loss_tracker,
            self.student_loss_tracker,
            self.distillation_loss_tracker
        ]
        if self.compiled_metrics is not None: # compiled_metrics is a MetricList object or None
            metrics_list += self.compiled_metrics.metrics # Access the list of metrics
        return metrics_list

    # get_metrics_result and reset_states are typically handled by the base Model
    # by virtue of including trackers in self.metrics.
    # However, the design doc specifies them, so we can ensure they are explicitly here
    # or confirm if the base Model behavior is sufficient.
    # For now, let's rely on the base Model's handling via the `metrics` property.
    # If custom behavior beyond what `metrics` property provides is needed,
    # we can implement them. The design doc's reset_states example is good practice.

    def get_metrics_result(self):
        # This method is often called by Callbacks or at the end of epoch.
        # Keras Model.get_metrics_result() already does this by iterating self.metrics.
        # We can provide a more specific dict if needed.
        results = {m.name: m.result() for m in self.metrics}
        # Ensure standard "loss" key is present, mapped from total_loss_tracker
        if "total_loss" in results:
             results["loss"] = results["total_loss"]
        return results

    def reset_states(self):
        # Reset custom metric trackers
        self.student_loss_tracker.reset_states()
        self.distillation_loss_tracker.reset_states()
        self.total_loss_tracker.reset_states()
        # Call parent's reset_states for compiled metrics and other internal states
        super().reset_states()

# The design doc mentioned `keras.backend.device_scope(self.distiller.student.device)`
# in train_step. This is generally not needed with Keras 3 as operations are dispatched
# based on tensor device placement. If specific device pinning is required for parts of
# the model, it's usually handled at the model/layer construction or backend configuration level.
# For now, I'm omitting it as Keras 3 aims to simplify this.
# If issues arise in specific backend/distribution scenarios, it could be revisited.
