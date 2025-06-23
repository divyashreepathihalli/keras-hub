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
"""Tests for KerasHub Distiller and strategies."""

import keras
import numpy as np
# import unittest # No longer primary, but useful for context of what was replaced
import io
import contextlib

from keras_hub.distillation import Distiller
from keras_hub.distillation import LogitsDistillation
from keras_hub.distillation import AttentionDistillation
from keras_hub.distillation import FeatureDistillation
from keras_hub.src.tests.test_case import TestCase


class DistillerTest(TestCase):

    def _create_simple_model(self, name="model", input_shape=(10,), output_dim=5, num_hidden_layers=1, hidden_dim=20):
        """Creates a simple sequential model for testing."""
        inputs = keras.Input(shape=input_shape, name=f"{name}_input")
        x = inputs
        for i in range(num_hidden_layers):
            x = keras.layers.Dense(hidden_dim, activation="relu", name=f"{name}_hidden_{i}")(x)
        x = keras.layers.Dense(hidden_dim // 2, activation="relu", name=f"{name}_intermediate_features")(x)
        outputs = keras.layers.Dense(output_dim, name=f"{name}_output_logits")(x)
        return keras.Model(inputs=inputs, outputs=outputs, name=name)

    def _get_teacher_model(self):
        return self._create_simple_model(name="teacher", input_shape=(10,), output_dim=5, num_hidden_layers=2, hidden_dim=32)

    def _get_student_model(self):
        return self._create_simple_model(name="student", input_shape=(10,), output_dim=5, num_hidden_layers=1, hidden_dim=16)

    def _get_dummy_data(self):
        x = np.random.rand(4, 10).astype(np.float32)
        y = np.random.rand(4, 5).astype(np.float32)
        return x, y

    def setUp(self):
        # Common setup for tests if needed, e.g., clearing Keras session
        keras.backend.clear_session()
        self.teacher_model = self._get_teacher_model()
        self.student_model = self._get_student_model()
        self.dummy_x, self.dummy_y = self._get_dummy_data()

        # Build models by calling them once
        _ = self.teacher_model(self.dummy_x)
        _ = self.student_model(self.dummy_x)


    def test_distiller_instantiation(self):
        distiller = Distiller(student=self.student_model, teacher=self.teacher_model)
        self.assertIs(distiller.student, self.student_model)
        self.assertIs(distiller.teacher, self.teacher_model)
        self.assertFalse(distiller.teacher.trainable)
        self.assertEqual(distiller.name, "distiller")

    def test_distiller_instantiation_with_strategies(self):
        dummy_loss = keras.losses.MeanSquaredError()
        strategies = [
            LogitsDistillation(temperature=2.0, loss_fn=dummy_loss, weight=0.5)
        ]
        distiller = Distiller(student=self.student_model, teacher=self.teacher_model, strategies=strategies)
        self.assertEqual(len(distiller.strategies), 1)
        self.assertIsInstance(distiller.strategies[0], LogitsDistillation)

    def test_distiller_compile_minimal(self):
        distiller = Distiller(student=self.student_model, teacher=self.teacher_model)
        distiller.compile(
            optimizer=keras.optimizers.Adam(),
            distillation_loss_fn=keras.losses.KLDivergence(),
            alpha=0.0
        )
        self.assertIsNotNone(distiller.distillation_loss_fn)
        self.assertIsNone(distiller.student_loss_fn)
        self.assertEqual(distiller.alpha, 0.0)

    def test_distiller_compile_with_student_loss(self):
        distiller = Distiller(student=self.student_model, teacher=self.teacher_model)
        distiller.compile(
            optimizer=keras.optimizers.Adam(),
            student_loss_fn=keras.losses.MeanSquaredError(),
            alpha=1.0
        )
        self.assertIsNotNone(distiller.student_loss_fn)
        self.assertIsNone(distiller.distillation_loss_fn)
        self.assertEqual(distiller.alpha, 1.0)

    def test_distiller_compile_full_classic(self):
        distiller = Distiller(student=self.student_model, teacher=self.teacher_model)
        distiller.compile(
            optimizer=keras.optimizers.Adam(),
            student_loss_fn=keras.losses.CategoricalCrossentropy(),
            distillation_loss_fn=keras.losses.KLDivergence(),
            metrics=[keras.metrics.CategoricalAccuracy()],
            alpha=0.3,
            temperature=2.5
        )
        self.assertIsNotNone(distiller.student_loss_fn)
        self.assertIsNotNone(distiller.distillation_loss_fn)
        # Check if the compiled metric (CategoricalAccuracy) is part of the model's metrics
        self.assertTrue(any(isinstance(m, keras.metrics.CategoricalAccuracy) for m in distiller.metrics))
        self.assertEqual(distiller.alpha, 0.3)
        self.assertEqual(distiller.temperature, 2.5)

    def test_distiller_compile_with_strategies_ignores_classic_distill_loss(self):
        strategies = [LogitsDistillation(temperature=2.0, loss_fn=keras.losses.MeanSquaredError(), weight=0.5)]
        distiller = Distiller(student=self.student_model, teacher=self.teacher_model, strategies=strategies)

        stdout_capture = io.StringIO()
        with contextlib.redirect_stdout(stdout_capture):
            distiller.compile(
                optimizer=keras.optimizers.Adam(),
                student_loss_fn=keras.losses.MeanSquaredError(),
                distillation_loss_fn=keras.losses.KLDivergence(), # This should be ignored
                alpha=0.5
            )
        output = stdout_capture.getvalue()
        self.assertIn("Warning: `distillation_loss_fn` was provided to `compile()` but `strategies`", output)
        self.assertIsNotNone(distiller.distillation_loss_fn)
        self.assertTrue(len(distiller.strategies) > 0)

    def test_distiller_compile_value_errors(self):
        distiller = Distiller(student=self.student_model, teacher=self.teacher_model)
        with self.assertRaisesRegex(ValueError, "either `student_loss_fn` or `distillation_loss_fn`"):
            distiller.compile(optimizer=keras.optimizers.Adam(), alpha=0.5)

        with self.assertRaisesRegex(ValueError, "`student` must be a `keras.Model` instance"):
            Distiller(student="not_a_model", teacher=self.teacher_model)

        with self.assertRaisesRegex(ValueError, "`teacher` must be a `keras.Model` instance"):
            Distiller(student=self.student_model, teacher="not_a_model")

        with self.assertRaisesRegex(ValueError, "`strategies` must be a list"):
            Distiller(student=self.student_model, teacher=self.teacher_model, strategies="not_a_list")

    def test_distiller_call_behaves_as_student(self):
        distiller = Distiller(student=self.student_model, teacher=self.teacher_model)
        # Call distiller to build it if not already (though setUp should handle model builds)
        _ = distiller(self.dummy_x)

        student_output = self.student_model(self.dummy_x, training=False)
        distiller_output = distiller(self.dummy_x, training=False)

        np.testing.assert_array_almost_equal(distiller_output.numpy(), student_output.numpy())

    def _run_train_step_classic_distillation(self, use_student_loss, use_classic_distill_loss):
        if not use_student_loss and not use_classic_distill_loss:
            # This case would be caught by compile, but for direct train_step test, we ensure a loss is active
            # Or rely on compile to be called first. Let's assume compile has set it up.
            return

        distiller = Distiller(student=self.student_model, teacher=self.teacher_model)
        student_loss_fn = keras.losses.MeanSquaredError() if use_student_loss else None
        distill_loss_fn = keras.losses.KLDivergence() if use_classic_distill_loss else None
        alpha = 0.0
        if use_student_loss and use_classic_distill_loss: alpha = 0.5
        elif use_student_loss: alpha = 1.0

        effective_distill_loss_fn = distill_loss_fn
        if student_loss_fn is None and distill_loss_fn is None: # Ensure compile validity
             effective_distill_loss_fn = keras.losses.KLDivergence() # Default if nothing else provided

        # If y_true will be None (student_loss_fn is None or alpha is 0 for student part),
        # then metrics requiring y_true (like MAE) should not be passed or will cause issues.
        test_metrics = [keras.metrics.MeanAbsoluteError()]
        if student_loss_fn is None: # This implies y_true will be None in train_step
            test_metrics = None # Or []

        distiller.compile(
            optimizer=keras.optimizers.Adam(),
            student_loss_fn=student_loss_fn,
            distillation_loss_fn=effective_distill_loss_fn,
            metrics=test_metrics,
            alpha=alpha, temperature=2.0
        )

        initial_student_weights = [w.numpy() for w in self.student_model.trainable_variables]

        data_to_pass = (self.dummy_x, self.dummy_y) if student_loss_fn else (self.dummy_x,)
        if not use_student_loss : # If student_loss_fn is None, y should be None
            data_to_pass = (self.dummy_x,)


        results = distiller.train_step(data=data_to_pass)

        self.assertIn("loss", results)
        self.assertIsInstance(results["loss"].numpy(), (float, np.floating))
        self.assertIn("student_loss", results)
        self.assertIsInstance(results["student_loss"].numpy(), (float, np.floating))
        self.assertIn("distillation_loss", results)
        self.assertIsInstance(results["distillation_loss"].numpy(), (float, np.floating))

        if use_student_loss and student_loss_fn:
            self.assertTrue(results["student_loss"].numpy() >= 0) # Can be >0 if y provided
        else:
            self.assertEqual(results["student_loss"].numpy(), 0.0)

        if use_classic_distill_loss:
            self.assertTrue(results["distillation_loss"].numpy() >= 0) # KLD is >=0

        updated_student_weights = [w.numpy() for w in self.student_model.trainable_variables]
        weight_changed = any(not np.array_equal(initial, updated) for initial, updated in zip(initial_student_weights, updated_student_weights))

        a_loss_was_active = (use_student_loss and student_loss_fn and alpha > 0) or \
                            (use_classic_distill_loss and distill_loss_fn and (1-alpha) > 0)

        if a_loss_was_active and self.student_model.trainable_variables:
            self.assertTrue(weight_changed, "Student weights were not updated when a loss was active.")
        elif self.student_model.trainable_variables: # No loss active but trainable vars exist
             self.assertFalse(weight_changed, "Student weights changed unexpectedly with no active loss.")

        if student_loss_fn and distiller.compiled_metrics is not None and results.get("compile_metrics"):
            self.assertIn("mean_absolute_error", results["compile_metrics"])
            self.assertTrue(results["compile_metrics"]["mean_absolute_error"].numpy() >= 0)

    def test_train_step_classic_student_and_distill_loss(self):
        self._run_train_step_classic_distillation(use_student_loss=True, use_classic_distill_loss=True)

    def test_train_step_classic_student_only_loss(self):
        self._run_train_step_classic_distillation(use_student_loss=True, use_classic_distill_loss=False)

    def test_train_step_classic_distill_only_loss(self):
        self._run_train_step_classic_distillation(use_student_loss=False, use_classic_distill_loss=True)


    def test_test_step_classic_distillation(self):
        distiller = Distiller(student=self.student_model, teacher=self.teacher_model)
        distiller.compile(
            optimizer=keras.optimizers.Adam(),
            student_loss_fn=keras.losses.MeanSquaredError(),
            distillation_loss_fn=keras.losses.KLDivergence(),
            metrics=[keras.metrics.MeanAbsoluteError()],
            alpha=0.5, temperature=2.0
        )
        initial_student_weights = [w.numpy() for w in self.student_model.trainable_variables]
        results = distiller.test_step(data=(self.dummy_x, self.dummy_y))

        self.assertIn("loss", results)
        self.assertIn("student_loss", results)
        self.assertIn("distillation_loss", results)

        updated_student_weights = [w.numpy() for w in self.student_model.trainable_variables]
        weight_changed = any(not np.array_equal(initial, updated) for initial, updated in zip(initial_student_weights, updated_student_weights))
        self.assertFalse(weight_changed, "Student weights were updated during test_step.")

        if distiller.compiled_metrics is not None and results.get("compile_metrics"):
            self.assertIn("mean_absolute_error", results["compile_metrics"])
            self.assertTrue(results["compile_metrics"]["mean_absolute_error"].numpy() >= 0)

    def test_train_step_with_no_labels(self):
        distiller = Distiller(student=self.student_model, teacher=self.teacher_model)
        distiller.compile(
            optimizer=keras.optimizers.Adam(),
            distillation_loss_fn=keras.losses.KLDivergence(),
            alpha=0.0
        )
        results = distiller.train_step(data=(self.dummy_x,))
        self.assertIn("loss", results)
        self.assertEqual(results["student_loss"].numpy(), 0.0)
        self.assertTrue(results["distillation_loss"].numpy() >= 0.0)


    def test_reset_states(self):
        distiller = Distiller(student=self.student_model, teacher=self.teacher_model)
        distiller.compile(
            optimizer=keras.optimizers.Adam(),
            student_loss_fn=keras.losses.MeanSquaredError(),
            distillation_loss_fn=keras.losses.KLDivergence(),
            metrics=[keras.metrics.MeanAbsoluteError()],
            alpha=0.5
        )
        distiller.train_step(data=(self.dummy_x, self.dummy_y))
        self.assertNotEqual(distiller.total_loss_tracker.result().numpy(), 0.0)
        self.assertNotEqual(distiller.student_loss_tracker.result().numpy(), 0.0)

        distiller.reset_states()
        self.assertEqual(distiller.total_loss_tracker.result().numpy(), 0.0)
        self.assertEqual(distiller.student_loss_tracker.result().numpy(), 0.0)
        self.assertEqual(distiller.distillation_loss_tracker.result().numpy(), 0.0)
        if distiller.compiled_metrics and distiller.compiled_metrics.metrics:
            self.assertEqual(distiller.compiled_metrics.metrics[0].result().numpy(), 0.0)

    def test_logits_distillation_strategy_train_step(self):
        logits_strategy = LogitsDistillation(
            temperature=3.0, loss_fn=keras.losses.KLDivergence(), weight=1.0
        )
        distiller = Distiller(student=self.student_model, teacher=self.teacher_model, strategies=[logits_strategy])
        distiller.compile(
            optimizer=keras.optimizers.Adam(),
            student_loss_fn=keras.losses.MeanSquaredError(), alpha=0.1
        )
        results = distiller.train_step(data=(self.dummy_x, self.dummy_y))
        self.assertIn("loss", results)
        if distiller.alpha > 0: self.assertTrue(results["student_loss"].numpy() >= 0)
        else: self.assertEqual(results["student_loss"].numpy(), 0.0)
        self.assertTrue(results["distillation_loss"].numpy() >= 0)

    def test_attention_distillation_strategy_train_step(self):
        teacher_att_layer = "teacher_intermediate_features" # From teacher_model (hidden_dim=32 -> intermediate_features_dim=16)

        # Create a student model for this test with compatible intermediate layer dimensions
        student_model_att_compat = self._create_simple_model(name="student_att_compat", hidden_dim=32)
        _ = student_model_att_compat(self.dummy_x) # Build it
        student_att_layer = "student_att_compat_intermediate_features" # intermediate_features_dim=16

        attention_strategy = AttentionDistillation(
            teacher_layer=teacher_att_layer, student_layer=student_att_layer,
            loss_fn=keras.losses.MeanSquaredError(), weight=0.7
        )
        # Use the compatible student model for this distiller instance
        distiller = Distiller(student=student_model_att_compat, teacher=self.teacher_model, strategies=[attention_strategy])
        distiller.compile(optimizer=keras.optimizers.Adam(), student_loss_fn=keras.losses.MeanSquaredError(), alpha=0.2)

        _ = distiller.train_step(data=(self.dummy_x,self.dummy_y)) # Build submodels
        results = distiller.train_step(data=(self.dummy_x,self.dummy_y))

        self.assertIn("loss", results)
        self.assertTrue(results["distillation_loss"].numpy() > 0)
        if distiller.alpha > 0: self.assertTrue(results["student_loss"].numpy() >= 0)
        else: self.assertEqual(results["student_loss"].numpy(), 0.0)
        self.assertIsNotNone(attention_strategy.teacher_submodel)
        self.assertIsNotNone(attention_strategy.student_submodel)

    def test_feature_distillation_strategy_train_step_no_projection(self):
        student_model_compat = self._create_simple_model(name="student_compat", hidden_dim=32) # intermediate = 16
        _ = student_model_compat(self.dummy_x) # Build it

        feature_strategy = FeatureDistillation(
            teacher_layer="teacher_intermediate_features", student_layer="student_compat_intermediate_features",
            loss_fn=keras.losses.CosineSimilarity(), weight=0.6, projection=False
        )
        distiller = Distiller(student=student_model_compat, teacher=self.teacher_model, strategies=[feature_strategy])
        distiller.compile(optimizer=keras.optimizers.Adam(), alpha=0.0)
        results = distiller.train_step(data=(self.dummy_x,))
        self.assertIn("loss", results)
        self.assertNotEqual(results["distillation_loss"].numpy(), 0)
        self.assertIsNone(feature_strategy.projection_layer)

    def test_feature_distillation_with_projection(self):
        teacher_feat_layer = "teacher_intermediate_features"
        student_feat_layer = "student_intermediate_features"
        feature_strategy = FeatureDistillation(
            teacher_layer=teacher_feat_layer, student_layer=student_feat_layer,
            loss_fn=keras.losses.MeanSquaredError(), weight=0.6, projection=True
        )
        distiller = Distiller(student=self.student_model, teacher=self.teacher_model, strategies=[feature_strategy])
        distiller.compile(optimizer=keras.optimizers.Adam(), alpha=0.0)
        _ = distiller.train_step(data=(self.dummy_x,))

        self.assertIsNotNone(feature_strategy.projection_layer)
        self.assertTrue(feature_strategy.projection_layer.built)
        expected_student_dim = self.student_model.get_layer(student_feat_layer).output_shape[-1]
        self.assertEqual(feature_strategy.projection_layer.units, expected_student_dim)
        results = distiller.train_step(data=(self.dummy_x,))
        self.assertIn("loss", results)
        self.assertTrue(results["distillation_loss"].numpy() > 0)

    def test_multiple_strategies_train_step(self):
        student_model_multi = self._create_simple_model(name="student_multi", hidden_dim=32)
        _ = student_model_multi(self.dummy_x) # Build

        logits_strat = LogitsDistillation(temperature=2.0, loss_fn=keras.losses.KLDivergence(), weight=0.5)
        feature_strat = FeatureDistillation(
            teacher_layer="teacher_intermediate_features", student_layer="student_multi_intermediate_features",
            loss_fn=keras.losses.MeanSquaredError(), weight=0.3, projection=False
        )
        distiller = Distiller(student=student_model_multi, teacher=self.teacher_model, strategies=[logits_strat, feature_strat])
        distiller.compile(optimizer=keras.optimizers.Adam(), student_loss_fn=keras.losses.MeanSquaredError(), alpha=0.2)
        results = distiller.train_step(data=(self.dummy_x, self.dummy_y))
        self.assertIn("loss", results)
        if distiller.alpha > 0: self.assertTrue(results["student_loss"].numpy() >= 0)
        else: self.assertEqual(results["student_loss"].numpy(), 0.0)
        self.assertNotEqual(results["distillation_loss"].numpy(), 0)

    def _check_layer_not_found(self, strategy_class, bad_teacher_layer, bad_student_layer, match_regex):
        if bad_teacher_layer:
            strategy = strategy_class("nonexistent_teacher_layer", "student_intermediate_features", keras.losses.MeanSquaredError(), 1.0)
        else: # bad_student_layer
            strategy = strategy_class("teacher_intermediate_features", "nonexistent_student_layer", keras.losses.MeanSquaredError(), 1.0)

        distiller = Distiller(self.student_model, self.teacher_model, strategies=[strategy])
        distiller.compile(optimizer=keras.optimizers.Adam())
        with self.assertRaisesRegex(ValueError, match_regex):
            distiller.train_step(data=(self.dummy_x,))

    def test_attention_distillation_layer_not_found(self):
        self._check_layer_not_found(AttentionDistillation, True, False, "Teacher layer 'nonexistent_teacher_layer' not found")
        self._check_layer_not_found(AttentionDistillation, False, True, "Student layer 'nonexistent_student_layer' not found")

    def test_feature_distillation_layer_not_found(self):
        self._check_layer_not_found(FeatureDistillation, True, False, "Teacher layer 'nonexistent_teacher_layer' not found")
        self._check_layer_not_found(FeatureDistillation, False, True, "Student layer 'nonexistent_student_layer' not found")

    def test_feature_distillation_projection_dim_inference_works_for_lambda(self):
        # Test that projection dim can be inferred for a Lambda layer with a defined output shape
        student_input = keras.Input(shape=(10,), name="student_input_lambda")
        # Define the output dimension for the lambda layer explicitly
        lambda_output_dim = self.teacher_model.get_layer("teacher_intermediate_features").output_shape[-1] // 2
        intermediate_out = keras.layers.Lambda(
            lambda t: t[:, :lambda_output_dim],
            output_shape=(lambda_output_dim,), # Explicitly provide output_shape
            name="student_intermediate_lambda"
        )(student_input)
        output_logits = keras.layers.Dense(5, name="student_output_logits_lambda")(intermediate_out)
        student_model_lambda = keras.Model(inputs=student_input, outputs=output_logits, name="student_lambda")
        _ = student_model_lambda(self.dummy_x) # Build

        feature_strategy = FeatureDistillation(
            teacher_layer="teacher_intermediate_features",
            student_layer="student_intermediate_lambda",
            loss_fn=keras.losses.MeanSquaredError(), weight=1.0, projection=True,
        )
        distiller = Distiller(student=student_model_lambda, teacher=self.teacher_model, strategies=[feature_strategy])
        distiller.compile(optimizer=keras.optimizers.Adam())

        try:
            _ = distiller.train_step(data=(self.dummy_x,))
            self.assertIsNotNone(feature_strategy.projection_layer)
            # Expected units: Lambda output's last dimension
            expected_units = student_model_lambda.get_layer("student_intermediate_lambda").output_shape[-1]
            self.assertEqual(feature_strategy.projection_layer.units, expected_units)
        except ValueError as e:
            self.fail(f"FeatureDistillation with projection failed to infer dim for Lambda: {e}")

    def test_feature_distillation_projection_dim_inference_fail_mocked(self):
        # Mock a layer to have output_shape[-1] as None to trigger the error
        student_model_mock = self._create_simple_model(name="student_mock_shape")
        _ = student_model_mock(self.dummy_x) # Build

        original_get_layer = student_model_mock.get_layer

        class MockLayerWithNoneShapeDim(keras.layers.Layer):
            def __init__(self, original_layer, **kwargs):
                super().__init__(name=original_layer.name, dtype=original_layer.dtype)
                self._original_layer_instance = original_layer
                self._dynamic_output_shape = list(original_layer.output_shape)
                self._dynamic_output_shape[-1] = None
            @property
            def output_shape(self): return tuple(self._dynamic_output_shape)
            def call(self, inputs, *args, **kwargs): return self._original_layer_instance(inputs, *args, **kwargs)
            def build(self, input_shape):
                if hasattr(self._original_layer_instance, 'build') and not self._original_layer_instance.built:
                    self._original_layer_instance.build(input_shape)
                super().build(input_shape)

        def mock_get_layer(name):
            layer = original_get_layer(name)
            if name == "student_mock_shape_intermediate_features":
                return MockLayerWithNoneShapeDim(layer)
            return layer

        student_model_mock.get_layer = mock_get_layer

        feature_strategy_fail = FeatureDistillation(
            teacher_layer="teacher_intermediate_features",
            student_layer="student_mock_shape_intermediate_features",
            loss_fn=keras.losses.MeanSquaredError(), weight=1.0, projection=True,
        )
        distiller_fail = Distiller(student=student_model_mock, teacher=self.teacher_model, strategies=[feature_strategy_fail])
        distiller_fail.compile(optimizer=keras.optimizers.Adam())

        with self.assertRaisesRegex(ValueError, "Cannot automatically infer `student_output_dim` for projection"):
            distiller_fail.train_step(data=(self.dummy_x,))

        student_model_mock.get_layer = original_get_layer # Restore


if __name__ == "__main__":
    # If TestCase is a custom one (e.g., from keras_hub or tf.test) and has its own main(), prefer that.
    # Otherwise, fall back to standard unittest.main().
    is_standard_unittest_case = TestCase.__module__ == 'unittest.case' and TestCase.__name__ == 'TestCase'

    if hasattr(TestCase, 'main') and not is_standard_unittest_case:
        TestCase.main()
    else:
        import unittest
        unittest.main()
