#!/usr/bin/env python3
"""
Customization by Subclassing Example
====================================

This example demonstrates how users can customize distillation strategies
by subclassing and overriding the compute_loss method.

Example usage:
    python customization_by_subclassing_example.py
"""

import keras
import numpy as np
from keras import ops

# Import our distillation classes
from keras_hub.src.distillation.distiller import Distiller
from keras_hub.src.distillation.strategies import FeatureDistillation
from keras_hub.src.distillation.strategies import LogitsDistillation


def create_simple_models():
    """Create simple teacher and student models for demonstration."""
    # Teacher model (larger)
    teacher = keras.Sequential(
        [
            keras.layers.Dense(128, activation="relu", input_shape=(10,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(2, activation="softmax"),
        ]
    )

    # Student model (smaller)
    student = keras.Sequential(
        [
            keras.layers.Dense(32, activation="relu", input_shape=(10,)),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(2, activation="softmax"),
        ]
    )

    return teacher, student


def demonstrate_custom_logits_distillation():
    """Demonstrate custom logits distillation by subclassing."""
    print("\n🔍 1. Custom Logits Distillation by Subclassing")
    print("=" * 60)

    class CustomLogitsDistillation(LogitsDistillation):
        """Custom logits distillation with L1 distance."""

        def __init__(self, temperature=2.0):
            super().__init__(
                temperature=temperature, loss_type="mse"
            )  # Base type doesn't matter
            self.temperature = temperature

        def compute_loss(self, teacher_outputs, student_outputs):
            """Custom L1 distance distillation loss."""
            # Apply temperature scaling
            teacher_logits = teacher_outputs / self.temperature
            student_logits = student_outputs / self.temperature

            # Convert to probabilities
            teacher_probs = ops.softmax(teacher_logits, axis=-1)
            student_probs = ops.softmax(student_logits, axis=-1)

            # Custom L1 distance loss
            l1_distance = keras.ops.mean(
                keras.ops.abs(teacher_probs - student_probs)
            )

            # Scale by temperature^2 for consistency
            return l1_distance * (self.temperature**2)

    # Create models
    teacher, student = create_simple_models()
    teacher.trainable = False

    # Use custom strategy
    custom_strategy = CustomLogitsDistillation(temperature=2.0)
    distiller = Distiller(
        teacher=teacher,
        student=student,
        strategies=[custom_strategy],
        alpha=0.5,
    )

    print("✅ Created custom L1 distance distillation strategy")
    print(f"   Temperature: {custom_strategy.temperature}")
    print("   Custom loss: L1 distance between probabilities")

    # Test with dummy data
    x = np.random.random((5, 10))
    y = np.random.randint(0, 2, size=(5,))

    metrics = distiller.train_step([x, y])
    print(f"   📊 Sample loss: {metrics['total_loss']:.4f}")

    return distiller


def demonstrate_custom_feature_distillation():
    """Demonstrate custom feature distillation by subclassing."""
    print("\n🔍 2. Custom Feature Distillation by Subclassing")
    print("=" * 60)

    class CustomFeatureDistillation(FeatureDistillation):
        """Custom feature distillation with L2 distance."""

        def __init__(self):
            super().__init__(loss_type="mse")  # Base type doesn't matter

        def compute_loss(self, teacher_features, student_features):
            """Custom L2 distance feature distillation loss."""
            # L2 distance between features
            l2_distance = keras.ops.mean(
                keras.ops.square(teacher_features - student_features)
            )
            return l2_distance

    # Create models
    teacher, student = create_simple_models()
    teacher.trainable = False

    # Use custom strategy
    custom_strategy = CustomFeatureDistillation()
    distiller = Distiller(
        teacher=teacher,
        student=student,
        strategies=[custom_strategy],
        alpha=0.5,
    )

    print("✅ Created custom L2 distance feature distillation strategy")
    print("   Custom loss: L2 distance between features")

    # Test with dummy data
    x = np.random.random((5, 10))
    y = np.random.randint(0, 2, size=(5,))

    metrics = distiller.train_step([x, y])
    print(f"   📊 Sample loss: {metrics['total_loss']:.4f}")

    return distiller


def demonstrate_multiple_custom_strategies():
    """Demonstrate multiple custom strategies."""
    print("\n🔍 3. Multiple Custom Strategies")
    print("=" * 60)

    class KLDivergenceStrategy(LogitsDistillation):
        """KL divergence strategy with custom temperature scaling."""

        def compute_loss(self, teacher_outputs, student_outputs):
            """KL divergence with custom temperature scaling."""
            # Apply temperature scaling
            teacher_logits = teacher_outputs / self.temperature
            student_logits = student_outputs / self.temperature

            # Convert to probabilities
            teacher_probs = ops.softmax(teacher_logits, axis=-1)
            student_probs = ops.softmax(student_logits, axis=-1)

            # KL divergence
            loss = keras.losses.kl_divergence(teacher_probs, student_probs)

            # Custom scaling (not temperature^2)
            return loss * self.temperature

    class MSEDistanceStrategy(FeatureDistillation):
        """MSE distance strategy for features."""

        def compute_loss(self, teacher_features, student_features):
            """MSE distance between features."""
            return keras.losses.mean_squared_error(
                teacher_features, student_features
            )

    # Create models
    teacher, student = create_simple_models()
    teacher.trainable = False

    # Use multiple custom strategies
    strategies = [KLDivergenceStrategy(temperature=2.0), MSEDistanceStrategy()]

    distiller = Distiller(
        teacher=teacher, student=student, strategies=strategies, alpha=0.5
    )

    print("✅ Created multiple custom strategies")
    print("   Strategy 1: Custom KL divergence with T scaling")
    print("   Strategy 2: Custom MSE distance for features")

    # Test with dummy data
    x = np.random.random((5, 10))
    y = np.random.randint(0, 2, size=(5,))

    metrics = distiller.train_step([x, y])
    print(f"   📊 Sample loss: {metrics['total_loss']:.4f}")

    return distiller


def demonstrate_advanced_customization():
    """Demonstrate advanced customization patterns."""
    print("\n🔍 4. Advanced Customization Patterns")
    print("=" * 60)

    class AdaptiveTemperatureStrategy(LogitsDistillation):
        """Strategy with adaptive temperature based on loss magnitude."""

        def __init__(self, base_temperature=2.0, adaptive_factor=0.1):
            super().__init__(
                temperature=base_temperature, loss_type="kl_divergence"
            )
            self.base_temperature = base_temperature
            self.adaptive_factor = adaptive_factor
            self.step_count = 0

        def compute_loss(self, teacher_outputs, student_outputs):
            """KL divergence with adaptive temperature."""
            # Adaptive temperature based on step count
            adaptive_temp = (
                self.base_temperature + self.adaptive_factor * self.step_count
            )

            # Apply adaptive temperature scaling
            teacher_logits = teacher_outputs / adaptive_temp
            student_logits = student_outputs / adaptive_temp

            # Convert to probabilities
            teacher_probs = ops.softmax(teacher_logits, axis=-1)
            student_probs = ops.softmax(student_logits, axis=-1)

            # KL divergence
            loss = keras.losses.kl_divergence(teacher_probs, student_probs)

            # Update step count
            self.step_count += 1

            return loss * (adaptive_temp**2)

    # Create models
    teacher, student = create_simple_models()
    teacher.trainable = False

    # Use adaptive strategy
    adaptive_strategy = AdaptiveTemperatureStrategy(
        base_temperature=2.0, adaptive_factor=0.1
    )
    distiller = Distiller(
        teacher=teacher,
        student=student,
        strategies=[adaptive_strategy],
        alpha=0.5,
    )

    print("✅ Created adaptive temperature strategy")
    print(f"   Base temperature: {adaptive_strategy.base_temperature}")
    print(f"   Adaptive factor: {adaptive_strategy.adaptive_factor}")
    print("   Temperature increases with training steps")

    # Test multiple steps
    x = np.random.random((5, 10))
    y = np.random.randint(0, 2, size=(5,))

    for step in range(3):
        metrics = distiller.train_step([x, y])
        temp = (
            adaptive_strategy.base_temperature
            + adaptive_strategy.adaptive_factor * step
        )
        print(
            f"   Step {step + 1}: loss = {metrics['total_loss']:.4f}, "
            f"temp = {temp:.2f}"
        )

    return distiller


def main():
    """Run all customization examples."""
    print("🚀 Customization by Subclassing Examples")
    print("=" * 60)
    print("This demonstrates how users can customize distillation strategies")
    print("by subclassing and overriding the compute_loss method.")
    print()

    # Run examples
    demonstrate_custom_logits_distillation()
    demonstrate_custom_feature_distillation()
    demonstrate_multiple_custom_strategies()
    demonstrate_advanced_customization()

    print("\n" + "=" * 60)
    print("📋 SUMMARY OF CUSTOMIZATION PATTERNS:")
    print("=" * 60)
    print("✅ Subclass LogitsDistillation for custom logits distillation")
    print("✅ Subclass FeatureDistillation for custom feature distillation")
    print("✅ Override compute_loss method for full control")
    print("✅ Combine multiple custom strategies")
    print("✅ Implement advanced patterns (adaptive temperature, etc.)")
    print()
    print("🎯 Key Benefits:")
    print("   • Clean OOP design")
    print("   • Full control over loss computation")
    print("   • Easy to extend and maintain")
    print("   • Consistent with Keras patterns")
    print()
    print("🔧 Usage Pattern:")
    print("   class MyCustomStrategy(LogitsDistillation):")
    print("       def compute_loss(self, teacher_outputs, student_outputs):")
    print("           # Your custom logic here")
    print("           return custom_loss")
    print()
    print("🎉 All customization patterns are working!")


if __name__ == "__main__":
    main()
