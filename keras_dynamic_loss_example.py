#!/usr/bin/env python3
"""
Keras Dynamic Loss Function Calling Example
==========================================

This example demonstrates how Keras allows you to call any loss function
using a string, similar to how we use it in our distillation strategies.

Example usage:
    python keras_dynamic_loss_example.py
"""

import keras
import numpy as np
from keras import ops


def demonstrate_keras_dynamic_loss_calling():
    """Demonstrate Keras's dynamic loss function calling with strings."""
    print("🚀 Keras Dynamic Loss Function Calling Example")
    print("=" * 60)

    # Create some dummy data
    y_true = np.array([[0.2, 0.8], [0.9, 0.1], [0.3, 0.7]])
    y_pred = np.array([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]])

    print("📊 Sample data:")
    print(f"   y_true shape: {y_true.shape}")
    print(f"   y_pred shape: {y_pred.shape}")
    print()

    # Demonstrate different ways to get loss functions

    print("🔧 Method 1: Direct class instantiation")
    print("-" * 40)
    kl_loss_class = keras.losses.KLDivergence()
    mse_loss_class = keras.losses.MeanSquaredError()
    print(f"   KL Divergence class: {kl_loss_class}")
    print(f"   MSE class: {mse_loss_class}")

    print("\n🔧 Method 2: Using keras.losses.get() with strings")
    print("-" * 40)
    kl_loss_string = keras.losses.get("kl_divergence")
    mse_loss_string = keras.losses.get("mean_squared_error")
    print(f"   KL Divergence from string: {kl_loss_string}")
    print(f"   MSE from string: {mse_loss_string}")

    print("\n🔧 Method 3: Direct function calls")
    print("-" * 40)
    kl_loss_func = keras.losses.kl_divergence
    mse_loss_func = keras.losses.mean_squared_error
    print(f"   KL Divergence function: {kl_loss_func}")
    print(f"   MSE function: {mse_loss_func}")

    # Test that they all work the same
    print("\n🧪 Testing that all methods work:")
    print("-" * 40)

    # Method 1: Class instantiation (returns scalar)
    result1 = kl_loss_class(y_true, y_pred)
    print(f"   Class method KL loss: {result1}")

    # Method 2: String lookup (returns per-sample array)
    result2 = kl_loss_string(y_true, y_pred)
    print(f"   String method KL loss: {result2}")

    # Method 3: Direct function (returns per-sample array)
    result3 = kl_loss_func(y_true, y_pred)
    print(f"   Function method KL loss: {result3}")

    print(
        "\n✅ All methods work! Class method returns scalar, "
        "others return per-sample arrays"
    )
    print(
        f"   Class method shape: "
        f"{result1.shape if hasattr(result1, 'shape') else 'scalar'}"
    )
    print(f"   String method shape: {result2.shape}")
    print(f"   Function method shape: {result3.shape}")

    # Demonstrate with our distillation context
    print("\n🎯 Application in Knowledge Distillation:")
    print("-" * 40)

    # Simulate teacher and student logits
    teacher_logits = np.array([[1.0, 2.0], [3.0, 1.0], [2.0, 3.0]])
    student_logits = np.array([[1.5, 1.8], [2.8, 1.2], [2.2, 2.8]])

    print(f"   Teacher logits: {teacher_logits}")
    print(f"   Student logits: {student_logits}")

    # Apply temperature scaling (like in distillation)
    temperature = 2.0
    teacher_scaled = teacher_logits / temperature
    student_scaled = student_logits / temperature

    # Convert to probabilities for KL divergence
    teacher_probs = ops.softmax(teacher_scaled, axis=-1)
    student_probs = ops.softmax(student_scaled, axis=-1)

    print(f"   Teacher probs: {teacher_probs}")
    print(f"   Student probs: {student_probs}")

    # Use dynamic loss function calling
    kl_loss_fn = keras.losses.get("kl_divergence")
    distillation_loss = kl_loss_fn(teacher_probs, student_probs)

    print(f"   Distillation loss: {distillation_loss}")
    print(f"   Scaled loss (T²): {distillation_loss * (temperature**2)}")

    print("\n🎉 Key Benefits of Dynamic Loss Function Calling:")
    print("-" * 40)
    print("✅ Can pass loss function names as strings")
    print("✅ Easy to switch between different loss functions")
    print(
        "✅ Consistent with Keras's "
        "model.compile(loss='sparse_categorical_crossentropy')"
    )
    print("✅ Works with any Keras loss function")
    print("✅ No need to import specific loss classes")

    return True


if __name__ == "__main__":
    demonstrate_keras_dynamic_loss_calling()
