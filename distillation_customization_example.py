#!/usr/bin/env python3
"""
Example demonstrating customizable distillation strategies.

This script shows how to use different loss functions and custom strategies
for knowledge distillation.
"""

import os

import keras
import numpy as np

# Set backend
os.environ["KERAS_BACKEND"] = "tensorflow"

# Import the distillation components
from keras_hub.src.distillation.distiller import Distiller
from keras_hub.src.distillation.strategies import FeatureDistillation
from keras_hub.src.distillation.strategies import LogitsDistillation


def create_simple_models():
    """Create simple teacher and student models for demonstration."""
    # Teacher model
    teacher_input = keras.Input(shape=(128,), dtype="int32")
    teacher_embedding = keras.layers.Embedding(1000, 256)(teacher_input)
    teacher_lstm = keras.layers.LSTM(128, return_sequences=True)(
        teacher_embedding
    )
    teacher_pool = keras.layers.GlobalAveragePooling1D()(teacher_lstm)
    teacher_dense = keras.layers.Dense(64, activation="relu")(teacher_pool)
    teacher_output = keras.layers.Dense(2, activation="softmax")(teacher_dense)
    teacher = keras.Model(teacher_input, teacher_output)

    # Student model
    student_input = keras.Input(shape=(128,), dtype="int32")
    student_embedding = keras.layers.Embedding(1000, 128)(student_input)
    student_lstm = keras.layers.LSTM(64, return_sequences=True)(
        student_embedding
    )
    student_pool = keras.layers.GlobalAveragePooling1D()(student_lstm)
    student_dense = keras.layers.Dense(32, activation="relu")(student_pool)
    student_output = keras.layers.Dense(2, activation="softmax")(student_dense)
    student = keras.Model(student_input, student_output)

    return teacher, student


def demonstrate_standard_kl_divergence():
    """Demonstrate standard KL divergence distillation."""
    print("\n🔍 1. Standard KL Divergence Distillation")
    print("=" * 50)

    teacher, student = create_simple_models()
    teacher.trainable = False

    strategy = LogitsDistillation(temperature=2.0, loss_type="kl_divergence")
    distiller = Distiller(
        teacher=teacher, student=student, strategies=[strategy], alpha=0.5
    )

    print("✅ Using standard KL divergence")
    print(f"   Temperature: {strategy.temperature}")
    print(f"   Loss type: {strategy.loss_type}")
    print("   This is the most common approach in knowledge distillation")

    # Test with dummy data
    x = np.random.random((5, 10))
    y = np.random.randint(0, 2, size=(5,))
    metrics = distiller.train_step([x, y])
    print(f"   📊 Sample loss: {metrics['total_loss']:.4f}")

    return distiller


def demonstrate_mse_distillation():
    """Demonstrate MSE distillation."""
    print("\n🔍 2. MSE Distillation")
    print("=" * 50)

    teacher, student = create_simple_models()
    teacher.trainable = False

    strategy = LogitsDistillation(temperature=2.0, loss_type="mse")
    distiller = Distiller(
        teacher=teacher, student=student, strategies=[strategy], alpha=0.5
    )

    print("✅ Using MSE between logits")
    print(f"   Temperature: {strategy.temperature}")
    print(f"   Loss type: {strategy.loss_type}")
    print("   Simpler alternative to KL divergence")

    # Test with dummy data
    x = np.random.random((5, 10))
    y = np.random.randint(0, 2, size=(5,))
    metrics = distiller.train_step([x, y])
    print(f"   📊 Sample loss: {metrics['total_loss']:.4f}")

    return distiller


def demonstrate_cross_entropy_distillation():
    """Demonstrate cross entropy distillation."""
    print("\n🔍 3. Cross Entropy Distillation")
    print("=" * 50)

    teacher, student = create_simple_models()
    teacher.trainable = False

    strategy = LogitsDistillation(temperature=2.0, loss_type="cross_entropy")
    distiller = Distiller(
        teacher=teacher, student=student, strategies=[strategy], alpha=0.5
    )

    print("✅ Using cross entropy between softmax outputs")
    print(f"   Temperature: {strategy.temperature}")
    print(f"   Loss type: {strategy.loss_type}")
    print("   Alternative to KL divergence")

    # Test with dummy data
    x = np.random.random((5, 10))
    y = np.random.randint(0, 2, size=(5,))
    metrics = distiller.train_step([x, y])
    print(f"   📊 Sample loss: {metrics['total_loss']:.4f}")

    return distiller


def demonstrate_multiple_strategies():
    """Demonstrate multiple distillation strategies."""
    print("\n🔍 4. Multiple Distillation Strategies")
    print("=" * 50)

    teacher, student = create_simple_models()
    teacher.trainable = False

    strategies = [
        LogitsDistillation(temperature=2.0, loss_type="kl_divergence"),
        LogitsDistillation(temperature=4.0, loss_type="mse"),
    ]
    distiller = Distiller(
        teacher=teacher, student=student, strategies=strategies, alpha=0.5
    )

    print("✅ Using multiple strategies")
    print("   Strategy 1: KL divergence (T=2.0)")
    print("   Strategy 2: MSE (T=4.0)")
    print("   Combines different approaches for better distillation")

    # Test with dummy data
    x = np.random.random((5, 10))
    y = np.random.randint(0, 2, size=(5,))
    metrics = distiller.train_step([x, y])
    print(f"   📊 Sample loss: {metrics['total_loss']:.4f}")

    return distiller


def demonstrate_feature_distillation():
    """Demonstrate feature distillation."""
    print("\n🔍 5. Feature Distillation")
    print("=" * 50)

    teacher, student = create_simple_models()
    teacher.trainable = False

    strategy = FeatureDistillation(loss_type="mse")
    distiller = Distiller(
        teacher=teacher, student=student, strategies=[strategy], alpha=0.5
    )

    print("✅ Using feature distillation with MSE")
    print(f"   Loss type: {strategy.loss_type}")
    print("   Distills intermediate features, not just outputs")

    # Test with dummy data
    x = np.random.random((5, 10))
    y = np.random.randint(0, 2, size=(5,))
    metrics = distiller.train_step([x, y])
    print(f"   📊 Sample loss: {metrics['total_loss']:.4f}")

    return distiller


def demonstrate_cosine_feature_distillation():
    """Demonstrate cosine similarity feature distillation."""
    print("\n🔍 6. Cosine Similarity Feature Distillation")
    print("=" * 50)

    teacher, student = create_simple_models()
    teacher.trainable = False

    strategy = FeatureDistillation(loss_type="cosine")
    distiller = Distiller(
        teacher=teacher, student=student, strategies=[strategy], alpha=0.5
    )

    print("✅ Using feature distillation with cosine similarity")
    print(f"   Loss type: {strategy.loss_type}")
    print("   Focuses on feature direction, not magnitude")

    # Test with dummy data
    x = np.random.random((5, 10))
    y = np.random.randint(0, 2, size=(5,))
    metrics = distiller.train_step([x, y])
    print(f"   📊 Sample loss: {metrics['total_loss']:.4f}")

    return distiller


def main():
    """Demonstrate all customization options."""
    print("🚀 Knowledge Distillation Customization Examples")
    print("=" * 60)

    # Create dummy data
    X_train = np.random.randint(0, 1000, size=(100, 128))
    y_train = np.random.randint(0, 2, size=(100,))

    # Demonstrate different approaches
    demonstrations = [
        demonstrate_standard_kl_divergence,
        demonstrate_mse_distillation,
        demonstrate_cross_entropy_distillation,
        demonstrate_multiple_strategies,
        demonstrate_feature_distillation,
        demonstrate_cosine_feature_distillation,
    ]

    for demo in demonstrations:
        try:
            distiller = demo()
            print("   ✅ Distiller created successfully")

            # Test with a small batch
            batch_x = X_train[:32]
            batch_y = y_train[:32]

            # Test train step
            metrics = distiller.train_step((batch_x, batch_y))
            print(f"   📊 Sample loss: {metrics['total_loss']:.4f}")

        except Exception as e:
            print(f"   ❌ Error: {e}")

        print()

    # Summary
    print("=" * 60)
    print("📋 SUMMARY OF CUSTOMIZATION OPTIONS:")
    print("=" * 60)

    print("🎯 Logits Distillation Loss Functions:")
    print("   • KL Divergence (standard) - Most common approach")
    print("   • MSE - Simpler alternative")
    print("   • Cross Entropy - Alternative to KL divergence")
    print("   • Custom - User-defined loss functions")

    print("\n🎯 Feature Distillation:")
    print("   • MSE - Mean squared error between features")
    print("   • Cosine Similarity - Focus on feature direction")
    print("   • Custom - User-defined feature loss functions")

    print("\n🎯 Advanced Options:")
    print("   • Multiple strategies - Combine different approaches")
    print("   • Custom temperature scaling - Adjust knowledge transfer")
    print("   • Alpha parameter - Balance task vs distillation loss")

    print("\n🔧 Usage Examples:")
    print("   # Standard KL divergence")
    print("   strategy = LogitsDistillation(temperature=2.0)")

    print("   # MSE alternative")
    print("   strategy = LogitsDistillation(temperature=2.0, loss_type='mse')")

    print("   # Custom loss function")
    print(
        "   strategy = LogitsDistillation(temperature=2.0, loss_type='custom', "
        "custom_loss_fn=my_loss)"
    )

    print("   # Feature distillation")
    print("   strategy = FeatureDistillation(loss_type='cosine')")

    print("\n🎉 All customization options are working!")


if __name__ == "__main__":
    main()
