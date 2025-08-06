#!/usr/bin/env python3
"""Debug script for distillation loss computation."""

# Add current directory to path
import sys

import keras
import numpy as np
from keras import ops

sys.path.append(".")

from keras_hub.src.distillation.strategies import LogitsDistillation


def debug_logits_distillation():
    """Debug the LogitsDistillation loss computation."""
    print("Debugging LogitsDistillation...")

    strategy = LogitsDistillation(temperature=2.0, weight=1.0)

    # Create dummy logits
    teacher_logits = np.array([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
    student_logits = np.array([[0.8, 1.8, 2.8], [0.3, 1.3, 2.3]])

    print(f"Teacher logits: {teacher_logits}")
    print(f"Student logits: {student_logits}")

    # Convert to backend tensors
    teacher_logits = keras.ops.convert_to_tensor(
        teacher_logits, dtype="float32"
    )
    student_logits = keras.ops.convert_to_tensor(
        student_logits, dtype="float32"
    )

    # Apply temperature scaling
    temperature = 2.0
    teacher_logits_soft = teacher_logits / temperature
    student_logits_soft = student_logits / temperature

    print(f"Teacher logits (soft): {teacher_logits_soft}")
    print(f"Student logits (soft): {student_logits_soft}")

    # Compute softmax probabilities
    teacher_probs = ops.softmax(teacher_logits_soft, axis=-1)
    student_probs = ops.softmax(student_logits_soft, axis=-1)

    print(f"Teacher probs: {teacher_probs}")
    print(f"Student probs: {student_probs}")

    # Check if probabilities sum to 1
    print(f"Teacher probs sum: {ops.sum(teacher_probs, axis=-1)}")
    print(f"Student probs sum: {ops.sum(student_probs, axis=-1)}")

    # Apply clipping
    eps = 1e-8
    teacher_probs = ops.clip(teacher_probs, eps, 1.0 - eps)
    student_probs = ops.clip(student_probs, eps, 1.0 - eps)

    print(f"Teacher probs (clipped): {teacher_probs}")
    print(f"Student probs (clipped): {student_probs}")

    # Compute KL divergence manually
    kl_div = ops.sum(
        teacher_probs * ops.log(teacher_probs / student_probs), axis=-1
    )
    print(f"KL divergence per sample: {kl_div}")

    loss = ops.mean(kl_div)
    print(f"Mean KL divergence: {loss}")

    # Scale loss by temperature squared
    scaled_loss = loss * (temperature**2)
    print(f"Scaled loss: {scaled_loss}")

    # Apply weight
    final_loss = 1.0 * scaled_loss
    print(f"Final loss: {final_loss}")

    # Try with different logits
    print("\nTrying with more different logits...")
    teacher_logits2 = np.array([[10.0, 1.0, 1.0]])
    student_logits2 = np.array([[8.0, 2.0, 2.0]])

    teacher_logits2 = keras.ops.convert_to_tensor(
        teacher_logits2, dtype="float32"
    )
    student_logits2 = keras.ops.convert_to_tensor(
        student_logits2, dtype="float32"
    )

    loss2 = strategy.compute_loss(teacher_logits2, student_logits2)
    print(f"Loss with different logits: {loss2}")


if __name__ == "__main__":
    debug_logits_distillation()
