# Knowledge Distillation with Keras Dynamic Loss Function Calling

This document shows how the Knowledge Distillation API leverages [Keras's dynamic loss function calling](https://keras.io/api/losses/) using strings, similar to how Keras models handle `loss="auto"` or `loss="sparse_categorical_crossentropy"`.

## 🎯 **Why Use Keras Dynamic Loss Function Calling?**

1. **✅ Flexibility**: Can pass loss function names as strings
2. **✅ Consistency**: Follows Keras conventions (like `model.compile(loss='sparse_categorical_crossentropy')`)
3. **✅ Simplicity**: No need to import specific loss classes
4. **✅ Maintainability**: Leverages Keras's maintained loss functions
5. **✅ Compatibility**: Works seamlessly with Keras's training ecosystem

## 🔧 **Keras Dynamic Loss Function Calling**

### **How It Works**

Instead of importing specific loss classes:
```python
# Old approach (still works)
from keras.losses import KLDivergence, MeanSquaredError
loss_fn = KLDivergence(reduction="sum_over_batch_size")
```

We use Keras's dynamic lookup:
```python
# New approach (more flexible)
loss_fn = keras.losses.get("kl_divergence")
loss_fn = keras.losses.get("mean_squared_error")
```

### **Available Loss Functions**

| **Loss Type** | **String Name** | **Usage** |
|---------------|-----------------|-----------|
| `kl_divergence` | `"kl_divergence"` | Standard KL divergence (default) |
| `mse` | `"mean_squared_error"` | MSE between logits/features |
| `cross_entropy` | `"categorical_crossentropy"` | Cross entropy |
| `cosine` | `"cosine_similarity"` | Cosine similarity for features |

## 📊 **Implementation Details**

### **LogitsDistillation with Dynamic Loss Calling**

```python
class LogitsDistillation(BaseDistillationStrategy):
    def __init__(self, temperature=2.0, loss_type="kl_divergence", custom_loss_fn=None):
        # Use Keras's dynamic loss function calling
        if loss_type == "kl_divergence":
            self.loss_fn = keras.losses.get("kl_divergence")
        elif loss_type == "mse":
            self.loss_fn = keras.losses.get("mean_squared_error")
        elif loss_type == "cross_entropy":
            self.loss_fn = keras.losses.get("categorical_crossentropy")
    
    def compute_loss(self, teacher_outputs, student_outputs):
        # Apply temperature scaling
        teacher_logits = teacher_outputs / self.temperature
        student_logits = student_outputs / self.temperature
        
        if self.loss_type == "kl_divergence":
            # Convert to probabilities for KL divergence
            teacher_probs = ops.softmax(teacher_logits, axis=-1)
            student_probs = ops.softmax(student_logits, axis=-1)
            
            # Use Keras KLDivergence dynamically
            loss = self.loss_fn(teacher_probs, student_probs)
            
        elif self.loss_type == "mse":
            # Use Keras MeanSquaredError directly on logits
            loss = self.loss_fn(teacher_logits, student_logits)
            
        elif self.loss_type == "cross_entropy":
            # Convert teacher to probabilities, keep student as logits
            teacher_probs = ops.softmax(teacher_logits, axis=-1)
            
            # Use Keras CategoricalCrossentropy dynamically
            loss = self.loss_fn(teacher_probs, student_logits)
        
        # Scale by temperature^2 for consistency with literature
        return loss * (self.temperature ** 2)
```

### **FeatureDistillation with Dynamic Loss Calling**

```python
class FeatureDistillation(BaseDistillationStrategy):
    def __init__(self, loss_type="mse", custom_loss_fn=None):
        # Use Keras's dynamic loss function calling
        if loss_type == "mse":
            self.loss_fn = keras.losses.get("mean_squared_error")
        elif loss_type == "cosine":
            self.loss_fn = keras.losses.get("cosine_similarity")
    
    def compute_loss(self, teacher_features, student_features):
        if self.loss_type == "mse":
            # Use Keras MeanSquaredError dynamically
            return self.loss_fn(teacher_features, student_features)
            
        elif self.loss_type == "cosine":
            # Use Keras CosineSimilarity dynamically (returns similarity, convert to distance)
            similarity = self.loss_fn(teacher_features, student_features)
            # Convert similarity to distance: distance = 1 - similarity
            return 1.0 - similarity
```

## 🎯 **Usage Examples**

### **Standard KL Divergence (Default)**
```python
# Uses keras.losses.get("kl_divergence") internally
strategy = LogitsDistillation(temperature=2.0, loss_type="kl_divergence")
```

### **MSE Alternative**
```python
# Uses keras.losses.get("mean_squared_error") internally
strategy = LogitsDistillation(temperature=2.0, loss_type="mse")
```

### **Cross Entropy**
```python
# Uses keras.losses.get("categorical_crossentropy") internally
strategy = LogitsDistillation(temperature=2.0, loss_type="cross_entropy")
```

### **Feature Distillation with MSE**
```python
# Uses keras.losses.get("mean_squared_error") for features
strategy = FeatureDistillation(loss_type="mse")
```

### **Feature Distillation with Cosine Similarity**
```python
# Uses keras.losses.get("cosine_similarity") for features
strategy = FeatureDistillation(loss_type="cosine")
```

### **Custom Loss Functions**
```python
def my_custom_loss(teacher_outputs, student_outputs):
    # Your custom logic here
    return custom_loss_value

strategy = LogitsDistillation(
    temperature=2.0, 
    loss_type="custom", 
    custom_loss_fn=my_custom_loss
)
```

## 🔍 **Key Benefits of Dynamic Loss Function Calling**

### **1. String-Based Configuration**
- Can pass loss function names as strings
- Easy to switch between different loss functions
- Consistent with Keras's `model.compile(loss='sparse_categorical_crossentropy')`

### **2. No Import Overhead**
- No need to import specific loss classes
- Works with any Keras loss function
- Reduces import complexity

### **3. Runtime Flexibility**
- Can change loss functions at runtime
- Easy to experiment with different loss functions
- Supports configuration-based loss selection

### **4. Keras Ecosystem Integration**
- Seamless integration with Keras's training ecosystem
- Consistent with Keras conventions
- Works with any Keras loss function

## 📈 **Comparison with Other Approaches**

| Approach | Pros | Cons |
|----------|------|------|
| **Dynamic String Calling** | ✅ Flexible, consistent, no imports | None |
| **Direct Class Import** | ✅ Explicit, type-safe | ❌ Import overhead, less flexible |
| **Custom Implementation** | ❌ Potential bugs, maintenance burden | ✅ Full control |

## 🎉 **Conclusion**

By using [Keras's dynamic loss function calling](https://keras.io/api/losses/), the Knowledge Distillation API:

1. **✅ Leverages proven implementations** from the Keras ecosystem
2. **✅ Maintains consistency** with Keras conventions
3. **✅ Provides flexibility** through string-based configuration
4. **✅ Reduces complexity** by eliminating import overhead
5. **✅ Ensures compatibility** with the broader Keras ecosystem

This approach gives users the best of both worlds: **standard, well-tested loss functions** with **maximum flexibility** through dynamic string-based calling! 