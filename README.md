# MAI - IDL 2026 — Final Project Assignment

Welcome to the official repository template for the **Introduction to Deep Learning (IDL) 2026 Final Assignment**.

### Overview

This repository contains the volatile, recovered remnants of a broken machine learning pipeline. Your mission is to audit the codebase, stabilize the system, optimize its computational footprint, and successfully deploy models across all target datasets.

* **Code:** All core source files can be found inside the `Code/` directory.
* **Instructions:** Background story and tasks are detailed in **`assignment_final.pdf`**.
* **Data:** Available for download here: https://cloud.fiw.fhws.de/s/LpYa2dCW85kwdNn

---

### Submission Guidelines

* **Platform:** Submit your final deliverables via the official **e-learning platform**.
* **Format:** Your submission must consist of a **direct link** to your created repository.
* **Branch:** Ensure all your final, production-ready code, your `AUDIT_LOG.md`, and your `REPORT.md` are completely merged into the **`main`** branch before the cutoff.
* **Deadline:** 09.07.2026, 23:59 (German Time). *Late submissions will not be processed.*

---

### Repository README

* **Professional Documentation:** Remember to update this `README` with a professional documentation of your repo.
* **Author(s):** Selma Saliaj, Nensi Mecalla

## Bug Fixes & Improvements

### Data Loading (`data.py`)

**File path construction**
- **Issue:** The loader was incorrectly appending `_data.pt` to dataset names, causing `FileNotFoundError`. For example, when trying to load `orgs`, the code was looking for `orgs_data.pt` instead of the actual file named `orgs.pt`. This happened because the original code had a hardcoded string that didn't match the actual naming convention of the data files.
- **Fix:** Changed `d_path = Path(data_path) / f"{data}_data.pt"` to `d_path = Path(data_path) / f"{data}.pt"`
- **Result:** The code now correctly loads `orgs.pt`, `chest.pt`, or whatever dataset name is passed to the function. This eliminates the file not found error that previously prevented training from even starting.

**Train/validation split**
- **Issue:** The validation data was being included in the training set, which is a classic data leakage problem. The original code used `train_data = data_dict['train_images']` without any slicing, meaning it took all training samples. Then it also took the last `val_size` samples as validation data. This resulted in the exact same samples appearing in both the training and validation sets - the model was effectively being evaluated on data it had already seen during training, giving falsely optimistic performance metrics.
- **Fix:** Added slicing to exclude the validation portion from training: `train_data = data_dict['train_images'][:val_start]` and `train_labels = data_dict['train_labels'][:val_start]`. Now the training set only includes samples from index 0 to `val_start-1`, while validation uses samples from `val_start` to the end.
- **Result:** Clean separation between training and validation sets. With 1000 samples and 10% validation, training now uses 900 samples and validation uses 100, with no overlap. This provides a realistic estimate of how the model will perform on unseen data, which is crucial for proper hyperparameter tuning and model selection.

---

### Training Loop (`fit.py`)

**Gradient accumulation**
- **Issue:** Gradients were accumulating across batches because PyTorch accumulates gradients by default. Without calling `zero_grad()`, each `loss.backward()` would add the new gradients to the existing ones from previous batches. This meant the optimizer would update weights based on the sum of gradients from multiple batches instead of just the current batch. Over time, this led to incorrect weight updates and training instability.
- **Fix:** Added `self.optimizer.zero_grad()` before `loss.backward()` in the training loop. This clears all gradient buffers so each batch starts with fresh gradients.
- **Result:** Each batch's gradients are calculated independently, ensuring proper optimization. The model now updates weights based on the current batch only, which is the intended behavior for SGD and its variants.

**Label shape mismatch**
- **Issue:** Labels were shaped as `(batch_size, 1)` (a column vector) but PyTorch's `CrossEntropyLoss` expects labels in the shape `(batch_size,)` (a 1D tensor). When labels had an extra dimension, the loss function would attempt to broadcast the tensors, leading to cryptic shape mismatch errors or incorrect loss computation.
- **Fix:** Added `labels = labels.flatten()` in both `train_one_epoch()` and `evaluate()`. This converts labels from `(32, 1)` to `(32,)`, removing the unnecessary extra dimension.
- **Result:** Labels are now correctly shaped for the loss function. This eliminates dimension-related crashes and ensures the loss is computed correctly for each sample in the batch.

---

### Model Architecture (`models.py`)

**Activation function**
- **Issue:** The global variable `activation_str = "Identity"` meant that all activation functions in the models were using Identity (which simply returns the input unchanged: f(x) = x). This is a critical problem because without non-linear activation functions, a deep neural network is mathematically equivalent to a single linear layer. No matter how many layers you stack, with Identity activation, the entire network reduces to a single linear transformation (W_n * ... * W_2 * W_1 * x = W_combined * x). This severely limited the network's ability to learn complex patterns, essentially making all deep models perform like linear classifiers.
- **Fix:** Changed `activation_str = "ReLU"`. ReLU (Rectified Linear Unit) introduces non-linearity by zeroing out negative values: f(x) = max(0, x). This allows the network to learn complex decision boundaries, capture non-linear relationships in the data, and is the most widely used activation function in modern CNNs.
- **Result:** Models now have the non-linearity they need to learn complex patterns. Training became effective and models started converging properly.

**ResNet18 forward pass**
- **Issue:** The `forward()` method in ResNet18 was missing a `return` statement at the end. The function would compute `self.classifier(out)` but then not return it. In Python, functions without a return statement implicitly return `None`. This meant that whenever you called the model forward, you would get `None` back instead of the actual predictions.
- **Fix:** Added `return self.classifier(out)` at the end of the forward function.
- **Result:** The forward pass now properly returns the model's predictions. This is essential for both training (computing loss) and inference (getting predictions).

**AlexNet flexibility**
- **Issue:** AlexNet was hardcoded to accept exactly 3 input channels (for RGB images) and output exactly 11 classes. This broke when trying to use it on datasets with different numbers of channels (like grayscale with 1 channel) or different numbers of classes (like 2 classes for binary classification). The hardcoded values meant the model would crash with shape mismatch errors when instantiated with different datasets.
- **Fix:** Added `in_channels=3` and `num_classes=11` as parameters to the `__init__` method. Updated the first Conv2d layer from `3` channels to `in_channels`, and the final Linear layer from `11` to `num_classes`.
- **Result:** The model now dynamically adapts to any dataset's specifications. You can use AlexNet on datasets with 1, 3, or any number of input channels, and with any number of output classes, making the architecture much more reusable.

**AlexNet feature dimensions**
- **Issue:** The classifier had `nn.Linear(2048, 1024)`, expecting 2048 flattened features from the convolutional layers. However, the actual output shape from the features was `(batch_size, 192, 4, 4)`, which flattens to `192 * 4 * 4 = 3072` dimensions. This discrepancy caused a dimension mismatch error when the model tried to pass features through the classifier - the 3072-dimensional tensor couldn't be fed into a layer expecting 2048 dimensions.
- **Fix:** Changed `nn.Linear(2048, 1024)` to `nn.Linear(3072, 1024)` to match the actual feature size.
- **Result:** The classifier now receives features of the correct size. The model no longer crashes due to dimension mismatches, and the forward pass completes successfully.

**VGGBlock padding**
- **Issue:** VGG's "C" configuration uses a 1x1 convolution at the tail end of its 3-conv blocks. The original code used the same padding value (1) for all convolutions. But for a 1x1 kernel, padding=1 is incorrect - it would add a border of zeros around the input, effectively increasing the spatial dimensions (e.g., going from 32x32 to 34x34). This caused the spatial dimensions to inflate incorrectly throughout the network, eventually leading to shape mismatches or the network outputting tensors of the wrong size.
- **Fix:** Added conditional padding: `pad = 0 if is_config_c_tail else padding`. Now 1x1 kernels use padding=0 (correct, since a 1x1 kernel doesn't need padding to preserve dimensions), while 3x3 kernels continue using padding=1.
- **Result:** Spatial dimensions are preserved correctly throughout the network. Each VGGBlock maintains the correct spatial size (except for the intentional reduction from MaxPooling), preventing dimension mismatches.

**VGGBlock channel tracking**
- **Issue:** In the loop that builds the VGGBlock layers, the variable `current_in_channels` was used to set the input channels for each Conv2d layer. However, after each convolution, `current_in_channels` was never updated. This meant that all convolutions in the block were trying to use the input channels of the first convolution. For example, if the block started with 64 input channels and had 3 convolutions, all three would expect 64 input channels, but the second and third convolutions should be expecting the output channels of the previous layer.
- **Fix:** Added `current_in_channels = out_channels` after each convolution layer. Now each layer receives the correct number of input channels from the previous layer's output.
- **Result:** The channel flow through the VGGBlock is now correct. Each convolution receives the proper number of input channels, ensuring the network architecture is built correctly.

---

### Training Script (`train.py`)

**Dropout rate**
- **Issue:** The dropout rate was set to 0.99 when instantiating the model. Dropout randomly deactivates a percentage of neurons during training to prevent overfitting. With p=0.99, it was deactivating 99% of neurons, meaning only 1% survived each forward pass. This is catastrophic for learning - almost all information flow through the network is blocked, and the few surviving neurons can't capture meaningful patterns. The model essentially couldn't learn anything because it was being forced to drop almost all information.
- **Fix:** Reduced dropout rate from 0.99 to 0.3. At 30% dropout, 70% of neurons survive each forward pass, which is a standard rate that provides regularization benefits without crippling learning. It's enough to prevent overfitting while still allowing the network to propagate meaningful information.
- **Result:** The model can now learn effectively. With balanced regularization, it achieves good performance on both training and validation sets without underfitting (from too much dropout) or overfitting (from too little dropout).

**Activation function**
- **Issue:** The model was being instantiated with `activation_str=None`. Since the models use `activation_str` to determine which activation function to apply, `None` resulted in using Identity activation (no activation at all). As explained in the models.py section, this made the entire network linear, regardless of depth.
- **Fix:** Added `activation_str="ReLU"` when instantiating the model. This ensures the model uses ReLU activation throughout, providing the necessary non-linearity.
- **Result:** The network now has the non-linearity needed to learn complex patterns. Combined with the dropout fix, this enables the model to properly converge during training.

---

## Training Test Results After Changes

### Test Configuration
| Parameter | Value |
|-----------|-------|
| **Dataset** | Organs (11 classes) |
| **Training Images** | 500 |
| **Test Images** | 200 |
| **Model** | AlexNet |
| **Batch Size** | 16 |
| **Learning Rate** | 0.0001 |
| **Epochs** | 10 |
| **Channels** | 1 (grayscale) |
| **Device** | CPU |

---

### Training Results

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|------------|-----------|----------|---------|
| 1 | 2.2695 | 21.56% | 2.2932 | 18.00% |
| 2 | 1.9777 | 28.22% | 1.8494 | 30.00% |
| 3 | 1.7981 | 32.89% | 1.8183 | 38.00% |
| 4 | 1.6107 | 40.44% | 1.5139 | 50.00% |
| 5 | 1.4991 | 41.56% | 1.5270 | 46.00% |
| 6 | 1.3460 | 48.89% | 1.2567 | 52.00% |
| 7 | 1.2266 | 53.11% | 1.1966 | 60.00% |
| 8 | 1.1208 | 54.67% | 1.0686 | 60.00% |
| **9** | **0.9235** | **63.56%** | **1.1746** | **66.00%** ⭐ |
| 10 | 0.9261 | 65.33% | 1.5059 | 48.00% |

---

### Summary

**Best Performance:** Epoch 9
- **Validation Accuracy:** 66.00%
- **Training Accuracy:** 63.56%
- **Training Loss:** 0.9235

**Key Observations:**
- Model started at 21.56% training accuracy (well above random guessing of ~9% for 11 classes)
- Consistent improvement through epoch 9
- Peak validation accuracy of 66% achieved at epoch 9
- Overfitting observed at epoch 10 (validation accuracy dropped from 66% to 48%)

**Analysis:**
The model successfully learned meaningful patterns from only 500 training images. The 66% validation accuracy demonstrates that the training pipeline works correctly and the model architecture (AlexNet) is suitable for this classification task. The drop in validation accuracy at epoch 10 indicates that training should be stopped earlier (at epoch 9) to prevent overfitting.

Based on this test run we will reduce epochs to 8-9 for future runs.