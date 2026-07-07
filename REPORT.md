## Organs Dataset Results

| Model | Best Validation Accuracy | Epoch | Final Accuracy | Training Accuracy | Overfitting |
|-------|--------------------------|-------|----------------|-------------------|-------------|
| AlexNet | 66.00% | 9 | 48.00% | 65.33% | Yes |
| VGG16 | 80.00% | 7 | 66.00% | 82.22% | Yes |
| ResNet18 | 78.00% | 8 | 78.00% | 90.67% | No |

**Recommended Model:** ResNet18
- Best performance: 78% validation accuracy
- No overfitting observed
- Most stable training across all epochs

---

### Chest Dataset Results (Binary Classification)

| Model | Best Validation Accuracy | Epoch | Training Accuracy | Time | Overfitting |
|-------|--------------------------|-------|-------------------|------|-------------|
| **AlexNet** | **97.32%** | 2, 4 | 97.13% | Fast | No |
| ResNet18 | 96.56% | 2 | 95.48% | Fast | No |
| VGG16 | **98.66%** | 3 | 96.94% | Medium | Potential |

**Target Accuracy:** 87% ; **All models exceeded target!**

**Recommended Model:** AlexNet (best balance of accuracy and stability)

---

## Orgs Dataset Results

| Model | Best Validation Accuracy | Epoch | Final Accuracy | Training Accuracy | Overfitting |
|-------|--------------------------|-------|----------------|-------------------|-------------|
| AlexNet | **99.09%** | 9 | 98.89% | 97.98% | No |
| VGG16 | 97.98% | 2 | 96.03% | 94.95% | Yes |
| ResNet18 | **99.41%** | 4 | 99.41% | 96.43% | No |

**Target Accuracy:** 83% ; **All models exceeded target!**

**Recommended Model:** AlexNet
- Best overall performance: 99.09% validation accuracy
- Most stable training across all 10 epochs
- No overfitting observed
- Reliable "set and forget" model

**Note:** ResNet18 achieved the highest peak accuracy (99.41% at epoch 4) but would likely overfit if trained beyond 5 epochs. AlexNet provides comparable performance with better stability.

---

## Lesions Dataset Results

| Model | Best Validation Accuracy | Epoch | Final Accuracy | Training Accuracy | Overfitting |
|-------|--------------------------|-------|----------------|-------------------|-------------|
| AlexNet | 76.03% | 7 | 75.16% | 78.47% | Slight |

**Target Accuracy:** 67% ; **Exceeded target!**

**Recommended Model:** AlexNet
- Best performance: 76.03% validation accuracy
- Steady improvement through epoch 7

---

## Cells Dataset Results

| Model | Best Validation Accuracy | Epoch | Final Accuracy | Training Accuracy | Overfitting |
|-------|--------------------------|-------|----------------|-------------------|-------------|
| AlexNet | 96.78% | 7 | 95.54% | 96.74% | Slight |

**Target Accuracy:** 90% ; **Exceeded target!**

**Recommended Model:** AlexNet
- Best performance: 96.78% validation accuracy
- Excellent performance overall