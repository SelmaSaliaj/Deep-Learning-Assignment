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