# CIFAR-10 Image Classification — Comparing Two CNN Architectures

Two convolutional neural networks trained and compared on the CIFAR-10 dataset (60,000 32x32 color images across 10 classes).

## Models

| | Model 1 | Model 2 |
|---|---|---|
| Depth | 3 conv blocks (32→64→128 filters) | 2 conv blocks (64→128 filters, wider) |
| Data augmentation | Yes (random flip, rotation, zoom) | No |
| Regularization | BatchNorm + Dropout (0.25–0.5) | BatchNorm + Dropout (0.3–0.5) |
| Optimizer | Adam (lr=0.001) | Adam (lr=0.001) |
| Parameters | 1.35M | 4.46M |
| **Test accuracy** | **86.72%** | **86.50%** |

Both models are trained with early stopping, learning-rate reduction on plateau, and checkpointing of the best weights (by validation accuracy).

## What the script does

1. Loads and normalizes CIFAR-10
2. Builds and trains both CNN architectures
3. Evaluates each on the test set (accuracy, confusion matrix, per-class precision/recall/F1)
4. Plots training curves and saves confusion matrix heatmaps
5. Compares the augmented deep model against the simpler wide model

## Run it

```bash
pip install -r requirements.txt
python cifar10_cnn.py
```

## Tech stack

TensorFlow / Keras, scikit-learn, Seaborn, Matplotlib

## Notes

The main question this project explores: does a deeper network with data augmentation generalize better than a shallower, wider network trained without augmentation? Interestingly, the two ended up close (86.72% vs. 86.50%) despite Model 2 having over 3x the parameters — suggesting augmentation and depth compensated for parameter count here. The confusion matrices and classification reports make it possible to see not just overall accuracy but which classes (e.g. cat vs. dog) are hardest to separate.
