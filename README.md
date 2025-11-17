# COSC 4368 – CNN Design Challenge (Tiny ImageNet Subset)

This repo contains my solution for the COSC 4368 CNN Design Challenge: a custom sequential CNN in PyTorch trained on a subset of Tiny ImageNet.

- **Images:** 64×64, RGB
- **Classes:** 200 logits (to match Tiny ImageNet label indices)
- **Splits:** 
  - 70% train (`train-70_.pkl`)
  - 10% validation (`validation-10_.pkl`)
  - 20% hidden test (held by instructor)

The instructor will import the model, load `model.pth`, and call `predict(...)` on their hidden test loader.

---

## 1. Environment Setup

### Python version

Tested with **Python 3.11** (3.10+ should also work) on Windows with a CUDA GPU (NVIDIA 2080 Super).

### Create and activate a virtual environment

```bash
# From the repo root
python -m venv .venv

# Activate (PowerShell on Windows)
.venv\Scripts\Activate.ps1

# (On CMD)
.venv\Scripts\activate.bat

# (On Linux/macOS)
source .venv/bin/activate

#Design Choice
```
Architecture. I used a straightforward 4-block CNN designed for 64×64 RGB images. Each block has two 3×3 convolutions followed by batch norm and ReLU, then a 2×2 max-pool and dropout. Channels go 32 → 64 → 128 → 256, so after four pools the spatial size is 4×4 and I flatten 256×4×4 features into a 512-unit fully connected layer and then a final linear layer over 200 classes. This keeps the model deep enough to extract hierarchical features, but still within the assignment rules—no pretraining, no residual connections, just Conv–BN–ReLU–Pool–Dropout–Linear in a sequential style.

Why 200 outputs. The labels in the provided pickle files are in the original Tiny ImageNet index space, which goes up to 199. So I keep the final layer at 200 logits to exactly match the label encoding. That keeps the architecture compatible with the instructor’s hidden test set and evaluation code.

Learning rate & optimizer. I use Adam with an initial learning rate of 1e-3, which is a standard starting point for this scale of CNN. On top of that, I use a ReduceLROnPlateau scheduler that monitors validation accuracy and halves the learning rate when the metric plateaus. That combination lets the model learn quickly at the start, then take smaller steps once improvements slow down, which helped stabilize validation performance.

Regularization & overfitting control. To improve generalization I combine data augmentation, dropout, weight decay, and label smoothing. On the input side I use random resized crops, horizontal flips, small rotations, and mild color jitter—just enough to simulate viewpoint and lighting changes without destroying the objects. Inside the network each conv block has dropout 0.25 and the fully connected layer has dropout 0.5, plus L2 weight decay of 3×10⁻⁴. I also use cross-entropy loss with label smoothing 0.05 to make the model less overconfident. Initially the model was overfitting with train accuracy much higher than validation; after adding these regularization choices, train accuracy is around the high-80s while validation accuracy is in the low-70s, which is a healthier gap.

Training strategy. I set a reasonably high max epoch count and rely on early stopping based on validation accuracy, so the model stops when it stops actually improving instead of just memorizing the training set. I also fix random seeds for Python, NumPy, and PyTorch (CPU and CUDA) so that I can reproduce runs when I find a good configuration.
```