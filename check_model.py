import torch
import numpy as np

from main import load_model, get_dataloaders, predict

# 1. Load best model
model = load_model("model.pth") 
print("Loaded model OK.")

# 2. Build val loader
_, val_loader = get_dataloaders(
    train_path="train-70_.pkl",
    val_path="validation-10_.pkl",
    batch_size=64,
    num_workers=0,
)

# 3. Run predict on val set
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
preds = predict(model, val_loader, device=device)
print("Predictions shape:", preds.shape)

# Optional sanity: compute validation accuracy here
# (only if you know how to recover labels from your val pickle)
import pickle

with open("validation-10_.pkl", "rb") as f:
    data = pickle.load(f)

if isinstance(data, dict) and "labels" in data:
    y_val = np.array(data["labels"])
elif isinstance(data, dict) and "y" in data:
    y_val = np.array(data["y"])
elif isinstance(data, (list, tuple)) and len(data) == 2:
    _, y_val = data
    y_val = np.array(y_val)
else:
    raise RuntimeError("Unexpected val pkl format")

acc = (preds == y_val).mean()
print("Recomputed val accuracy:", acc)
