# ResNeXt Image Classification (from-scratch PyTorch)

This repository contains a **from-scratch implementation of ResNeXt** (no pretrained torchvision ResNeXt used) and a full training/evaluation pipeline for a 3‑class image classification task. It includes data augmentation, training loops with LR scheduling, and evaluation with confusion matrix and precision/recall/F1.

> Notebook: `main code.ipynb`  

## Highlights
- Custom **ResNeXt building block** and **ResNeXt model** implemented in PyTorch (`nn.Module`).
- Data loading via `torchvision.datasets.ImageFolder` with strong augmentations (resize, flips, rotation, color jitter, normalization).
- Train/val/test split with `torch.utils.data.random_split`.
- Optimizer: **Adam** with weight decay; Scheduler: **StepLR** (step=5, gamma=0.5).
- Metrics: accuracy, loss curves, **confusion matrix**, and **precision/recall/F1** (via `sklearn.metrics`).
- Reproducibility: configurable random seed helper.
- Model checkpoint saved to `./a1_bonus_resnext_aarushij_singh72.pth`.

## Project Structure
```
.
├── main code.ipynb          # End-to-end notebook (model, training, evaluation)
├── cnn_part_2_dataset.zip          # (Expected) zipped dataset
└── README.md
```

## Dataset
The notebook expects an **image classification dataset organized by class folders** (compatible with `ImageFolder`). In this project it looks for a zip named `cnn_part_2_dataset.zip` and extracts to `cnn_part_2_dataset` on first run.

Expected folder structure after extraction:
```
cnn_part_2_dataset/
├── class_1/
│   ├── img_001.jpg
│   └── ...
├── class_2/
│   ├── ...
└── class_3/
    ├── ...
```

> The notebook sets `num_classes=3`. Update this if your dataset has a different number of classes.

## Setup

### 1) Create and activate a virtual environment (recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

> **Note on PyTorch**: For best results (and GPU acceleration), install a CUDA‑compatible build from [pytorch.org](https://pytorch.org/get-started/locally/). Otherwise, the CPU build will work but train more slowly.

## How to Run

1. Place your dataset zip (e.g., `cnn_part_2_dataset.zip`) in the project root.
2. Open Jupyter and run the notebook:
   ```bash
   jupyter lab   # or: jupyter notebook
   ```
3. In `main code.ipynb`:
   - The zip will be extracted to `cnn_part_2_dataset` if not already present.
   - Augmentations and loaders are created.
   - A **ResNeXt** model is instantiated and trained (default: 40 epochs, batch size 64, lr 1e-3).
   - Best model weights are saved to `a1_bonus_resnext_aarushij_singh72.pth`.
4. Evaluate:
   - The notebook reports **train/val/test accuracy** and **loss**.
   - It renders a **confusion matrix** and prints **precision/recall/F1** by class.

## Results (from the accompanying notebook)
Below is the summary table captured from the notebook’s analysis:

### Model Performance Summary

| Model        | Final Train Accuracy | Final Val Accuracy | Final Test Accuracy | Final Train Loss | Final Val Loss |
|--------------|----------------------|---------------------|----------------------|------------------|----------------|
| VGG-16     |     96.99%            |   94.07%             |     94.11           |          0.3511   |       0.4029    |
| ResNet-18    |    98.69%          |          96.89%  |          96.38%     |     0.3191     |      0.3489    |
| ResNeXt-50   |               98.50%  | 96.38%              | 96.53%                | 0.3479         | 0.3814      |


### b. Discussion of the observed differences in performance.

## Reproducing / Modifying
- Change augmentations in the `transforms.Compose([...])` block.
- Adjust training hyperparameters (`batch_size`, `num_epochs`, optimizer/scheduler) near the end of the notebook.
- If you use a different dataset or number of classes, update `num_classes` when constructing the model.

## References
- **ResNeXt: Aggregated Residual Transformations for Deep Neural Networks** — https://arxiv.org/abs/1611.05431
- torchvision / PyTorch docs for `ImageFolder`, `DataLoader`, transforms, and training utilities.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
