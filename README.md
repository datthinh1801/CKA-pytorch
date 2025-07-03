# Centered Kernel Alignment (CKA) - PyTorch Implementation

A PyTorch implementation of Centered Kernel Alignment (CKA) with GPU support for fast and efficient computation.

> [!WARNING]
> This project is for educational and academic purposes (and for fun 🤷🏻).

## Features

- **GPU Accelerated:** Leverages the power of GPUs for significantly faster CKA calculations compared to NumPy-based implementations.
- **On-the-Fly Calculation:** Computes CKA on-the-fly using mini-batches, avoiding the need to cache large intermediate feature representations.
- **Easy to Use:** Simple and intuitive API for calculating the CKA matrix between two models.
- **Flexible:** Can be used with any PyTorch models and dataloaders.

## Installation
```bash
pip install cka-pytorch
```

## Usage

```python
import torch

from torchvision.models import resnet18
from torch.utils.data import DataLoader

from cka_pytorch.cka import CKACalculator


# 1. Define your models and dataloader
model1 = resnet18(pretrained=True).cuda()
model2 = resnet18(pretrained=True).cuda() # Or a different model

# Create a dummy dataloader for demonstration
dummy_data = torch.randn(100, 3, 224, 224)
dummy_labels = torch.randint(0, 10, (100,))
dummy_dataset = torch.utils.data.TensorDataset(dummy_data, dummy_labels)
dataloader = DataLoader(dummy_dataset, batch_size=32)

# 2. Initialize the CKACalculator
calculator = CKACalculator(model1, model2, dataloader)

# 3. Calculate the CKA matrix
cka_matrix = calculator.calculate_cka_matrix()

print("CKA Matrix:")
print(cka_matrix)

# 4. Plot the CKA Matrix as heatmap
calculator.plot_cka_matrix(title="CKA between ResNet18 and ResNet18")
```

## Contributing

- If you find this repository helpful, please give it a :star:.
- If you encounter any bugs or have suggestions for improvements, feel free to open an issue.
- This implementation has been primarily tested with ResNet architectures.

## Acknowledgement
This project is based on:
- [CKA.pytorch](https://github.com/numpee/CKA.pytorch)
- [centered-kernel-alignment](https://github.com/RistoAle97/centered-kernel-alignment)