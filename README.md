# crystal-augment

Crystal augmentation and dataset loading for MACE.

- `loader`: Utilities for loading CIFs, scaling, supercell generation
- `augmentations`: Functions to apply Gaussian noise and species shuffling to crystal structures.
- `batchloader`: Dataset and DataLoader classes for training MACE with augmented data.
- `losses`: Custom loss functions for training classifiers and rankers.

Test files are prefixed with `test_` and contain unit tests for the corresponding modules.