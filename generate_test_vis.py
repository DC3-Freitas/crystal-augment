import torch

import loader
import augmentations

struct = loader.load_prototype("tests/mp-2258_Cu3Au.cif")
supercell = loader.make_supercell(struct, min_box_length=1.0, min_atoms=108)
supercell = loader.rescale_to_dnn(supercell, target_dnn=2.55)

strained_positions, strained_cell = augmentations.apply_cell_strain(
    torch.tensor(supercell.cart_coords, dtype=torch.float), torch.tensor(supercell.lattice.matrix, dtype=torch.float), magnitude=0.05
)

aug = augmentations.AugmentedStructure(
    species=torch.tensor(supercell.atomic_numbers, dtype=torch.long),
    positions=torch.tensor(strained_positions, dtype=torch.float),
    cell=torch.tensor(strained_cell, dtype=torch.float),
    family_id=0,
    augmentation_type="cell_strain",
    sigma=0.1,
    p=1.0,
)
print(aug.family_id, aug.augmentation_type, aug.sigma, aug.p)
aug.to_xyz(f"test_vis/test_aug_family_{aug.family_id}_{aug.augmentation_type}_sigma{aug.sigma}_p{aug.p}.xyz")

# family = augmentations.sample_augmentation_family(
#     parent=supercell,
#     family_id=0,
#     batch_size=None,
#     sigmas=[0.05,0.1,0.2,0.3,0.5],
#     ps=[0.1,0.3,0.5,0.7,1.0],
# )

# for aug in family:
#     print(aug.family_id, aug.augmentation_type, aug.sigma, aug.p)
#     aug.to_xyz(f"test_aug_family_{aug.family_id}_{aug.augmentation_type}_sigma{aug.sigma}_p{aug.p}.xyz")