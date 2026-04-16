from typing import Tuple, List, Dict
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset
from pymatgen.core import Structure
from mace.data import AtomicData, Configuration
from mace.tools import AtomicNumberTable

from loader import load_prototype, make_supercell, rescale_to_dnn, BRAVAIS_LABELS

Z_TABLE = AtomicNumberTable([1, 2]) # Two abstract types
R_CUT = 5.0 # Angstroms

def make_atomic_data(positions, species, cell, metadata):
    """Convert augmented structure to MACE AtomicData."""
    # Map type indices to MACE atomic number IDs
    # Type 0-> 1, Type 1-> 2 (arbitrary, not chemical)
    atomic_numbers = species + 1
    config = Configuration(
        atomic_numbers=atomic_numbers,
        positions=positions,
        cell=cell,
        pbc=[True, True, True],
    )
    data = AtomicData.from_config(
        config, z_table=Z_TABLE, cutoff=R_CUT
    )
    # Attach metadata as custom attributes
    data.bravais_label = metadata["bravais_label"]
    data.ordering_type_label = metadata["ordering_type_label"]
    data.sigma = metadata["sigma"]
    data.shuffle_fraction = metadata["shuffle_fraction"]
    data.family_id = metadata["family_id"]
    return data


class PrototypeDataset(Dataset):
    def __init__(
        self,
        cif_dir: Path,
        manifest: Dict,
        prototype_ids: List[str],
        sigma_levels: List[float],
        shuffle_levels: List[float],
        d_nn_range: Tuple[float, float],
        n_scales: int = 5,
        r_cut: float = 5.0,
    ):
        self.z_table = AtomicNumberTable([1, 2])
        self.r_cut = r_cut
        self.parents = []
        for pid in prototype_ids:
            info = manifest[pid]
            atoms = self.load_and_prepare(
            cif_dir / info["cif_file"])
            self.parents.append((atoms, info))

    def load_and_prepare(self, cif_path):
        """Load CIF, expand symmetry, build supercell.
        Uses your Assignment 2 functions."""
        ...

    def __len__(self):
        return len(self.parents) * self.n_augmentations
    
    def __getitem__(self, idx):
        parent_idx = idx // self.n_augmentations
        aug_idx = idx % self.n_augmentations
        atoms, info = self.parents[parent_idx]
        # Sample a lattice scale
        d_nn = self.sample_scale(atoms)
        scaled = self.rescale_to_dnn(atoms, d_nn)
        # Apply augmentation based on aug_idx
        positions, species, cell, aug_metadata = \
        self.apply_augmentation(scaled, aug_idx)
        # Build metadata dict
        metadata = {
            "bravais_label": BRAVAIS_LABELS[info["bravais_lattice"]],
            "ordering_type_label": ...,
            "sigma": aug_metadata["sigma"],
            "shuffle_fraction": aug_metadata["p"],
            "family_id": parent_idx,
        }
        # Convert to AtomicData through MACE’s pipeline
        return make_atomic_data(
        positions, species, cell, metadata)