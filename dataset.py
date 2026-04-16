from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import random

import torch
from torch import Tensor
from torch.utils.data import BatchSampler, Dataset
from torch_geometric.loader import DataLoader

from mace.data import AtomicData, Configuration
from mace.tools import AtomicNumberTable
from pymatgen.core import Structure

from augmentations import (
    AugmentedStructure,
    sample_augmentation_family,
)
from loader import BRAVAIS_LABELS, load_prototype, make_supercell, rescale_to_dnn

Z_TABLE = AtomicNumberTable([1, 2])


def _to_python_float(value: float | Tensor) -> float:
    if isinstance(value, Tensor):
        return float(value.item())
    return float(value)


def _convert_real_z_to_binary(species_real_z: Tensor) -> Tensor:
    """Map atomic numbers to abstract classes {0,1} expected by z-table [1,2]."""
    unique = torch.unique(species_real_z)
    if len(unique) == 1:
        return torch.zeros_like(species_real_z, dtype=torch.long)
    if len(unique) > 2:
        raise ValueError(
            f"Only unary/binary prototypes are supported, found {len(unique)} species: {unique.tolist()}"
        )
    unique_sorted, _ = torch.sort(unique)
    mapped = torch.zeros_like(species_real_z, dtype=torch.long)
    mapped[species_real_z == unique_sorted[1]] = 1
    return mapped


def make_atomic_data(
    augmented: AugmentedStructure,
    metadata: Dict[str, float | int],
    z_table: AtomicNumberTable,
    cutoff: float,
) -> AtomicData:
    """Convert AugmentedStructure into MACE AtomicData and attach metadata."""
    atomic_numbers = (augmented.species.to(torch.long) + 1).cpu().numpy()
    config = Configuration(
        atomic_numbers=atomic_numbers,
        positions=augmented.positions.to(torch.float32).cpu().numpy(),
        cell=augmented.cell.to(torch.float32).cpu().numpy(),
        pbc=[True, True, True],
    )
    data = AtomicData.from_config(config, z_table=z_table, cutoff=cutoff)

    data.bravais_label = torch.tensor(metadata["bravais_label"], dtype=torch.long)
    data.ordering_type_label = torch.tensor(
        metadata["ordering_type_label"], dtype=torch.long
    )
    data.sigma = torch.tensor(metadata["sigma"], dtype=torch.float32)
    data.shuffle_fraction = torch.tensor(
        metadata["shuffle_fraction"], dtype=torch.float32
    )
    data.family_id = torch.tensor(metadata["family_id"], dtype=torch.long)
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
        min_box_length: float = 10.0,
        min_atoms: int = 64,
        include_special_augs: bool = True,
    ):
        if n_scales < 1:
            raise ValueError("n_scales must be >= 1")

        self.z_table = Z_TABLE
        self.r_cut = float(r_cut)
        self.sigma_levels = [float(s) for s in sigma_levels]
        self.shuffle_levels = [float(p) for p in shuffle_levels]
        self.d_nn_range = (float(d_nn_range[0]), float(d_nn_range[1]))
        self.n_scales = int(n_scales)
        self.min_box_length = float(min_box_length)
        self.min_atoms = int(min_atoms)

        self.scale_values = torch.linspace(
            self.d_nn_range[0], self.d_nn_range[1], steps=self.n_scales
        ).tolist()
        self.include_special_augs = bool(include_special_augs)

        # sample_augmentation_family(None, ...) defines the canonical family structure.
        base_n = len(self.sigma_levels) * len(self.shuffle_levels)
        extras_n = len(self.sigma_levels) + len(self.shuffle_levels) + 4
        self.n_augmentations = base_n + (extras_n if self.include_special_augs else 0)

        self.ordering_label_map: Dict[str, int] = {}
        next_ordering_label = 0

        self.parents = []
        for family_id, pid in enumerate(prototype_ids):
            info = manifest[pid]
            atoms = self.load_and_prepare(cif_dir / info["cif_file"])
            self.parents.append(
                {
                    "family_id": family_id,
                    "prototype_id": pid,
                    "atoms": atoms,
                    "info": info,
                }
            )

            raw_ordering = info.get("ordering_type")
            if raw_ordering is None:
                raw_ordering = info.get("ordering")
            if isinstance(raw_ordering, str) and raw_ordering not in self.ordering_label_map:
                self.ordering_label_map[raw_ordering] = next_ordering_label
                next_ordering_label += 1

    def load_and_prepare(self, cif_path: Path) -> Structure:
        atoms = load_prototype(str(cif_path))
        atoms = make_supercell(
            atoms,
            min_box_length=self.min_box_length,
            min_atoms=self.min_atoms,
        )
        return atoms

    def _sample_scale(self) -> float:
        return random.choice(self.scale_values)

    def _ordering_label(self, info: Dict) -> int:
        raw = info.get("ordering_type_label")
        if raw is None:
            raw = info.get("ordering_type")
        if raw is None:
            raw = info.get("ordering")
        if raw is None:
            return 0
        if isinstance(raw, (int, float)):
            return int(raw)
        if isinstance(raw, str):
            if raw not in self.ordering_label_map:
                self.ordering_label_map[raw] = len(self.ordering_label_map)
            return self.ordering_label_map[raw]
        return int(raw)

    def __len__(self) -> int:
        return len(self.parents) * self.n_augmentations

    def __getitem__(self, idx: int) -> AtomicData:
        parent_idx = idx // self.n_augmentations
        aug_idx = idx % self.n_augmentations

        parent = self.parents[parent_idx]
        atoms = parent["atoms"]
        info = parent["info"]
        target_dnn = self._sample_scale()
        scaled = rescale_to_dnn(atoms, target_dnn)

        family = sample_augmentation_family(
            parent=scaled,
            family_id=parent["family_id"],
            batch_size=None,
            sigmas=self.sigma_levels,
            ps=self.shuffle_levels,
        )
        if not self.include_special_augs:
            family = [aug for aug in family if aug.augmentation_type == "cross"]

        augmented = family[aug_idx]
        augmented.species = _convert_real_z_to_binary(augmented.species.to(torch.long))
        augmented.sigma = _to_python_float(augmented.sigma)
        augmented.p = _to_python_float(augmented.p)

        bravais_key = info.get("bravais_lattice", info.get("bravais"))
        if bravais_key not in BRAVAIS_LABELS:
            raise KeyError(
                f"Manifest bravais lattice '{bravais_key}' not in BRAVAIS_LABELS"
            )

        metadata = {
            "bravais_label": BRAVAIS_LABELS[bravais_key],
            "ordering_type_label": self._ordering_label(info),
            "sigma": augmented.sigma,
            "shuffle_fraction": augmented.p,
            "family_id": parent["family_id"],
        }
        return make_atomic_data(
            augmented=augmented,
            metadata=metadata,
            z_table=self.z_table,
            cutoff=self.r_cut,
        )


class FamilySampler(BatchSampler):
    """
    Yield batches made of complete augmentation families.

    Each family contributes `dataset.n_augmentations` samples.
    """

    def __init__(
        self,
        dataset: PrototypeDataset,
        families_per_batch: int = 1,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        if families_per_batch < 1:
            raise ValueError("families_per_batch must be >= 1")
        self.dataset = dataset
        self.families_per_batch = int(families_per_batch)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)

        self.num_families = len(dataset.parents)
        self.family_size = dataset.n_augmentations

    def __iter__(self):
        family_ids = list(range(self.num_families))
        if self.shuffle:
            random.shuffle(family_ids)

        for i in range(0, self.num_families, self.families_per_batch):
            chunk = family_ids[i : i + self.families_per_batch]
            if self.drop_last and len(chunk) < self.families_per_batch:
                continue

            batch_indices: List[int] = []
            for family_id in chunk:
                start = family_id * self.family_size
                batch_indices.extend(range(start, start + self.family_size))
            yield batch_indices

    def __len__(self):
        if self.drop_last:
            return self.num_families // self.families_per_batch
        return (self.num_families + self.families_per_batch - 1) // self.families_per_batch


def build_family_dataloader(
    dataset: PrototypeDataset,
    families_per_batch: int = 1,
    shuffle_families: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Build a DataLoader where every batch contains full augmentation families."""
    family_batch_sampler = FamilySampler(
        dataset=dataset,
        families_per_batch=families_per_batch,
        shuffle=shuffle_families,
    )
    return DataLoader(
        dataset,
        batch_sampler=family_batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    