import numpy as np
import torch
import pytest
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

import loader
import augmentations
from augmentations import AugmentedStructure

""" Structural augmentation tests """

STRUCTURAL_TEST_STRUCTURE = loader.load_prototype("tests/mp-13_Fe.cif")


def test_zero_gaussian():
    positions = torch.tensor(STRUCTURAL_TEST_STRUCTURE.cart_coords, dtype=torch.float)
    cell = torch.tensor(STRUCTURAL_TEST_STRUCTURE.lattice.matrix, dtype=torch.float)
    new_positions = augmentations.apply_gaussian_noise(positions, cell, sigma=0.0)
    print(new_positions - positions)
    assert torch.allclose(
        new_positions, positions
    ), "Positions should be unchanged with zero noise"


def test_gaussian_rms():
    supercell = loader.make_supercell(
        STRUCTURAL_TEST_STRUCTURE, min_box_length=1.0, min_atoms=256
    )
    positions = torch.tensor(supercell.cart_coords, dtype=torch.float)
    cell = torch.tensor(supercell.lattice.matrix, dtype=torch.float)
    avg_rms = 0.0
    for it in range(5):
        sigma = 0.1
        new_positions = augmentations.apply_gaussian_noise(positions, cell, sigma=sigma)
        # compute displacement with PBC
        displacement = new_positions - positions
        for i in range(3):
            displacement[:, i] = displacement[:, i] - cell[i, i] * torch.round(
                displacement[:, i] / cell[i, i]
            )
        rms_displacement = torch.sqrt(torch.mean(displacement**2)).item()
        print(f"Iteration {it}: RMS displacement = {rms_displacement:.3f}")
        avg_rms += rms_displacement
    avg_rms /= 5
    assert (
        abs(avg_rms - sigma) < 0.01
    ), f"Average RMS displacement {avg_rms:.3f} should be close to sigma {sigma:.3f}"


def test_gaussian_interatomic_distance():
    positions = torch.tensor(STRUCTURAL_TEST_STRUCTURE.cart_coords, dtype=torch.float)
    cell = torch.tensor(STRUCTURAL_TEST_STRUCTURE.lattice.matrix, dtype=torch.float)
    sigma = 0.1
    new_positions = augmentations.apply_gaussian_noise(positions, cell, sigma=sigma)
    # Ensure no interatomic distances are below a reasonable threshold (0.5 * dnn)
    dists = torch.cdist(new_positions, new_positions)
    dists = dists[
        torch.triu(torch.ones_like(dists), diagonal=1).bool()
    ]  # upper triangle
    min_dist = dists.min().item()
    assert (
        min_dist > 0.5
    ), f"Minimum interatomic distance {min_dist:.3f} is too small, indicating possible overlaps"


def test_cell_strain():
    positions = torch.tensor(STRUCTURAL_TEST_STRUCTURE.cart_coords, dtype=torch.float)
    cell = torch.tensor(STRUCTURAL_TEST_STRUCTURE.lattice.matrix, dtype=torch.float)
    magnitude = 0.05
    new_positions, new_cell = augmentations.apply_cell_strain(
        positions, cell, magnitude
    )

    # fractional coordinates preserved
    frac_coords = torch.linalg.solve(cell.T, positions.T).T
    new_frac_coords = torch.linalg.solve(new_cell.T, new_positions.T).T
    assert torch.allclose(
        frac_coords, new_frac_coords, atol=1e-5
    ), "Fractional coordinates should be preserved under cell strain"

    # cartesian positions should change
    assert not torch.allclose(
        new_positions, positions
    ), "Cartesian positions should change under cell strain"

    # cell has relationship H = (I + strain) @ cell
    expected_new_cell = cell + (new_cell - cell)
    assert torch.allclose(
        new_cell, expected_new_cell
    ), "New cell should be related to original cell by the applied strain"


""" Chemical augmentation tests """

CHEMICAL_TEST_STRUCTURE = loader.load_prototype("tests/mp-2258_Cu3Au.cif")
B2_TEST_STRUCTURE = loader.load_prototype("tests/mp-571_TiNi.cif")
B2_TEST_STRUCTURE = loader.make_supercell(
    B2_TEST_STRUCTURE, min_box_length=10.0, min_atoms=64
)


def test_species_shuffle_zero():
    species = torch.tensor(CHEMICAL_TEST_STRUCTURE.atomic_numbers, dtype=torch.long)
    new_species = augmentations.apply_species_shuffle(species, fraction=0.0)
    assert torch.equal(
        new_species, species
    ), "Species should be unchanged with zero shuffle fraction"


@pytest.mark.parametrize("fraction", [0.1, 0.5, 1.0])
def test_species_shuffle_composition(fraction):
    species = torch.tensor(CHEMICAL_TEST_STRUCTURE.atomic_numbers, dtype=torch.long)
    new_species = augmentations.apply_species_shuffle(species, fraction=fraction)
    for element in torch.unique(species):
        original_count = (species == element).sum().item()
        new_count = (new_species == element).sum().item()
        assert (
            original_count == new_count
        ), f"Element {element} count changed from {original_count} to {new_count}"


def warren_cowley_a1(AugmentedStructure):
    # Get rcut (first shell)
    dists = torch.cdist(AugmentedStructure.positions, AugmentedStructure.positions)
    rcut = dists[dists > 0.1].min().item() * 1.1
    print(f"Using rcut = {rcut:.3f} for Warren-Cowley alpha_1 calculation")

    # alpha_1 = 1 - (N_AB / (N_AB + N_AA)) / (N_B / (N_A + N_B))
    # ref formula: https://github.com/killiansheriff/WarrenCowleyParameters
    species = AugmentedStructure.species
    species_numbers = torch.unique(species)
    assert (
        len(species_numbers) == 2
    ), "Warren-Cowley alpha_1 is only defined for binary systems"
    N_A = (species == species_numbers[0]).sum().item()
    N_B = (species == species_numbers[1]).sum().item()
    N_AB = 0
    N_AA = 0
    for i in range(len(species)):
        for j in range(i + 1, len(species)):
            if dists[i, j] < rcut:
                if (
                    species[i] == species_numbers[0]
                    and species[j] == species_numbers[1]
                ):
                    N_AB += 1
                elif (
                    species[i] == species_numbers[0]
                    and species[j] == species_numbers[0]
                ):
                    N_AA += 1
    print(f"N_A={N_A}, N_B={N_B}, N_AB={N_AB}, N_AA={N_AA}")
    return 1 - (N_AB / (N_AB + N_AA)) / (N_B / (N_A + N_B))


def test_species_shuffle_order():
    CHEMICAL_TEST_SUPERCELL = loader.make_supercell(
        CHEMICAL_TEST_STRUCTURE, min_box_length=10.0, min_atoms=64
    )
    species = torch.tensor(CHEMICAL_TEST_SUPERCELL.atomic_numbers, dtype=torch.long)
    new_species = augmentations.apply_species_shuffle(species, fraction=0.5)
    assert not torch.equal(
        new_species, species
    ), "Species should be different after shuffling with nonzero fraction"
    alpha_1_before = warren_cowley_a1(
        AugmentedStructure(
            positions=torch.tensor(
                CHEMICAL_TEST_SUPERCELL.cart_coords, dtype=torch.float
            ),
            cell=torch.tensor(
                CHEMICAL_TEST_SUPERCELL.lattice.matrix, dtype=torch.float
            ),
            species=species,
        )
    )
    alpha_1_after = warren_cowley_a1(
        AugmentedStructure(
            positions=torch.tensor(
                CHEMICAL_TEST_SUPERCELL.cart_coords, dtype=torch.float
            ),
            cell=torch.tensor(
                CHEMICAL_TEST_SUPERCELL.lattice.matrix, dtype=torch.float
            ),
            species=new_species,
        )
    )
    assert abs(alpha_1_after) * 2 < abs(
        alpha_1_before
    ), f"Warren-Cowley alpha_1 should decrease after shuffling, but went from {alpha_1_before:.3f} to {alpha_1_after:.3f}"


def test_species_permutation():
    # A B2 structure should be invariant to swapping the two species, so this is a good test case for permutation invariance
    species = torch.tensor(B2_TEST_STRUCTURE.atomic_numbers, dtype=torch.long)
    new_species = augmentations.apply_species_permutation(species)
    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True)
    original_struct = Structure(
        B2_TEST_STRUCTURE.lattice,
        B2_TEST_STRUCTURE.species,
        B2_TEST_STRUCTURE.cart_coords,
    )
    new_struct = Structure(
        B2_TEST_STRUCTURE.lattice, new_species.tolist(), B2_TEST_STRUCTURE.cart_coords
    )
    assert matcher.fit(
        original_struct, new_struct
    ), "Permuted structure should match original structure"


def test_mono_species():
    species = torch.tensor(CHEMICAL_TEST_STRUCTURE.atomic_numbers, dtype=torch.long)
    new_species = augmentations.apply_mono_species(species)
    assert torch.all(
        new_species == new_species[0]
    ), "All species should be the same after mono augmentation"
