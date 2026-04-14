import loader
from matplotlib import pyplot as plt
import numpy as np


def plot_structure(struct):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        struct.cart_coords[:, 0], struct.cart_coords[:, 1], struct.cart_coords[:, 2]
    )
    plt.show()


def plot_histogram(struct):
    dists = struct.distance_matrix
    n_atoms = len(struct)
    volume = struct.lattice.volume
    density = n_atoms / volume

    # Use only unique pairs (upper triangle) to avoid double-counting distances.
    pair_dists = dists[np.triu_indices(n_atoms, k=1)]
    pair_dists = pair_dists[pair_dists > 0]  # exclude zero distances (self-distances)
    pair_dists = pair_dists[pair_dists < 10.0]  # focus on distances up to 10 Angstrom

    bins = 200
    counts, bin_edges = np.histogram(pair_dists, bins=bins, range=(0, 10.0))
    dr = np.diff(bin_edges)
    r = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    shell_volume = 4.0 * np.pi * r**2 * dr
    expected_counts = 0.5 * n_atoms * density * shell_volume
    g_r = counts / expected_counts

    plt.plot(r, counts)
    plt.xlabel("Interatomic Distance (Angstrom)")
    plt.ylabel("g(r)")
    plt.title("Radial Distribution Function")
    plt.show()


struct = loader.load_prototype("tests/fe.cif")
supercell = loader.make_supercell(struct, min_box_length=20.0, min_atoms=64)
scaled_struct = loader.rescale_to_dnn(supercell, target_dnn=1)
print(loader.compute_scaling_bounds(scaled_struct, r_cut=1.4, d_nn_init=1))
# plot_structure(scaled_struct)
plot_histogram(scaled_struct)
