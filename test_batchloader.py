import torch
from torch.utils.data import DataLoader

import batchloader


def test_batchloader():
    ds = batchloader.PrototypeDataset(
        cif_dir="tests",
        manifest={
            571: {
                "cif_file": "mp-571_TiNi.cif",
                "bravais_label": "aP",
                "ordering_type_label": "oP",
            },
            13: {
                "cif_file": "mp-13_Fe.cif",
                "bravais_label": "cP",
                "ordering_type_label": "cP",
            }
        },
        sigma_levels=[0.1, 0.2, 0.3, 0.4, 0.5],
        shuffle_levels=[0.1, 0.2, 0.3, 0.4, 0.5],
        d_nn_range=(2.0, 4.0),
        n_scales=5,
        r_cut=5.0,
        min_box_length=10.0,
    )
    print(f"Dataset initialized with {len(ds)} samples")
    dl = DataLoader(
        ds,
        batch_sampler=batchloader.FamilySampler(ds, batch_size=64, family_size=16),
        collate_fn=batchloader.atomic_data_collate_fn,
    )
    for epoch in range(2):
        print(f"Epoch {epoch}")
        dl.dataset.set_epoch(epoch)
        for i, batch in enumerate(dl):
            dl.dataset.set_batch(i)
            print(f"Batch {i}, batch size: {len(batch)}")
            for data in batch:
                print(
                    f"  Sample with bravais_label={data.bravais_label}, ordering_type_label={data.ordering_type_label}, sigma={data.sigma}, shuffle_fraction={data.shuffle_fraction}"
                )
            for data_1 in batch:
                for data_2 in batch:
                    if data_1.family_id != data_2.family_id:
                        continue
                    if data_1.sigma == data_2.sigma:
                        assert torch.equal(
                            data_1.positions, data_2.positions
                        ), "Positions should be the same for samples with the same sigma"

                    if data_1.shuffle_fraction == data_2.shuffle_fraction:
                        assert torch.equal(data_1.node_attrs, data_2.node_attrs), (
                            "Species should be the same for samples with the same shuffle_fraction.\n"
                            f"data_1.shuffle_fraction={data_1.shuffle_fraction}, data_2.shuffle_fraction={data_2.shuffle_fraction}\n"
                            f"data_1.node_attrs={"".join(str(i[0]) for i in data_1.node_attrs)}, data_2.node_attrs={"".join(str(i[0]) for i in data_2.node_attrs)}"
                        )


if __name__ == "__main__":
    test_batchloader()
