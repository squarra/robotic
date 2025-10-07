#!/usr/bin/env python3
import sys

import h5py


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <dataset_path>")
        sys.exit(1)

    dataset_path = sys.argv[1]

    with h5py.File(dataset_path, "r") as f:
        print(f"Number of datapoints: {len(f)}")

        primitives = f.attrs.get("primitives", [])
        print(f"Primitives: {list(primitives)}")

        first_dp = sorted(f)[0]
        print(f"Start seed: {int(first_dp.split('_')[1])}")

        for name, dataset in f[first_dp].items():
            print(f"{name:15s} shape={dataset.shape}, dtype={dataset.dtype}")


if __name__ == "__main__":
    main()
