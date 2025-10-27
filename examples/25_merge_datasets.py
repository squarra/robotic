#!/usr/bin/env python3
import argparse

import h5py


def main():
    parser = argparse.ArgumentParser(description=("Concatenate multiple HDF5 files into a single output file."))
    parser.add_argument("inputs", nargs="+", help="Input HDF5 files in the desired concatenation order.")
    args = parser.parse_args()

    with h5py.File("dataset-merged.h5", "w") as fout:
        first = args.inputs[0]
        with h5py.File(first, "r") as f0:
            for k, v in f0.attrs.items():
                fout.attrs[k] = v

        for path in args.inputs:
            with h5py.File(path, "r") as fin:
                for key in fin.keys():
                    fin.copy(source=key, dest=fout)


if __name__ == "__main__":
    main()
