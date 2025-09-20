import h5py

input_paths = ["dataset-0000-1000.h5", "dataset-1000-1500.h5"]
output_path = "dataset-merged.h5"

with h5py.File(output_path, "w") as fout:
    # Copy attributes
    with h5py.File(input_paths[0], "r") as f0:
        for k, v in f0.attrs.items():
            fout.attrs[k] = v

    # Go through each dataset file
    for path in input_paths:
        with h5py.File(path, "r") as fin:
            for key in fin.keys():
                fin.copy(key, fout)
