import h5py

DATASET_PATH = "dataset.h5"


with h5py.File(DATASET_PATH, "r") as f:
    primitives = f.attrs.get("primitives", [])
    print(f"Primitives: {list(primitives)}")

    datapoints = [k for k in f.keys() if k.startswith("dp_")]
    print(f"Number of datapoints: {len(datapoints)}")

    sample_name = sorted(datapoints)[0]
    dp = f[sample_name]

    for name, dataset in dp.items():
        shape = dataset.shape
        dtype = dataset.dtype
        print(f"{name:15s} shape={shape}, dtype={dtype}")
