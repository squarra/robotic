import time

from torch.utils.data import DataLoader

from robotic.datasets import InMemoryDataset, LazyDataset


def profile_dataset(dataset, batch_size=32, num_workers=4, epochs=10):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    start = time.time()
    n_samples = 0

    for _ in range(epochs):
        for batch in loader:
            fake = []
            for x in batch:
                fake.append(x.float())
            _ = sum(t.mean() for t in fake if t.numel() > 0)
            n_samples += len(batch[0])

    elapsed = time.time() - start
    throughput = n_samples / elapsed
    return throughput, elapsed


DATASET_PATH = "dataset.h5"
FIELDS = ["poses", "sizes", "feasibles"]
NUM_EPOCHS = 10

# InMemory
inm = InMemoryDataset(DATASET_PATH, FIELDS)
th_inm, time_inm = profile_dataset(inm, batch_size=64, num_workers=4, epochs=NUM_EPOCHS)
print(f"[InMemory] {th_inm:.2f} samples/s over {NUM_EPOCHS} epochs ({time_inm:.2f}s)")

# Lazy
lazy = LazyDataset(DATASET_PATH, FIELDS)
th_lazy, time_lazy = profile_dataset(lazy, batch_size=64, num_workers=4, epochs=NUM_EPOCHS)
print(f"[Lazy] {th_lazy:.2f} samples/s over {NUM_EPOCHS} epochs ({time_lazy:.2f}s)")
