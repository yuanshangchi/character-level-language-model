import os
from itertools import chain

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from tqdm import tqdm

from src.data_loading.streaming_prefetcher import MultiThreadedStreamingPrefetcher


def save_stream_shards_parquet(
    dataset_iter, out_dir, total_samples=10_000_000, shard_size=100_000
):
    """
    Stream samples from a dataset iterator, save them to sharded Parquet files, and
    resume safely from previously saved shards if re-run.

    Args:
        dataset_iter (iterator): An iterable/iterator yielding dictionary samples, typically from a streaming dataset.
        out_dir (str): Path to the output directory where Parquet shards will be saved.
        total_samples (int): Total number of samples to save. Processing stops after this number is reached.
        shard_size (int): Number of samples per Parquet file (shard).

    Behavior:
        - If the output directory already contains shards, this function will detect how many
          samples have already been saved and resume from that point.
        - If all required samples are already saved, it exits without reprocessing.
        - The function writes data in batches of `shard_size` to numbered Parquet files
          (e.g., shard_00000.parquet, shard_00001.parquet, ...).
        - Skips and rewinds the dataset iterator as needed to align with already saved shards.
        - Prints a progress bar and logging messages for transparency.

    Raises:
        None. If the dataset stream is empty or fully consumed, it will exit cleanly.
    """

    # DEBUG: Check first item
    sample_iterator = iter(dataset_iter)
    try:
        first = next(sample_iterator)
        print("First item received from stream:")
        print(first)

        dataset_iter = chain([first], sample_iterator)
    except StopIteration:
        print("Dataset stream is empty!")
        return

    os.makedirs(out_dir, exist_ok=True)
    # shard_idx = 0
    # sample_count = 0
    # batch = []

    # Determine existing shards
    existing_shards = sorted(
        [
            f
            for f in os.listdir(out_dir)
            if f.endswith(".parquet") and f.startswith("shard_")
        ]
    )
    shard_idx = len(existing_shards)
    sample_count = shard_idx * shard_size

    if sample_count < total_samples:
        print(f"Resuming from shard {shard_idx}, sample {sample_count}")
    elif sample_count == total_samples:
        print(f"All {total_samples} samples already saved.")
        return

    # Skip samples already saved
    for _ in range(sample_count):
        try:
            next(dataset_iter)
        except StopIteration:
            print("Reached end of dataset while skipping already-saved samples.")
            return

    batch = []

    with tqdm(total=total_samples, desc="Streaming & Saving") as pbar:
        for item in dataset_iter:
            batch.append(item)
            sample_count += 1
            pbar.update(1)

            if len(batch) >= shard_size:
                table = pa.Table.from_pylist(batch)
                pq.write_table(
                    table, os.path.join(out_dir, f"shard_{shard_idx:05}.parquet")
                )
                shard_idx += 1
                batch = []

            if sample_count >= total_samples:
                break

        # Save any remaining samples
        if batch:
            table = pa.Table.from_pylist(batch)
            pq.write_table(
                table, os.path.join(out_dir, f"shard_{shard_idx:05}.parquet")
            )

    print(
        f"\n Saved {sample_count} samples across {shard_idx + (1 if batch else 0)} Parquet shards."
    )


if __name__ == "__main__":
    stream = load_dataset(
        # "oscar",
        # "unshuffled_deduplicated_en",
        # "cc100",
        # lang="en",
        "allenai/c4",
        "en",
        split="train",
        streaming=True,
        cache_dir="cache",
        trust_remote_code=True,
    )

    # Manually test if at least one sample exists
    sample_iterator = iter(stream)
    try:
        first_sample = next(sample_iterator)
        print("First sample received:")
        print(first_sample)

        # Rebuild stream with first item + rest (required because iterator is consumed)
        from itertools import chain

        stream = chain([first_sample], sample_iterator)
    except StopIteration:
        print("Dataset stream is empty!")
        exit(1)

    # change the buffer size and number of threads as needed by your system
    # For example, on a 12-core M3, you might want to set buffer_size=2000 and num_threads=10
    prefetched = MultiThreadedStreamingPrefetcher(
        stream, buffer_size=2000, num_threads=10
    )

    # change the total_samples and shard_size as needed by your system
    # For example, on a 12-core M3, you might want to set total_samples=10_000_000 and shard_size=100_000
    save_stream_shards_parquet(
        prefetched,
        # "data/oscar_10m_parquet",
        # total_samples=10_000_000, # [08:11<00:00, 20363.31it/s]
        # shard_size=100_000,
        "data/allenai_C4__10k_parquet",
        total_samples=10_000,
        shard_size=500,
    )
