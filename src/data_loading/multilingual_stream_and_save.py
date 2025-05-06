from src.data_loading.stream_and_save import save_stream_shards_parquet
from src.data_loading.streaming_prefetcher import (
    MultilingualStreamProportionalSamplerAllenaiC4,
    MultilingualStreamRoundRobinAllenaiC4,
    MultiThreadedStreamingPrefetcher,
)

if __name__ == "__main__":
    stream = MultilingualStreamRoundRobinAllenaiC4(
        languages=["en", "de", "hi"],
        dataset_name="allenai/c4",
        dataset_split="train",
        load_kwargs={"cache_dir": "cache", "trust_remote_code": True},
    )
    prefetched = MultiThreadedStreamingPrefetcher(
        stream, buffer_size=2000, num_threads=10
    )
    save_stream_shards_parquet(
        prefetched,
        # "data/c4_round_robin__10m_parquet",
        # total_samples=10_000_000,
        # shard_size=100_000,
        "data/c4_round_robin__10k_parquet",
        total_samples=10_000,
        shard_size=500,
    )

    """
    ==========================xx==========================xx========================
    """

    stream = MultilingualStreamProportionalSamplerAllenaiC4(
        lang_weights={"en": 0.5, "de": 0.25, "hi": 0.25},
        dataset_name="allenai/c4",
        dataset_split="train",
        load_kwargs={"cache_dir": "cache", "trust_remote_code": True},
    )
    prefetched = MultiThreadedStreamingPrefetcher(
        stream, buffer_size=2000, num_threads=10
    )
    save_stream_shards_parquet(
        prefetched,
        # "data/c4_proportional__10m_parquet",
        # total_samples=10_000_000,
        # shard_size=100_000,
        "data/c4_proportional__10k_parquet",
        total_samples=10_000,
        shard_size=500,
    )
