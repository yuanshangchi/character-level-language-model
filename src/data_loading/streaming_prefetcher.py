import queue
import random
import threading
from itertools import cycle

from datasets import load_dataset


class MultiThreadedStreamingPrefetcher:
    """
    Multi-threaded prefetcher for any Python iterable (e.g., streaming datasets).

    This class launches multiple background threads to pull data from a given iterator
    (such as a streaming dataset) and places the results in a thread-safe buffer (queue),
    allowing for parallel prefetching and faster downstream processing.

    Args:
        dataset_iter (iterator): An iterable yielding data samples.
        buffer_size (int): Max number of prefetched items to keep in memory.
        num_threads (int): Number of threads to use for concurrent prefetching.

    Usage:
        Wrap any iterable (e.g., dataset, generator) to prefetch in the background.

    Example:
        stream = MultiThreadedStreamingPrefetcher(dataset_iter, buffer_size=1000, num_threads=4)
        for item in stream:
            process(item)
    """

    def __init__(self, dataset_iter, buffer_size=1000, num_threads=4):
        self.dataset_iter = dataset_iter
        self.buffer = queue.Queue(maxsize=buffer_size)
        self.lock = threading.Lock()
        self.stop_signal = object()
        self.threads = []
        self.num_threads = num_threads
        self.started = False

    def _worker(self):
        while True:
            with self.lock:
                try:
                    item = next(self.dataset_iter)
                except StopIteration:
                    break
                except Exception:
                    # Handle any streaming error here if needed
                    break

            self.buffer.put(item)

        # Each thread signals when it's done
        self.buffer.put(self.stop_signal)

    def start(self):
        if not self.started:
            for _ in range(self.num_threads):
                t = threading.Thread(target=self._worker, daemon=True)
                t.start()
                self.threads.append(t)
            self.started = True

    def __iter__(self):
        self.start()
        stop_signals_received = 0
        while stop_signals_received < self.num_threads:
            item = self.buffer.get()
            if item is self.stop_signal:
                stop_signals_received += 1
            else:
                yield item


class MultilingualStreamRoundRobinAllenaiC4:
    """
    Multilingual streaming iterator using a round-robin strategy over multiple language streams
    from the allenai/c4 dataset.

    At each iteration, it cycles through the provided languages in order and returns the next
    valid text sample (based on a minimum text length filter). This ensures balanced sampling
    across all included languages.

    Args:
        languages (list): ISO 639-1 codes (e.g., ["en", "es", "hi"]).
        dataset_name (str): Hugging Face dataset name (default: "allenai/c4").
        dataset_split (str): Dataset split to load (default: "train").
        streaming (bool): Whether to stream the dataset (default: True).
        load_kwargs (dict): Extra kwargs passed to `load_dataset`.
        min_text_length (int): Minimum length (non-whitespace chars) of returned text.
        seed (int): Random seed for deterministic language order.

    Yields:
        dict: A dictionary with a single key `"text"`.

    Example:
        stream = MultilingualStreamRoundRobinAllenaiC4(["en", "fr"])
        for item in stream:
            print(item["text"])
    """

    def __init__(
        self,
        languages,
        dataset_name="allenai/c4",
        dataset_split="train",
        streaming=True,
        load_kwargs=None,
        min_text_length=10,
        seed=42,
    ):
        """
        Args:
            languages (list): ISO 639-1 language codes (e.g., ["en", "es", "fr"])
            dataset_name (str): Dataset name (default: "allenai/c4")
            dataset_split (str): Split string (default: "train")
            streaming (bool): Enable streaming (default: True)
            load_kwargs (dict): Additional kwargs for `load_dataset()`
            min_text_length (int): Minimum number of characters in a sample
            seed (int): Random seed for reproducibility
        """
        self.languages = languages
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.streaming = streaming
        self.load_kwargs = load_kwargs or {}
        self.min_text_length = min_text_length
        self.random = random.Random(seed)

        self.streams = {
            lang: iter(
                load_dataset(
                    self.dataset_name,
                    lang,
                    split=self.dataset_split,
                    streaming=self.streaming,
                    **self.load_kwargs,
                )
            )
            for lang in self.languages
        }

        self.lang_cycle = cycle(self.languages)

    def __iter__(self):
        return self

    def __next__(self):
        tried = set()
        while len(tried) < len(self.languages):
            lang = next(self.lang_cycle)
            tried.add(lang)

            try:
                while True:  # Loop until we get a usable item or exhaust this language
                    text = next(self.iterators[lang])["text"]
                    if len(text.strip()) > 10:  # skip short or empty lines
                        return {"text": text}
            except StopIteration:
                continue  # Stream exhausted
            except Exception:
                continue  # Could also log errors here

        raise StopIteration  # All streams exhausted


class MultilingualStreamProportionalSamplerAllenaiC4:
    """
    Multilingual streaming iterator using proportional random sampling over multiple language
    streams from the allenai/c4 dataset.

    Each sample is drawn from one of the configured languages based on its assigned weight.
    This allows you to bias the stream toward high-resource or preferred languages.

    Args:
        lang_weights (dict): Dictionary mapping language codes to weights (e.g., {"en": 0.6, "fr": 0.4}).
        dataset_name (str): Hugging Face dataset name (default: "allenai/c4").
        dataset_split (str): Dataset split to load (default: "train").
        streaming (bool): Whether to use streaming mode (default: True).
        load_kwargs (dict): Additional kwargs for `load_dataset`.
        min_text_length (int): Minimum number of non-whitespace characters required.
        seed (int): Random seed for consistent sampling.

    Yields:
        dict: A dictionary with keys `"text"` and `"lang"`.

    Example:
        stream = MultilingualStreamProportionalSamplerAllenaiC4({"en": 0.8, "de": 0.2})
        for item in stream:
            print(f"[{item['lang']}] {item['text']}")
    """

    def __init__(
        self,
        lang_weights,
        dataset_name="allenai/c4",
        dataset_split="train",
        streaming=True,
        load_kwargs=None,
        min_text_length=10,
        seed=42,
    ):
        """
        Args:
            lang_weights (dict): Dictionary of language weights (e.g., {"en": 0.5, "es": 0.3, "fr": 0.2})
            dataset_name (str): Dataset name (default: "allenai/c4")
            dataset_split (str): Split string (default: "train")
            streaming (bool): Enable streaming (default: True)
            load_kwargs (dict): Additional kwargs for `load_dataset()`
            min_text_length (int): Minimum number of characters in a sample
            seed (int): Random seed for reproducibility
        """
        self.lang_weights = lang_weights
        self.languages = list(lang_weights.keys())
        self.weights = list(lang_weights.values())
        self.min_text_length = min_text_length
        self.random = random.Random(seed)
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.streaming = streaming
        self.load_kwargs = load_kwargs or {}

        self.streams = {
            lang: iter(
                load_dataset(
                    self.dataset_name,
                    lang,
                    split=self.dataset_split,
                    streaming=self.streaming,
                    **self.load_kwargs,
                )
            )
            for lang in self.languages
        }

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            lang = self.random.choices(self.languages, weights=self.weights, k=1)[0]
            try:
                while True:
                    sample = next(self.streams[lang])
                    text = sample["text"].strip()
                    if len(text) >= self.min_text_length:
                        return {"text": text, "lang": lang}
            except StopIteration:
                print(f"Stream exhausted for language: {lang}")
                self.languages.remove(lang)
                self.weights = [self.lang_weights[l] for l in self.languages]
                if not self.languages:
                    raise StopIteration
