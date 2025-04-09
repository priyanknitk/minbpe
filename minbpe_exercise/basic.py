"""
This module implements a simple Byte-Pair Encoding (BPE) tokenizer.
It includes functions for training on text, encoding text into tokens,
and decoding tokens back to text.
"""

from minbpe.base import Tokenizer, get_stats, merge


class MyBasicTokenizer(Tokenizer):
    """
    MyBasicTokenizer is a simple tokenizer that can train on text to build a 
    vocabulary,
    and encode/decode text into sequences of integers.

    Methods:
        train(text: str, vocab_size: int, verbose: bool):
            Train the tokenizer on the given text by splitting it into words
            and constructing a vocabulary.
        encode(text: str) -> List[int]:
            Encode the text into a sequence of integer tokens
            based on the vocabulary.
        decode(tokens: List[int]) -> str:
            Decode a sequence of integer tokens back into the original text.
    """

    def train(self, text, vocab_size, verbose=False):
        """
        Train the tokenizer on the given text.
        :param text: The text to train on.
        :param vocab_size: The size of the vocabulary.
        :param verbose: If True, print the training process.
        """
        num_merges = vocab_size - 256  # number of merges to perform
        utf_bytes = text.encode("utf-8")  # raw bytes
        ids = list(utf_bytes)  # list of integers in range 0..255
        merges = {}  # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)}  # int -> bytes
        for _ in range(num_merges):
            counts = get_stats(ids)
            pair = max(counts, key=counts.get)
            idx = 256 + len(merges)
            ids = merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f"merge {len(merges)}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {counts[pair]} occurrences")
        self.vocab = vocab  # int -> bytes
        self.merges = merges

    def encode(self, text: str):
        """
        Encode the text into a sequence of integers.
        :param text: The text to encode.
        :return: A list of integers representing the encoded text.
        """
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        while len(ids) >= 2:
            pairs = get_stats(ids)
            if not pairs:
                break
            pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def decode(self, ids):
        """
        Decode a sequence of integers back into text.
        :param ids: The list of integers to decode.
        :return: The decoded text.
        """
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text
