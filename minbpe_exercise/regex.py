import regex as re
from minbpe.base import Tokenizer, get_stats, merge

GPT4_SPLIT_PATTERN = r"""(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class MyRegexTokenizer(Tokenizer):
    # Uses GPT-4 regex pattern to split text and processes each part separately.
    def train(self, text, vocab_size, verbose=False):
        num_merges = vocab_size - 256
        # Split text using GPT-4 pattern
        parts = re.findall(GPT4_SPLIT_PATTERN, text)
        # Convert each part to a list of byte ids
        token_seqs = [list(part.encode("utf-8")) for part in parts]
        vocab = {idx: bytes([idx]) for idx in range(256)}
        merges = {}
        for _ in range(num_merges):
            counts = {}
            stats = {}
            for seq in token_seqs:
                get_stats(seq, stats)
            pair = max(counts, key=counts.get)
            idx = 256 + len(merges)
            # Merge the chosen pair in each token sequence (do not merge across parts)
            token_seqs = [merge(seq, pair, idx) for seq in token_seqs]
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(
                    f"merge {len(merges)}/{num_merges}: {pair} -> {idx} (vocab token length: {len(vocab[idx])}) had {counts[pair]} occurrences"
                )
        self.vocab = vocab
        self.merges = merges

    def encode(self, text: str):
        parts = re.findall(GPT4_SPLIT_PATTERN, text)
        result = []
        for part in parts:
            ids = list(part.encode("utf-8"))
            while len(ids) >= 2:
                pairs = get_stats(ids)
                if not pairs:
                    break
                pair = max(pairs, key=pairs.get)
                if pair not in self.merges:
                    break
                idx = self.merges[pair]
                ids = merge(ids, pair, idx)
            result.extend(ids)
        return result

    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        return text_bytes.decode("utf-8", errors="replace")
