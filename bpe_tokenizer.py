import json
import re
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Iterable


class BPETokenizer:
    def __init__(self):
        self.merges: List[Tuple[str, str]] = []
        self.merge_ranks: Dict[Tuple[str, str], int] = {}
        self.vocab: Dict[str, int] = {}

    @staticmethod
    def _split_words(text: str) -> List[str]:
        return [w for w in text.strip().split() if w]

    @staticmethod
    def _word_to_symbols(word: str) -> Tuple[str, ...]:
        # Append </w> to mark end-of-word so merges don't cross word boundaries
        return tuple(list(word) + ["</w>"])

    @staticmethod
    def _get_stats(vocab: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            if not word:
                continue
            prev = word[0]
            for ch in word[1:]:
                pairs[(prev, ch)] += freq
                prev = ch
        return pairs

    @staticmethod
    def _merge_vocab(pair: Tuple[str, str], vocab: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, ...], int]:
        a, b = pair
        new_vocab = {}
        bigram = a + b
        for word, freq in vocab.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    new_word.append(bigram)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_vocab[tuple(new_word)] = freq
        return new_vocab

    def train(self, corpus_path: str, vocab_size: int = 1000, min_frequency: int = 2) -> None:
        """
        Train BPE merges from the given corpus.

        vocab_size: approximate desired vocabulary size (characters + merges)
        min_frequency: minimum pair frequency to be considered
        """
        with open(corpus_path, "r", encoding="utf-8") as f:
            text = f.read()

        words = self._split_words(text)
        # Build vocabulary of words as sequences of symbols with counts
        vocab_counter = Counter(self._word_to_symbols(w) for w in words)

        # Initialize base vocabulary with all characters
        base_symbols = Counter()
        for word, freq in vocab_counter.items():
            for s in word:
                base_symbols[s] += freq

        merges: List[Tuple[str, str]] = []
        merge_ranks: Dict[Tuple[str, str], int] = {}

        # We aim for vocab_size total symbols. Initial symbols are unique characters + </w>.
        # Estimate max merges as vocab_size - len(unique_symbols)
        unique_symbols = set(base_symbols.keys())
        max_merges = max(vocab_size - len(unique_symbols), 0)

        for i in range(max_merges):
            stats = self._get_stats(vocab_counter)
            if not stats:
                break
            # Filter by min_frequency
            stats = {k: v for k, v in stats.items() if v >= min_frequency}
            if not stats:
                break
            best = max(stats, key=stats.get)
            vocab_counter = self._merge_vocab(best, vocab_counter)
            merges.append(best)
            merge_ranks[best] = i

        # Build final vocabulary from resulting symbol sequences
        final_symbols = Counter()
        for word, freq in vocab_counter.items():
            for s in word:
                final_symbols[s] += freq

        # Exclude the end-of-word token from exposed vocab
        exposed_vocab = {s: c for s, c in final_symbols.items() if s != "</w>"}

        # Save state
        self.merges = merges
        self.merge_ranks = merge_ranks
        # Convert counts to indices sorted by frequency (optional)
        sorted_vocab = sorted(exposed_vocab.items(), key=lambda x: (-x[1], x[0]))
        self.vocab = {token: i for i, (token, _) in enumerate(sorted_vocab)}

    def _encode_word(self, word: str) -> List[str]:
        # Start with characters + </w>
        symbols = list(word) + ["</w>"]
        if not self.merges:
            return symbols[:-1]  # drop </w>

        # Greedy BPE: merge lowest-ranked pairs first until none apply
        pairs = self._adjacent_pairs(symbols)
        while pairs:
            # Choose pair with best rank (lowest value)
            ranked_pairs = [(self.merge_ranks.get(p, float("inf")), idx, p) for idx, p in pairs]
            best_rank, best_index, best_pair = min(ranked_pairs, key=lambda x: x[0])
            if best_rank == float("inf"):
                break
            # Merge the chosen pair at first occurrence
            i = best_index
            a, b = best_pair
            symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            pairs = self._adjacent_pairs(symbols)

        # Remove end-of-word marker
        return [s for s in symbols if s != "</w>"]

    @staticmethod
    def _adjacent_pairs(symbols: List[str]) -> List[Tuple[int, Tuple[str, str]]]:
        pairs = []
        for i in range(len(symbols) - 1):
            pairs.append((i, (symbols[i], symbols[i + 1])))
        return pairs

    def encode(self, text: str) -> List[str]:
        tokens: List[str] = []
        for w in self._split_words(text):
            tokens.extend(self._encode_word(w))
        return tokens

    def decode(self, tokens: Iterable[str]) -> str:
        """
        Best-effort detokenization suitable for human-readable output.

        Rationale:
        - Our encode() drops </w> markers, so we cannot perfectly reconstruct original word
          boundaries. Joining with spaces yields "A l l e n" artifacts.
        - This heuristic concatenates alphanumeric subword pieces into words and inserts
          spaces between words. It also fixes spacing around punctuation.

        Strategy:
        1) Build a provisional string by concatenating tokens without spaces.
        2) Insert spaces between transitions that likely indicate word boundaries.
        3) Normalize spaces around punctuation.

        This is not guaranteed to match the original text, but produces readable output.
        """
        toks = list(tokens)
        if not toks:
            return ""

        # Step 1: provisional concatenation (no spaces)
        s = "".join(toks)

        # Step 2: heuristic word-boundary insertion
        # Insert space between:
        # - lowercase/digit -> uppercase (camel-like)
        # - letter -> digit or digit -> letter
        # - sequences where a non-punctuation is followed by an opening quote if missing space
        s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", s)
        s = re.sub(r"(?<=[A-Za-z])(?=[0-9])", " ", s)
        s = re.sub(r"(?<=[0-9])(?=[A-Za-z])", " ", s)

        # Step 3: punctuation spacing normalization
        # Remove spaces before , . ! ? : ; ) ] } and add a single space after if needed
        # Ensure spaces after sentence-ending punctuation.
        def fix_punct(text: str) -> str:
            text = re.sub(r"\s+([,.;:!?)}\]])", r"\1", text)
            text = re.sub(r"([(\[{])\s+", r"\1", text)  # no space after opening
            # Ensure exactly one space after , ; : if followed by a word char
            text = re.sub(r"([,;:])([^\s])", r"\1 \2", text)
            # Ensure space after sentence enders if followed by a letter/quote
            text = re.sub(r"([.!?])([A-Za-z\"'])", r"\1 \2", text)
            # Collapse multiple spaces
            text = re.sub(r"\s{2,}", " ", text)
            # Trim
            return text.strip()

        s = fix_punct(s)
        return s

    def save(self, merges_path: str, vocab_path: str) -> None:
        with open(merges_path, "w", encoding="utf-8") as fm:
            for a, b in self.merges:
                fm.write(f"{a} {b}\n")
        with open(vocab_path, "w", encoding="utf-8") as fv:
            json.dump(self.vocab, fv, ensure_ascii=False, indent=2)

    def load(self, merges_path: str, vocab_path: str) -> None:
        merges: List[Tuple[str, str]] = []
        with open(merges_path, "r", encoding="utf-8") as fm:
            for line in fm:
                line = line.strip()
                if not line:
                    continue
                a, b = line.split()
                merges.append((a, b))
        self.merges = merges
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
        with open(vocab_path, "r", encoding="utf-8") as fv:
            self.vocab = json.load(fv)