import argparse
import json
import math
import random
from typing import List, Optional, Sequence, Set

from bpe_tokenizer import BPETokenizer
from embedding import EmbeddingLayer
from positional_encoding import SinusoidalPositionalEncoding
from transformer_blocks import TransformerEncoder, generate_causal_mask


def _softmax_vec(logits: Sequence[float], temperature: float = 1.0) -> List[float]:
    if not logits:
        return []
    t = max(1e-6, float(temperature))
    m = max(logits)
    exps = [math.exp((x - m) / t) for x in logits]
    s = sum(exps)
    if s == 0.0:
        return [1.0 / len(logits)] * len(logits)
    return [e / s for e in exps]


def _matvec(A: Sequence[Sequence[float]], x: Sequence[float]) -> List[float]:
    m, n = len(A), (len(A[0]) if A else 0)
    if n != len(x):
        raise ValueError(f"matvec mismatch: {n} vs {len(x)}")
    out = [0.0] * m
    for i in range(m):
        s = 0.0
        Ai = A[i]
        for k in range(n):
            s += Ai[k] * x[k]
        out[i] = s
    return out


def _top_k_indices(probs: Sequence[float], k: int) -> List[int]:
    V = len(probs)
    if V == 0:
        return []
    k = max(1, min(k, V))
    return sorted(range(V), key=lambda i: probs[i], reverse=True)[:k]


def _sample_from_top_k(probs: Sequence[float], k: int, rng: random.Random) -> int:
    idxs = _top_k_indices(probs, k)
    mass = sum(probs[i] for i in idxs)
    if mass <= 0.0:
        return idxs[0]
    renorm = [probs[i] / mass for i in idxs]
    r = rng.random()
    c = 0.0
    for j, p in zip(idxs, renorm):
        c += p
        if r <= c:
            return j
    return idxs[-1]


def _sample_from_top_p(probs: Sequence[float], p: float, rng: random.Random) -> int:
    V = len(probs)
    if V == 0:
        return 0
    p = max(1e-6, min(float(p), 1.0))
    idx_sorted = sorted(range(V), key=lambda i: probs[i], reverse=True)
    nucleus: List[int] = []
    acc = 0.0
    for j in idx_sorted:
        nucleus.append(j)
        acc += probs[j]
        if acc >= p:
            break
    mass = sum(probs[i] for i in nucleus)
    if mass <= 0.0:
        return nucleus[0]
    renorm = [probs[i] / mass for i in nucleus]
    r = rng.random()
    c = 0.0
    for j, q in zip(nucleus, renorm):
        c += q
        if r <= c:
            return j
    return nucleus[-1]


def _post_process_probs(
    probs: Sequence[float],
    emb: EmbeddingLayer,
    allow_only: Optional[Set[str]] = None,
    ban_tokens: Optional[Set[str]] = None,
    exclude_pad_token: Optional[str] = None,
    min_prob: float = 0.0,
) -> List[float]:
    V = len(probs)
    if V == 0:
        return []
    out = list(probs)

    if allow_only:
        allow_ids = {emb.token_to_id.get(t, emb.unk_id) for t in allow_only}
        for i in range(V):
            if i not in allow_ids:
                out[i] = 0.0

    if ban_tokens:
        for t in ban_tokens:
            idx = emb.token_to_id.get(t, None)
            if idx is not None and 0 <= idx < V:
                out[idx] = 0.0

    if exclude_pad_token:
        idx = emb.token_to_id.get(exclude_pad_token, None)
        if idx is not None and 0 <= idx < V:
            out[idx] = 0.0

    if min_prob > 0.0:
        thr = float(min_prob)
        for i in range(V):
            if out[i] < thr:
                out[i] = 0.0

    s = sum(out)
    if s > 0.0:
        out = [x / s for x in out]
    return out


class Chatbot:
    def __init__(
        self,
        merges_path: str,
        vocab_path: str,
        head_path: str,
        dim: int,
        layers: int,
        heads: int,
        ff: int,
        add_pe: bool,
        seed: int = 123,
    ) -> None:
        with open(head_path, "r", encoding="utf-8") as f:
            head = json.load(f)
        W_out = head["W_out"]
        b_out = head["b_out"]
        head_dim = int(head["dim"])
        if head_dim != dim:
            raise ValueError(f"Head dim {head_dim} does not match model dim {dim}")

        self.W_out = W_out
        self.b_out = b_out

        self.tok = BPETokenizer()
        self.tok.load(merges_path, vocab_path)
        self.emb = EmbeddingLayer.from_vocab_file(vocab_path, dim=dim, seed=42)

        self.encoder = TransformerEncoder(
            num_layers=layers,
            dim=dim,
            num_heads=heads,
            ff_hidden=ff,
            seed=seed,
        )
        self.pe = SinusoidalPositionalEncoding(dim=dim) if add_pe else None

        self.inv_vocab = {i: t for t, i in self.emb.token_to_id.items()}
        if self.emb.unk_id not in self.inv_vocab:
            self.inv_vocab[self.emb.unk_id] = "<unk>"

        self._rng = random.Random(seed ^ 0xBEEF)

    def _forward_hidden(self, tokens: Sequence[str]) -> List[List[float]]:
        X = self.emb.embed_tokens(tokens)
        if self.pe is not None:
            X = self.pe.add_to(X, offset=0)
        mask = generate_causal_mask(len(tokens))
        return self.encoder(X, mask=mask)

    def _next_token_distribution(
        self,
        tokens: Sequence[str],
        temperature: float = 1.0,
    ) -> List[float]:
        H = self._forward_hidden(tokens)
        x_last = H[-1]
        logits = _matvec(self.W_out, x_last)
        for i in range(len(logits)):
            logits[i] += self.b_out[i]
        return _softmax_vec(logits, temperature=temperature)

    def generate(
        self,
        prompt_tokens: Sequence[str],
        max_new_tokens: int = 32,
        temperature: float = 0.9,
        top_k: int = 20,
        top_p: float = 0.0,
        stop_token: Optional[str] = None,
        greedy: bool = False,
        stream: bool = False,
        allow_only: Optional[Set[str]] = None,
        ban_tokens: Optional[Set[str]] = None,
        exclude_pad_token: Optional[str] = None,
        min_prob: float = 0.0,
        no_repeat_ngram_size: int = 0,
        ban_immediate_repeat: bool = False,
        require_alpha_start: bool = False,
        min_token_len: int = 0,
        repeat_window: int = 0,
        max_token_repeat: int = 0,
        allow_alpha_only: bool = False,
    ) -> List[str]:
        tokens = list(prompt_tokens)
        V = len(self.inv_vocab)

        def violates_no_repeat(candidate: str) -> bool:
            n = int(no_repeat_ngram_size)
            if n <= 0:
                return False
            # Candidate n-gram is last n-1 tokens + candidate
            if len(tokens) < n - 1:
                return False
            cand_ngram = tokens[-(n - 1):] + [candidate]
            # Scan history for the same contiguous n-gram
            for i in range(len(tokens) - n + 1):
                if tokens[i : i + n] == cand_ngram:
                    return True
            return False

        if stream:
            initial = self.tok.decode(prompt_tokens)
            if initial:
                print(initial, end=" ", flush=True)

        for _ in range(max_new_tokens):
            probs_raw = self._next_token_distribution(tokens, temperature=temperature)
            probs = _post_process_probs(
                probs_raw,
                emb=self.emb,
                allow_only=allow_only,
                ban_tokens=ban_tokens,
                exclude_pad_token=exclude_pad_token,
                min_prob=min_prob,
            )

            # Position-aware masking to enforce clean starts/lengths
            new_idx = len(tokens) - len(prompt_tokens)
            if (require_alpha_start and new_idx == 0) or (min_token_len > 0):
                mask = [1.0] * len(probs)
                for i in range(len(probs)):
                    t = self.inv_vocab.get(i, "<unk>")
                    violates_alpha = (require_alpha_start and new_idx == 0 and (not t or not t[0].isalpha()))
                    violates_len = (min_token_len > 0 and (len(t) < min_token_len))
                    if violates_alpha or violates_len:
                        mask[i] = 0.0
                # Apply mask and renormalize if any mass remains
                probs = [p * m for p, m in zip(probs, mask)]
                s = sum(probs)
                if s > 0.0:
                    probs = [p / s for p in probs]
                # If s == 0.0, keep original probs (fallback to constraints check below)

            # Alphabetic-only constraint (optional hard filter)
            if allow_alpha_only:
                mask_alpha = [1.0] * len(probs)
                for i in range(len(probs)):
                    t = self.inv_vocab.get(i, "<unk>")
                    if not t.isalpha():
                        mask_alpha[i] = 0.0
                probs = [p * m for p, m in zip(probs, mask_alpha)]
                s = sum(probs)
                if s > 0.0:
                    probs = [p / s for p in probs]

            # Frequency penalty within recent window to discourage repeated tokens
            if repeat_window and repeat_window > 0 and max_token_repeat and max_token_repeat > 0:
                # Count occurrences in the last repeat_window generated tokens (excluding prompt)
                recent = tokens[-min(len(tokens), repeat_window):]
                freq = {}
                for t in recent:
                    freq[t] = freq.get(t, 0) + 1
                penalized = list(probs)
                for i in range(len(penalized)):
                    tok = self.inv_vocab.get(i, "<unk>")
                    c = freq.get(tok, 0)
                    if c >= max_token_repeat:
                        penalized[i] *= 0.1  # downweight heavily
                s2 = sum(penalized)
                if s2 > 0.0:
                    probs = [x / s2 for x in penalized]

            # Choose a candidate id
            def pick_candidate() -> int:
                if greedy:
                    idxs = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
                else:
                    if top_p and top_p > 0.0:
                        # Build nucleus set deterministically by prob order
                        idxs_sorted = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
                        nucleus = []
                        acc = 0.0
                        for j in idxs_sorted:
                            nucleus.append(j)
                            acc += probs[j]
                            if acc >= top_p:
                                break
                        # Sample within nucleus
                        mass = sum(probs[j] for j in nucleus) or 1.0
                        renorm = [probs[j] / mass for j in nucleus]
                        r = self._rng.random()
                        c = 0.0
                        for j, q in zip(nucleus, renorm):
                            c += q
                            if r <= c:
                                return j
                        return nucleus[-1]
                    else:
                        # Sample from top-k, but we'll allow fallback if constraints violate
                        return _sample_from_top_k(probs, min(max(1, top_k), V), self._rng)

                # Greedy or fallback iteration over sorted candidates
                for j in idxs:
                    return j
                return 0

            # Initial pick
            next_id = pick_candidate()
            next_tok = self.inv_vocab.get(next_id, "<unk>")

            # Constraint checks and fallback search
            def violates(candidate: str, idx_generated: int) -> bool:
                # idx_generated: index of the next new token relative to start of generation
                if ban_immediate_repeat and tokens and candidate == tokens[-1]:
                    return True
                if violates_no_repeat(candidate):
                    return True
                # Require first generated token to start with a letter and be at least min_token_len
                if require_alpha_start and idx_generated == 0:
                    if not candidate or not candidate[0].isalpha():
                        return True
                    if min_token_len > 0 and len(candidate) < min_token_len:
                        return True
                # Generic minimum token length (applies to all positions)
                if min_token_len > 0 and len(candidate) < min_token_len:
                    return True
                return False

            if violates(next_tok, new_idx):
                # Try alternatives in descending probability order
                idxs = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
                chosen = None
                for j in idxs:
                    t = self.inv_vocab.get(j, "<unk>")
                    if not violates(t, new_idx):
                        chosen = (j, t)
                        break
                if chosen is not None:
                    next_id, next_tok = chosen  # use best non-violating
                # else keep original choice

            tokens.append(next_tok)
            if stream:
                print(next_tok, end=" ", flush=True)
            if stop_token and next_tok == stop_token:
                break
        return tokens

    def chat_once(
        self,
        history: List[str],
        user_message: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        stop_token: Optional[str],
        greedy: bool = False,
        top_p: float = 0.0,
        stream: bool = False,
        allow_only: Optional[Set[str]] = None,
        ban_tokens: Optional[Set[str]] = None,
        exclude_pad_token: Optional[str] = None,
        min_prob: float = 0.0,
        no_repeat_ngram_size: int = 0,
        ban_immediate_repeat: bool = False,
        allow_alpha_only: bool = False,
        repeat_window: int = 0,
        max_token_repeat: int = 0,
        require_alpha_start: bool = False,
        min_token_len: int = 0,
    ) -> str:
        prompt_text = " ".join(history + [f"User: {user_message}", "Assistant:"])
        prompt_toks = self.tok.encode(prompt_text)
        out_tokens = self.generate(
            prompt_tokens=prompt_toks,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_token=stop_token,
            greedy=greedy,
            stream=stream,
            allow_only=allow_only,
            ban_tokens=ban_tokens,
            exclude_pad_token=exclude_pad_token,
            min_prob=min_prob,
            no_repeat_ngram_size=no_repeat_ngram_size,
            ban_immediate_repeat=ban_immediate_repeat,
            require_alpha_start=require_alpha_start,
            min_token_len=min_token_len,
            repeat_window=repeat_window,
            max_token_repeat=max_token_repeat,
            allow_alpha_only=allow_alpha_only,
        )
        new_toks = out_tokens[len(prompt_toks):]
        reply = self.tok.decode(new_toks).strip()
        if stream:
            print()
        return reply


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Console chatbot using BPETokenizer + TransformerEncoder + trained next-token head."
    )
    parser.add_argument("--head", type=str, required=True, help="Path to trained head JSON.")
    parser.add_argument("--merges", type=str, default="bpe_merges.txt", help="Path to BPE merges.")
    parser.add_argument("--vocab", type=str, default="bpe_vocab.json", help="Path to BPE vocab.")
    parser.add_argument("--dim", type=int, default=32, help="Model/embedding dimension.")
    parser.add_argument("--layers", type=int, default=2, help="Number of encoder layers.")
    parser.add_argument("--heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--ff", type=int, default=64, help="FFN hidden size.")
    parser.add_argument("--add_pe", action="store_true", help="Add sinusoidal positional encodings (must match training).")
    parser.add_argument("--max_new_tokens", type=int, default=32, help="Generation length per turn.")
    parser.add_argument("--temperature", type=float, default=0.9, help="Softmax temperature.")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k sampling.")
    parser.add_argument("--top_p", type=float, default=0.0, help="Nucleus (top-p) sampling cutoff; 0 disables.")
    parser.add_argument("--stop_token", type=str, default="", help="Optional token that stops generation, e.g., '<eos>'.")
    parser.add_argument("--system", type=str, default="", help="Optional system prompt that prefixes the conversation.")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding instead of sampling.")
    parser.add_argument("--stream", action="store_true", help="Stream tokens as they are generated.")
    parser.add_argument("--preset", type=str, default="", choices=["deterministic", "balanced", "creative"], help="Decoding preset: deterministic|balanced|creative.")
    # Post-processing flags
    parser.add_argument("--allow_only", type=str, default="", help="Comma-separated tokens to allow; mask others.")
    parser.add_argument("--ban_tokens", type=str, default="", help="Comma-separated tokens to ban.")
    parser.add_argument("--exclude_pad", action="store_true", help="Exclude pad token by name provided via --pad_token.")
    parser.add_argument("--pad_token", type=str, default="", help="Pad token string to exclude when --exclude_pad is set.")
    parser.add_argument("--min_prob", type=float, default=0.0, help="Minimum probability threshold; lower values zeroed then renormalized.")
    # Repetition controls
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0, help="Disallow repeating contiguous n-grams of this size (0 disables).")
    parser.add_argument("--ban_immediate_repeat", action="store_true", help="Disallow immediately repeating the last generated token.")
    parser.add_argument("--repeat_window", type=int, default=0, help="Window size (in tokens) to track repetition frequency (0 disables).")
    parser.add_argument("--max_token_repeat", type=int, default=0, help="Maximum allowed repeats within the repeat window (0 disables).")
    # Start/length constraints
    parser.add_argument("--require_alpha_start", action="store_true", help="Require the first generated token to start with a letter.")
    parser.add_argument("--min_token_len", type=int, default=0, help="Minimum token length for generated tokens (0 disables).")
    # Strict token set constraint
    parser.add_argument("--allow_alpha_only", action="store_true", help="Mask out non-alphabetic tokens during generation.")
    args = parser.parse_args()

    # Apply decoding presets
    if args.preset == "deterministic":
        args.greedy = True
        args.temperature = 0.7
        args.top_k = max(1, args.top_k)
        args.top_p = 0.0
        if not args.ban_tokens.strip():
            args.ban_tokens = "<unk>"
        if not args.exclude_pad:
            args.exclude_pad = True
        if not args.pad_token.strip():
            args.pad_token = "<pad>"
        if args.min_prob <= 0.0:
            args.min_prob = 0.0
        # Repetition/cleanliness controls for cleaner output
        if args.no_repeat_ngram_size <= 0:
            args.no_repeat_ngram_size = 3
        args.ban_immediate_repeat = True
        args.require_alpha_start = True
        if args.min_token_len <= 0:
            args.min_token_len = 2
        # Frequency penalty in recent window
        if args.repeat_window <= 0:
            args.repeat_window = 32
        if args.max_token_repeat <= 0:
            args.max_token_repeat = 2
    elif args.preset == "balanced":
        args.greedy = False
        args.temperature = 0.9
        args.top_k = 20
        args.top_p = 0.9
        if not args.ban_tokens.strip():
            args.ban_tokens = "<unk>"
        if not args.exclude_pad:
            args.exclude_pad = True
        if not args.pad_token.strip():
            args.pad_token = "<pad>"
        if args.min_prob <= 0.0:
            args.min_prob = 0.001
        # Mild repetition/cleanliness control
        if args.no_repeat_ngram_size <= 0:
            args.no_repeat_ngram_size = 3
        args.ban_immediate_repeat = True
        args.require_alpha_start = True
        if args.min_token_len <= 0:
            args.min_token_len = 2
        if args.repeat_window <= 0:
            args.repeat_window = 32
        if args.max_token_repeat <= 0:
            args.max_token_repeat = 2
    elif args.preset == "creative":
        args.greedy = False
        args.temperature = 1.1
        args.top_k = 0
        args.top_p = 0.92
        if not args.ban_tokens.strip():
            args.ban_tokens = "<unk>"
        # leave exclude_pad/min_prob unchanged for maximum diversity
        # Optional light repetition control
        if args.no_repeat_ngram_size <= 0:
            args.no_repeat_ngram_size = 2
        if args.repeat_window <= 0:
            args.repeat_window = 16
        if args.max_token_repeat <= 0:
            args.max_token_repeat = 3

    bot = Chatbot(
        merges_path=args.merges,
        vocab_path=args.vocab,
        head_path=args.head,
        dim=args.dim,
        layers=args.layers,
        heads=args.heads,
        ff=args.ff,
        add_pe=args.add_pe,
    )

    history: List[str] = []
    if args.system.strip():
        history.append(f"System: {args.system.strip()}")

    print("Chatbot ready. Type your message and press Enter. Ctrl+C to exit.")
    stop_tok = args.stop_token.strip() or None
    allow_only = {t.strip() for t in args.allow_only.split(",") if t.strip()} if args.allow_only.strip() else None
    ban_tokens = {t.strip() for t in args.ban_tokens.split(",") if t.strip()} if args.ban_tokens.strip() else None
    exclude_pad_token = (args.pad_token.strip() or None) if args.exclude_pad else None

    try:
        while True:
            user = input("You: ").strip()
            if not user:
                continue
            reply = bot.chat_once(
                history=history,
                user_message=user,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                stop_token=stop_tok,
                greedy=args.greedy,
                top_p=args.top_p,
                stream=args.stream,
                allow_only=allow_only,
                ban_tokens=ban_tokens,
                exclude_pad_token=exclude_pad_token,
                min_prob=args.min_prob,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                ban_immediate_repeat=args.ban_immediate_repeat,
                allow_alpha_only=args.allow_alpha_only,
                repeat_window=args.repeat_window,
                max_token_repeat=args.max_token_repeat,
                require_alpha_start=args.require_alpha_start,
                min_token_len=args.min_token_len,
            )
            print(f"Assistant: {reply}")
            history.append(f"User: {user}")
            history.append(f"Assistant: {reply}")
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")


if __name__ == "__main__":
    main()