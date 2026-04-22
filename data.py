"""
================================================================================
 DATA PIPELINE — Tokenizer and DataLoader (From Scratch)
================================================================================
A simple character-level tokenizer and batch data loader for training.

Character-level tokenization is the simplest form of tokenization:
each unique character in the text becomes its own token. This means:
- Vocabulary is small (~50-70 unique characters for English text)
- No need for complex subword algorithms (BPE, WordPiece, etc.)
- The model must learn to "spell" words character by character
- Perfect for learning/demonstration purposes

Real LLMs use subword tokenizers (BPE) with 32K-100K tokens,
but the principles are identical — just with a larger vocabulary.
================================================================================
"""
import numpy as np


class CharTokenizer:
    """
    Character-Level Tokenizer: Maps each unique character to an integer ID.

    Example:
        text = "hello"
        chars = ['e', 'h', 'l', 'o']   — sorted unique characters
        encode("hello") → [1, 0, 2, 2, 3]
        decode([1, 0, 2, 2, 3]) → "hello"
    """

    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        self._vocab_size = 0

    def fit(self, text):
        """
        Build vocabulary from text.

        Sorts characters for deterministic ordering, then creates
        bidirectional mappings (char→id and id→char).
        """
        chars = sorted(set(text))
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char = {i: ch for ch, i in self.char_to_id.items()}
        self._vocab_size = len(chars)
        print(f"[Tokenizer] Vocabulary: {self._vocab_size} unique characters")
        print(f"[Tokenizer] Characters: {repr(''.join(chars))}")

    @property
    def vocab_size(self):
        return self._vocab_size

    def encode(self, text):
        """Convert text string → list of integer token IDs."""
        return [self.char_to_id[ch] for ch in text]

    def decode(self, ids):
        """Convert list of integer token IDs → text string."""
        return ''.join(self.id_to_char[i] for i in ids)


class DataLoader:
    """
    Batch Data Loader for Next-Token Prediction.

    Given an encoded text (array of token IDs), produces random batches
    of (input, target) pairs for training.

    =========================================================================
    NEXT-TOKEN PREDICTION SETUP:
    =========================================================================
    If the text is: "the cat sat"
    And seq_length = 4:

        Input (X):   [t, h, e, ' ']     → positions 0, 1, 2, 3
        Target (Y):  [h, e, ' ', c]     → positions 1, 2, 3, 4

    Y is X shifted RIGHT by 1. At each position, the model learns to
    predict the NEXT character given all previous characters.

    This is how GPT-style models learn: by predicting what comes next,
    they implicitly learn grammar, facts, reasoning patterns, etc.
    =========================================================================
    """

    def __init__(self, data, batch_size, seq_length):
        """
        Args:
            data (list[int]): Encoded text as token IDs
            batch_size (int): Number of sequences per batch
            seq_length (int): Length of each sequence
        """
        self.data = np.array(data, dtype=np.int64)
        self.batch_size = batch_size
        self.seq_length = seq_length

        # Validate we have enough data
        assert len(self.data) > seq_length + 1, \
            f"Text too short ({len(self.data)} chars) for seq_length={seq_length}"

        print(f"[DataLoader] Total tokens: {len(self.data)}")
        print(f"[DataLoader] Batch size: {batch_size}, Seq length: {seq_length}")

    def get_batch(self):
        """
        Sample a random batch of (input, target) pairs.

        Returns:
            X (np.ndarray): Input sequences,  shape (batch_size, seq_length)
            Y (np.ndarray): Target sequences, shape (batch_size, seq_length)

        Each sequence starts at a random position in the text.
        Y[i] = X[i] shifted right by 1 character.
        """
        # Maximum valid starting position (need seq_length + 1 chars for X and Y)
        max_start = len(self.data) - self.seq_length - 1

        # Random starting positions for each sequence in the batch
        starts = np.random.randint(0, max_start, size=self.batch_size)

        # Build X and Y arrays
        X = np.array([self.data[s : s + self.seq_length] for s in starts])
        Y = np.array([self.data[s + 1 : s + 1 + self.seq_length] for s in starts])

        return X, Y
