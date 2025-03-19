import re
from string import ascii_lowercase
from collections import defaultdict
import torch

# TODO add CTC decode
# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = ""
    EMPTY_IND = 0

    def __init__(self, alphabet=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        """
        Decodes a sequence of token indices using CTC (Connectionist Temporal Classification).

        This function removes repeated tokens and empty tokens to obtain the decoded text.

        Args:
            inds (list): A list of token indices.

        Returns:
            str: The decoded text string.
        """
        decoded = []
        last_char_ind = self.EMPTY_IND  # Initialize with the empty token index

        for ind in inds:
            if ind == last_char_ind:  # Skip repeated tokens
                continue
            if ind != self.EMPTY_IND:  # Append non-empty tokens
                decoded.append(self.ind2char[ind])
                last_char_ind = ind

        return "".join(decoded)


    def expand_and_merge_path(self, dp, next_token_prob):
        """
        Expands and merges paths in the beam search based on the next token probabilities.

        This function iterates through the current paths in `dp` and extends them with the
        possible next tokens. It merges paths with the same prefix and updates their
        probabilities.

        Args:
            dp (dict): A dictionary representing the current paths in the beam search.
                         Keys are tuples (prefix, last_char), where prefix is the decoded string so far
                         and last_char is the last character added.  Values are probabilities.
            next_token_prob (list): Probabilities of the next tokens.

        Returns:
            dict: A dictionary representing the expanded and merged paths.
        """
        new_dp = defaultdict(float)
        for ind, prob in enumerate(next_token_prob): # Iterate over token probabilities with indices
            current_char = self.ind2char[ind]  # Map index to character
            for (prefix, last_char), v in dp.items():  # Iterate through current paths
                if last_char == current_char:
                    new_prefix = prefix  # Merge paths if the last character is the same
                else:
                    if current_char != self.EMPTY_TOK:
                        new_prefix = prefix + current_char  # Extend prefix with non-empty token
                    else:
                        new_prefix = prefix  # Keep prefix unchanged for empty token
                new_dp[(new_prefix, current_char)] += v * prob  # Update probability

        return new_dp


    def truncate_paths(self, dp, beam_size):
        """
        Truncates the paths in the beam search to keep only the top `beam_size` paths.

        Args:
            dp (dict): A dictionary representing the current paths in the beam search.
            beam_size (int): The maximum number of paths to keep.

        Returns:
            dict: A dictionary containing the top `beam_size` paths.
        """
        return dict(sorted(list(dp.items()), key=lambda x: -x[1])[:beam_size])


    def ctc_beam_search(self, probs, beam_size):
        """
        Performs CTC beam search decoding.

        Args:
            probs (list): A list of probability distributions over tokens for each time step.
            beam_size (int): The beam size to use.

        Returns:
            list: A list of tuples (prefix, probability), representing the decoded prefixes and their probabilities.
                 Probabilities are normalized by length.
        """
        dp = {
            ('', self.EMPTY_TOK): 1.0,  # Initialize with empty prefix and probability 1.0
        }
        for prob in probs:
            dp = self.expand_and_merge_path(dp, prob)  # Expand and merge paths
            dp = self.truncate_paths(dp, beam_size)      # Truncate to keep top paths

        # Normalize probabilities by length and sort
        dp = [
            (prefix, proba / len(prefix) if len(prefix) > 0 else proba)  # Length normalization
            for (prefix, _), proba in sorted(dp.items(), key=lambda x: -x[1])
        ]
        return dp 


    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
