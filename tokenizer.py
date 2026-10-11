
class BPETokenizer:
    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
        self.merges = []
        self.vocab = set()

    def train(self, texts):
        # Start with characters, including spaces.
        sequences = [list(text) for text in texts]
        self.vocab = {
            token
            for sequence in sequences
            for token in sequence
        }

        while len(self.vocab) < self.vocab_size:
            pair_counts = {}

            # Count adjacent pairs across all training texts.
            for sequence in sequences:
                for i in range(len(sequence) - 1):
                    pair = (sequence[i], sequence[i + 1])
                    pair_counts[pair] = (
                        pair_counts.get(pair, 0) + 1
                    )

            if not pair_counts:
                break

            # Find the most frequent pair.
            best_pair = max(
                pair_counts,
                key=pair_counts.get
            )

            # Stop if no pair appears more than once.
            if pair_counts[best_pair] < 2:
                break

            merged_token = best_pair[0] + best_pair[1]
            self.merges.append(best_pair)

            # Replace occurrences of the selected pair.
            new_sequences = []

            for sequence in sequences:
                merged_sequence = []
                i = 0

                while i < len(sequence):
                    if (
                        i < len(sequence) - 1
                        and (sequence[i], sequence[i + 1])
                        == best_pair
                    ):
                        merged_sequence.append(merged_token)
                        i += 2
                    else:
                        merged_sequence.append(sequence[i])
                        i += 1

                new_sequences.append(merged_sequence)

            sequences = new_sequences
            self.vocab.add(merged_token)

        return self

    def encode(self, text):
        tokens = list(text)

        # Apply learned merges in training order.
        for best_pair in self.merges:
            merged_token = best_pair[0] + best_pair[1]
            result = []
            i = 0

            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and (tokens[i], tokens[i + 1])
                    == best_pair
                ):
                    result.append(merged_token)
                    i += 2
                else:
                    result.append(tokens[i])
                    i += 1

            tokens = result

        return tokens

    def decode(self, tokens):
        return "".join(tokens)
