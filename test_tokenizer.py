
import unittest

from tokenizer import BPETokenizer


class TestBPETokenizer(unittest.TestCase):
    def setUp(self):
        self.training_texts = [
            "the cat sat on the mat",
            "the cat sat on the floor",
            "the dog sat on the mat",
        ]
        self.tokenizer = BPETokenizer(vocab_size=50)
        self.tokenizer.train(self.training_texts)

    def test_encode_decode_round_trip(self):
        text = "the cat sat on the mat"
        self.assertEqual(
            self.tokenizer.decode(self.tokenizer.encode(text)),
            text,
        )

    def test_empty_text(self):
        self.assertEqual(self.tokenizer.encode(""), [])

    def test_unknown_characters_round_trip(self):
        text = "你好!"
        self.assertEqual(
            self.tokenizer.decode(self.tokenizer.encode(text)),
            text,
        )


if __name__ == "__main__":
    unittest.main()
