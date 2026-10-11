
from tokenizer import BPETokenizer

# Training data: English only.
training_texts = [
    "the cat is sitting on the mat",
    "the dog is sitting on the floor",
    "the cat and the dog are playing",
    "machine learning models process text",
    "language models learn patterns in language",
    "the machine is learning from text",
    "natural language processing uses tokens",
    "the cat is playing with the dog",
    "Die Katze sitzt auf der Matte",
"Der Hund sitzt auf dem Boden",
"Die Katze und der Hund spielen",
"Sprachmodelle lernen Muster in Sprache",
]

tokenizer = BPETokenizer(vocab_size=100)
tokenizer.train(training_texts)

english = "The cat is sitting on the mat."
german = "Die Katze sitzt auf der Matte."

for language, sentence in [
    ("English", english),
    ("German", german),
]:
    tokens = tokenizer.encode(sentence)

    print(f"\n{language}: {sentence}")
    print(f"Tokens: {tokens}")
    print(f"Token count: {len(tokens)}")
    print(f"Decoded: {tokenizer.decode(tokens)}")

# Experiment with vocabulary size.
print("\n--- Vocabulary experiment ---")

for size in [30, 50, 100, 150]:
    model = BPETokenizer(vocab_size=size)
    model.train(training_texts)

    print(
        f"Vocabulary target: {size:3} | "
        f"English tokens: {len(model.encode(english)):3} | "
        f"German tokens: {len(model.encode(german)):3}"
    )
