
# LLM Fundamentals

Learning how Large Language Models work by building
small components from scratch using Python.

## Roadmap

- [x] 01 - Tokens: BPE tokenizer
- [ ] 02 - Embeddings
- [ ] 03 - Inference
- [ ] 04 - Context Window
- [ ] 05 - Retrieval-Augmented Generation (RAG)
- [ ] 06 - Prompting and Evaluation

## Project 01: Tokens

A simple Byte Pair Encoding (BPE) tokenizer implemented
without external libraries.

### What it demonstrates

- How frequent character pairs are merged into tokens.
- How the vocabulary size affects tokenization.
- How an English-trained tokenizer handles German text.
- How text is encoded into tokens and decoded again.

## Run the project

```bash
python demo.py
```

## Limitations

This is an educational character-level BPE implementation.
It is not a byte-level production tokenizer and does not
necessarily match the token counts of commercial LLMs.
