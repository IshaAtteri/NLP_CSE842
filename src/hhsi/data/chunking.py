from transformers import AutoTokenizer
import spacy
import re

MODEL_NAME = "BAAI/bge-base-en-v1.5"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("Document text must be a string")

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

def split_sentences(text: str) -> list[str]:
    doc = nlp(text)
    return [
        sent.text.strip()
        for sent in doc.sents
        if sent.text.strip()
    ]

def split_long_sentence(sentence: str, max_tokens: int = 256) -> list[str]:
    token_budget = max_tokens - tokenizer.num_special_tokens_to_add(pair=False)

    if token_budget <= 0:
        raise ValueError("max_tokens is too small")

    if len(tokenizer.encode(sentence, add_special_tokens=True, verbose=False)) <= max_tokens:
        return [sentence]

    words = sentence.split()
    pieces = []
    current = []

    for word in words:
        candidate = " ".join(current + [word])

        if len(tokenizer.encode(candidate)) > max_tokens:
            if current:
                pieces.append(" ".join(current))
                current = []

            if len(tokenizer.encode(word)) > max_tokens:
                raise ValueError(f"Single word exceeds token limit: {word[:50]}")

        current.append(word)

    if current:
        pieces.append(" ".join(current))

    return pieces

def chunk_text_with_spans(text: str, max_tokens: int = 256, overlap_tokens: int = 32) -> list[dict]:
    """Chunk with half-open offsets in the non-whitespace source character stream."""

    if max_tokens <= tokenizer.num_special_tokens_to_add(pair=False):
        raise ValueError("max_tokens is too small")

    if not 0 <= overlap_tokens < max_tokens:
        raise ValueError("Invalid overlap_tokens")

    text = normalize_text(text)
    sentences = [piece for sentence in split_sentences(text) for piece in split_long_sentence(sentence, max_tokens)]

    canonical = "".join(text.split())
    units = ["".join(sentence.split()) for sentence in sentences]
    if "".join(units) != canonical:
        raise ValueError("Sentence splitting changed non-whitespace source content")
    offsets = [0]
    for unit in units:
        offsets.append(offsets[-1] + len(unit))

    token_budget = max_tokens - tokenizer.num_special_tokens_to_add(pair=False)

    chunks = []
    start = 0

    while start < len(sentences):
        current = []
        end = start

        while end < len(sentences):
            candidate = " ".join(current + [sentences[end]])

            if len(tokenizer.encode(candidate)) > max_tokens:
                break

            current.append(sentences[end])
            end += 1

        if not current:
            raise ValueError(
                f"Sentence {start} exceeds the {token_budget}-token budget"
            )

        chunks.append({"text": " ".join(current),
                       "source_start": offsets[start], "source_end": offsets[end]})

        if end == len(sentences):
            break

        new_start = end
        overlap_count = 0

        while new_start > start + 1:
            sentence = sentences[new_start - 1]
            count = len(
                tokenizer.encode(sentence, add_special_tokens=False)
            )

            if overlap_count + count > overlap_tokens:
                break

            overlap_count += count
            new_start -= 1

        # Reduce overlap until the next unseen unit fits; never emit overlap alone.
        while new_start < end and len(tokenizer.encode(" ".join(sentences[new_start:end + 1]))) > max_tokens:
            new_start += 1
        start = new_start

    return chunks

def chunk_text(text: str, max_tokens: int = 256, overlap_tokens: int = 32) -> list[str]:
    """Backward-compatible text-only interface."""
    return [chunk["text"] for chunk in chunk_text_with_spans(text, max_tokens, overlap_tokens)]


if __name__ == "__main__":
    text = "Sam Bankman-Fried founded FTX, a cryptocurrency exchange."

    long_sentence = (
        "The company reported significant developments in its operations, "
        * 80
    ).strip() + "."

    chunks = chunk_text(long_sentence)

    print("Number of chunks:", len(chunks))

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}: {len(tokenizer.encode(chunk))} tokens")




