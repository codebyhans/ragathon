from typing import List


class NaiveTokenBasedTextSplitter:
    """Naive token-based text splitter."""

    def __init__(self, max_token_size: int, overlap: int) -> None:
        """Initialize the splitter.

        Args:
            max_token_size (int): Maximum token size.
            overlap (int): Overlap between chunks.
        """

        self._max_token_size: int = max_token_size
        self._overlap: int = overlap

    def split_text(self, text: str) -> List[str]:
        chunks: List[str] = []

        tokens = text.split()

        start = 0
        while start < len(tokens):
            end = start + self._max_token_size
            chunk = " ".join(tokens[start:end])
            if len(chunk) < 2:
                break

            chunks.append(chunk)

            # Break if we have reached the end of the tokens
            if start + self._max_token_size >= len(tokens):
                break

            # Move the start pointer for the next chunk
            start += self._max_token_size - self._overlap

        return chunks
