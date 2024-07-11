import re
from typing import List

from ragathon.chunking.sentence import SentenceSplitter


class ParagraphSplitter:
    def __init__(
        self,
        max_chunk_size: int,
        sentence_splitter: SentenceSplitter,
    ) -> None:
        """Initialize a paragraph splitter optimized for Danish text.

        Args:
            max_chunk_size (int, optional): The maximum number of chunks in a paragraph. Defaults to 128.
            sentence_splitter (Optional[SentenceSplitter], optional): Sentence splitter.
                If None then `DanishSentenceSplitter` is used.
        """

        self._max_chunk_size: int = max_chunk_size
        self._sentence_splitter: SentenceSplitter = sentence_splitter

    def split_text(self, text: str) -> List[str]:
        """Split text into paragraphs.

        Args:
            text (str): Text to split.

        Returns:
            List[str]: List of paragraphs.
        """

        paragraphs = self._split_into_paragraphs(text=text)
        paragraphs = self._split_large_paragraphs_into_smaller_ones(
            paragraphs=paragraphs,
        )

        return paragraphs

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs.

        Args:
            text (str): Text to split.

        Returns:
            List[str]: List of paragraphs.
        """

        # Remove leading and trailing whitespace and newline characters
        text = text.strip(" \n")

        # Remove whitespaces from lines that only contain whitespace
        text = re.sub(r"\n\s+\n", "\n\n", text)

        raw_paragraphs = text.split("\n\n")

        return [
            raw_paragraph.strip()
            for raw_paragraph in raw_paragraphs
            if len(raw_paragraph.strip()) > 0
        ]

    def _split_large_paragraphs_into_smaller_ones(
        self,
        paragraphs: List[str],
    ) -> List[str]:
        """Split large paragraphs into smaller ones that do not exceed a fixed number of chunks.

        Args:
            paragraphs (List[str]): List of paragraphs.
            max_chunk_size (int): The maximum number of chunks in a paragraph.

        Returns:
            List[str]: List of paragraphs that do not exceed the maximum number of chunks.
        """

        result = []

        for paragraph in paragraphs:
            # We only split when the `max_chunk_size` is exceeded. As such, we start by
            # adding the paragraph to the result if it is already small enough.
            if len(paragraph.split(" ")) <= self._max_chunk_size:
                result.append(paragraph)
                continue

            # At this point, we know that the paragraph is too large and needs to be split.
            # Reset the current paragraph
            current_paragraph = ""
            current_paragraph_chunk_count = 0

            # We split based on sentences because we want to avoid splitting a sentence in the middle.
            sentences = self._sentence_splitter.split_text(text=paragraph)
            for sentence in sentences:
                sentence_chunk_count = len(sentence.split(" "))
                chunk_count_after_adding_sentence = (
                    current_paragraph_chunk_count + sentence_chunk_count
                )

                # If adding the next sentence would exceed the maximum chunk size,
                # then add the current paragraph to the result and reset the current paragraph.
                if (
                    chunk_count_after_adding_sentence > self._max_chunk_size
                    and current_paragraph_chunk_count > 0
                ):
                    result.append(current_paragraph.strip())
                    current_paragraph = ""
                    current_paragraph_chunk_count = 0
                current_paragraph += f"{sentence} "
                current_paragraph_chunk_count += sentence_chunk_count

            # Ensure any added spaces are removed.
            current_paragraph = current_paragraph.strip()

            if len(current_paragraph) > 0:
                # Add the last paragraph
                result.append(current_paragraph)

        return result
