import os
import re
from typing import List, Optional, Protocol

import nltk
from loguru import logger
from nltk.tokenize.punkt import PunktSentenceTokenizer


class SentenceSplitter(Protocol):
    """Interface for sentence splitters."""

    def split_text(self, text: str) -> List[str]:
        """Split text into sentences.

        Args:
            text (str): Text to split.

        Returns:
            List[str]: List of sentences.
        """
        ...


def ensure_nltk_data_is_loaded() -> None:
    """Load NLTK data."""

    nltk_data_dir = os.environ.get("NLTK_DATA")
    if not nltk_data_dir:
        nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk-data")

    # update nltk path for nltk so that it finds the data
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.append(nltk_data_dir)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        try:
            nltk.download("punkt", download_dir=nltk_data_dir)
        except FileExistsError:
            logger.info(
                "Tried to re-download NLTK files but they already exists. "
                "This could happen in multi-threaded deployments, should be benign"
            )


class DanishSentenceSplitter:
    """Class for splitting Danish text into sentences."""

    def __init__(self) -> None:
        ensure_nltk_data_is_loaded()

        self._tokenizer: PunktSentenceTokenizer = nltk.data.load(
            resource_url="tokenizers/punkt/danish.pickle"
        )

    def split_text(self, text: str) -> List[str]:
        """Split text into sentences.

        Args:
            text (str): Text to split.

        Returns:
            List[str]: List of sentences.
        """

        preprocessed_text = self._clean_text(text=text)
        raw_sentences = self._tokenize_sentences(text=preprocessed_text)
        new_sentences = self._fix_not_capitalized_sentences(raw_sentences=raw_sentences)
        new_sentences = self._fix_orphaned_list_start(raw_sentences=new_sentences)

        return new_sentences

    def _clean_text(self, text: str) -> str:
        """Cleans text by removing unwanted artifacts.

        Args:
            text (str): Text to preprocess.

        Returns:
            str: Preprocessed text.
        """
        return text.replace("..", ".")

    def _tokenize_sentences(self, text: str) -> List[str]:
        """Tokenize text into sentences.

        Args:
            text (str): Text to tokenize.

        Returns:
            List[str]: List of sentences.
        """
        new_sentences = []

        raw_sentences = self._tokenizer.tokenize(text=text)

        for raw_sentence in raw_sentences:
            parts = raw_sentence.split("\n")
            if len(parts) > 1:
                for part in parts:
                    if len(part.strip()) > 0:
                        new_sentences.append(part.strip())
            else:
                if len(raw_sentence.strip()) > 0:
                    new_sentences.append(raw_sentence.strip())
        return new_sentences

    def _fix_not_capitalized_sentences(self, raw_sentences: List[str]) -> List[str]:
        """Fix sentences that are not capitalized.

        Args:
            raw_sentences (List[str]): List of sentences.

        Returns:
            List[str]: List of sentences.
        """

        new_sentences = []

        for raw_sentence in raw_sentences:
            if raw_sentence[0].capitalize() != raw_sentence[0]:
                # if the first letter is not capitalized, it is not a new sentence
                if len(new_sentences) > 0:
                    new_sentences[-1] += " " + raw_sentence
                else:
                    new_sentences.append(raw_sentence)
            else:
                new_sentences.append(raw_sentence)

        return new_sentences

    def _fix_orphaned_list_start(self, raw_sentences: List[str]) -> List[str]:
        """Fix orphaned list start.

        Args:
            raw_sentences (List[str]): List of sentences.

        Returns:
            List[str]: List of sentences.
        """
        new_sentences = []

        prev_raw_sentence: Optional[str] = None
        for raw_sentence in raw_sentences:
            if prev_raw_sentence is None:
                prev_raw_sentence = raw_sentence.strip()
                continue
            orpahned_list_start_match = re.search(r"^\d+\.$", prev_raw_sentence)
            if orpahned_list_start_match:
                new_sentences.append(f"{prev_raw_sentence} {raw_sentence}")
                prev_raw_sentence = None
            else:
                new_sentences.append(prev_raw_sentence)
                prev_raw_sentence = raw_sentence.strip()

        if prev_raw_sentence is not None and len(prev_raw_sentence) > 0:
            new_sentences.append(prev_raw_sentence)

        return new_sentences
