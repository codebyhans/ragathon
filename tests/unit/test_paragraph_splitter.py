import re
from typing import List

import pytest
from ragathon.chunking.paragraph import ParagraphSplitter


class MockSentenceSplitter:
    def split_text(self, text: str) -> List[str]:
        # Split on '. ' but keep the periods
        sentences = re.split(r"(?<=\.)\s", text)
        # Ensure the last sentence has a period if the original text ended with one
        if text.endswith(".") and sentences and not sentences[-1].endswith("."):
            sentences[-1] += "."
        return sentences


class TestParagraphSplitter:
    @pytest.fixture
    def default_splitter(self) -> ParagraphSplitter:
        return ParagraphSplitter(
            max_chunk_size=128, sentence_splitter=MockSentenceSplitter()
        )

    @pytest.fixture
    def custom_splitter(self) -> ParagraphSplitter:
        return ParagraphSplitter(
            max_chunk_size=5, sentence_splitter=MockSentenceSplitter()
        )

    def test_empty_input(self, default_splitter: ParagraphSplitter) -> None:
        assert default_splitter.split_text("") == []

    def test_only_newlines(self, default_splitter: ParagraphSplitter) -> None:
        assert default_splitter.split_text("\n\n\n") == []

    def test_split_into_paragraphs(self, default_splitter: ParagraphSplitter) -> None:
        text = """København er Danmarks hovedstad. Byen ligger på Sjælland.

Århus er den næststørste by. Den ligger i Jylland.

Nyt afsnit her. Æbler og pærer er frugter.

Nyt afsnit. Østers og rejer er skaldyr.

Sidste afsnit her. Åen løber gennem skoven."""
        result = default_splitter.split_text(text=text)
        assert result == [
            "København er Danmarks hovedstad. Byen ligger på Sjælland.",
            "Århus er den næststørste by. Den ligger i Jylland.",
            "Nyt afsnit her. Æbler og pærer er frugter.",
            "Nyt afsnit. Østers og rejer er skaldyr.",
            "Sidste afsnit her. Åen løber gennem skoven.",
        ]

    def test_remove_extraneous_whitespace(
        self, default_splitter: ParagraphSplitter
    ) -> None:
        text = """

        Første afsnit med ekstra mellemrum i starten og slutningen.






    Andet afsnit med ekstra linjer før og efter.


        Tredje afsnit med blandet ekstra whitespace.

        """

        expected_result = [
            "Første afsnit med ekstra mellemrum i starten og slutningen.",
            "Andet afsnit med ekstra linjer før og efter.",
            "Tredje afsnit med blandet ekstra whitespace.",
        ]

        result = default_splitter.split_text(text)
        assert result == expected_result

    def test_split_into_paragraphs_with_empty_lines(
        self, default_splitter: ParagraphSplitter
    ) -> None:
        text = "Første afsnit.\n\n\nAndet afsnit.\n\n\n\nTredje afsnit."
        result = default_splitter.split_text(text=text)
        assert result == ["Første afsnit.", "Andet afsnit.", "Tredje afsnit."]

    def test_split_large_paragraph(self) -> None:
        custom_splitter = ParagraphSplitter(
            max_chunk_size=5, sentence_splitter=MockSentenceSplitter()
        )

        large_paragraph = "En. To. Tre. Fire. Fem. Seks. Syv. Otte. Ni. Ti."
        result = custom_splitter.split_text(large_paragraph)
        assert result == ["En. To. Tre. Fire. Fem.", "Seks. Syv. Otte. Ni. Ti."]

    def test_mixed_paragraph_sizes(self, custom_splitter: ParagraphSplitter) -> None:
        text = "Kort afsnit.\n\nEn. To. Tre. Fire. Fem. Seks. Syv. Otte.\n\nEndnu et kort afsnit."
        result = custom_splitter.split_text(text=text)
        assert result == [
            "Kort afsnit.",
            "En. To. Tre. Fire. Fem.",
            "Seks. Syv. Otte.",
            "Endnu et kort afsnit.",
        ]

    def test_very_long_sentence(self, custom_splitter: ParagraphSplitter) -> None:
        long_sentence = "Dette er en meget lang sætning, der overstiger den maksimale størrelse for en chunk, men som ikke bør opdeles."
        result = custom_splitter.split_text(text=long_sentence)
        print(f"result: {result}")
        assert result == [long_sentence]

    @pytest.mark.parametrize("max_chunk_size", [1, 3, 5, 10])
    def test_different_max_chunk_sizes(self, max_chunk_size: int) -> None:
        splitter = ParagraphSplitter(
            max_chunk_size=max_chunk_size, sentence_splitter=MockSentenceSplitter()
        )
        text = "En. To. Tre. Fire. Fem. Seks. Syv. Otte. Ni. Ti."
        result = splitter.split_text(text=text)
        assert all(len(paragraph.split()) <= max_chunk_size for paragraph in result)

    def test_preserve_paragraph_breaks(
        self, default_splitter: ParagraphSplitter
    ) -> None:
        text = "Afsnit 1 linje 1\nAfsnit 1 linje 2\n\nAfsnit 2\n\nAfsnit 3 linje 1\nAfsnit 3 linje 2"
        result = default_splitter.split_text(text)
        assert result == [
            "Afsnit 1 linje 1\nAfsnit 1 linje 2",
            "Afsnit 2",
            "Afsnit 3 linje 1\nAfsnit 3 linje 2",
        ]
