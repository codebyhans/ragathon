import pytest
from ragathon.chunking.token import NaiveTokenBasedTextSplitter


class TestNaiveTokenBasedTextSplitter:
    @pytest.fixture
    def splitter(self) -> NaiveTokenBasedTextSplitter:
        return NaiveTokenBasedTextSplitter(max_token_size=5, overlap=2)

    def test_empty_text(self, splitter: NaiveTokenBasedTextSplitter) -> None:
        """Test that an empty text is handled correctly."""
        result = splitter.split_text("")
        assert result == []

    def test_single_chunk(self, splitter: NaiveTokenBasedTextSplitter) -> None:
        """Test handling of text that fits into a single chunk."""
        text = "This is a short text."
        result = splitter.split_text(text)
        assert result == ["This is a short text."]

    def test_multiple_chunks(self, splitter: NaiveTokenBasedTextSplitter) -> None:
        """Test handling of text that requires multiple chunks."""
        text = "This is a longer text that should be split into multiple chunks."
        result = splitter.split_text(text)
        assert result == [
            "This is a longer text",
            "longer text that should be",
            "should be split into multiple",
            "into multiple chunks.",
        ]

    def test_overlap(self, splitter: NaiveTokenBasedTextSplitter) -> None:
        """Test that chunks overlap correctly."""
        text = "One two three four five six seven eight nine ten."
        result = splitter.split_text(text)
        assert "four five" in result[0] and "four five" in result[1]

    def test_max_token_size(self) -> None:
        """Test that max_token_size is respected."""
        splitter = NaiveTokenBasedTextSplitter(max_token_size=3, overlap=1)
        text = "One two three four five six."
        result = splitter.split_text(text)
        assert len(result) == 3
        assert all(len(chunk.split()) <= 3 for chunk in result)

    def test_overlap_size(self) -> None:
        """Test that overlap size is respected."""
        splitter = NaiveTokenBasedTextSplitter(max_token_size=4, overlap=2)
        text = "One two three four five six seven eight."
        result = splitter.split_text(text)
        assert "three four" in result[0] and "three four" in result[1]

    def test_long_words(self, splitter: NaiveTokenBasedTextSplitter) -> None:
        """Test handling of very long words."""
        text = "This supercalifragilisticexpialidocious word is very long."
        result = splitter.split_text(text)
        assert "supercalifragilisticexpialidocious" in result[0]

    def test_punctuation(self, splitter: NaiveTokenBasedTextSplitter) -> None:
        """Test handling of text with punctuation."""
        text = "Hello, world! How are you? I'm fine, thank you."
        result = splitter.split_text(text)
        assert result[0] == "Hello, world! How are you?"

    def test_newlines(self, splitter: NaiveTokenBasedTextSplitter) -> None:
        """Test handling of text with newlines."""
        text = "This is the first line.\nThis is the second line.\nAnd the third."
        result = splitter.split_text(text)
        assert result == [
            "This is the first line.",
            "first line. This is the",
            "is the second line. And",
            "line. And the third.",
        ]

    @pytest.mark.parametrize(
        "max_token_size, overlap",
        [
            (10, 3),
            (7, 2),
            (5, 1),
            (3, 0),
        ],
    )
    def test_various_configurations(self, max_token_size: int, overlap: int) -> None:
        """Test the splitter with various configurations."""
        splitter = NaiveTokenBasedTextSplitter(
            max_token_size=max_token_size, overlap=overlap
        )
        text = (
            "This is a test sentence to check various configurations of the splitter."
        )
        result = splitter.split_text(text)
        assert len(result) > 0
        assert all(len(chunk.split()) <= max_token_size for chunk in result)

    def test_consistency(self, splitter: NaiveTokenBasedTextSplitter) -> None:
        """Test that the splitter produces consistent results."""
        text = "This is a test sentence to check consistency of the splitter."
        result1 = splitter.split_text(text)
        result2 = splitter.split_text(text)
        assert result1 == result2
