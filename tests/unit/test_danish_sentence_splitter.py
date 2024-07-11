import pytest
from ragathon.chunking.sentence import DanishSentenceSplitter


class TestDanishSentenceSplitter:
    @pytest.fixture
    def splitter(self) -> DanishSentenceSplitter:
        return DanishSentenceSplitter()

    def test_empty_text(self, splitter: DanishSentenceSplitter) -> None:
        """Test that an empty text is handled correctly."""
        result = splitter.split_text("")
        assert result == []

    def test_single_sentence(self, splitter: DanishSentenceSplitter) -> None:
        """Test handling of a single sentence."""
        text = "Dette er en sætning."
        result = splitter.split_text(text)
        assert result == ["Dette er en sætning."]

    def test_multiple_sentences(self, splitter: DanishSentenceSplitter) -> None:
        """Test handling of multiple sentences."""
        text = "Dette er den første sætning. Dette er den anden sætning."
        result = splitter.split_text(text)
        assert result == ["Dette er den første sætning.", "Dette er den anden sætning."]

    def test_sentence_with_abbreviation(self, splitter: DanishSentenceSplitter) -> None:
        """Test handling of sentences with abbreviations."""
        text = "Hr. Jensen gik en tur. Fru Hansen fulgte efter."
        result = splitter.split_text(text)
        assert result == ["Hr. Jensen gik en tur.", "Fru Hansen fulgte efter."]

    def test_clean_text(self, splitter: DanishSentenceSplitter) -> None:
        """Test the _clean_text method."""
        text = "Dette er en sætning.. Dette er en anden sætning."
        cleaned_text = splitter._clean_text(text)
        assert cleaned_text == "Dette er en sætning. Dette er en anden sætning."

    def test_fix_not_capitalized_sentences(
        self, splitter: DanishSentenceSplitter
    ) -> None:
        """Test handling of sentences that are not capitalized."""
        raw_sentences = ["Dette er en sætning.", "og dette er en fortsættelse."]
        result = splitter._fix_not_capitalized_sentences(raw_sentences)
        assert result == ["Dette er en sætning. og dette er en fortsættelse."]

    def test_fix_orphaned_list_start(self, splitter: DanishSentenceSplitter) -> None:
        """Test handling of orphaned list starts."""
        raw_sentences = ["1.", "Dette er det første punkt."]
        result = splitter._fix_orphaned_list_start(raw_sentences)
        assert result == ["1. Dette er det første punkt."]

    def test_sentence_with_newlines(self, splitter: DanishSentenceSplitter) -> None:
        """Test handling of sentences with newlines."""
        text = "Dette er en sætning\nmed et linjeskift."
        result = splitter.split_text(text)
        assert result == ["Dette er en sætning med et linjeskift."]

    def test_multiple_paragraphs(self, splitter: DanishSentenceSplitter) -> None:
        """Test handling of multiple paragraphs."""
        text = "Dette er det første afsnit.\n\nDette er det andet afsnit."
        result = splitter.split_text(text)
        assert result == ["Dette er det første afsnit.", "Dette er det andet afsnit."]

    def test_sentence_with_quotes(self, splitter: DanishSentenceSplitter) -> None:
        """Test handling of sentences with quotes."""
        text = 'Han sagde: "Dette er et citat." Derefter gik han.'
        result = splitter.split_text(text)
        assert result == ['Han sagde: "Dette er et citat."', "Derefter gik han."]

    def test_sentence_with_parentheses(self, splitter: DanishSentenceSplitter) -> None:
        """Test handling of sentences with parentheses."""
        text = "Dette er en sætning (med parentes). Dette er en anden sætning."
        result = splitter.split_text(text)
        assert result == [
            "Dette er en sætning (med parentes).",
            "Dette er en anden sætning.",
        ]

    def test_multiple_punctuation(self, splitter: DanishSentenceSplitter) -> None:
        """Test handling of sentences with multiple punctuation marks."""
        text = "Hvad sker der her? Det ved jeg ikke!"
        result = splitter.split_text(text)
        assert result == ["Hvad sker der her?", "Det ved jeg ikke!"]

    def test_decimal_numbers(self, splitter: DanishSentenceSplitter) -> None:
        """Test handling of sentences with decimal numbers."""
        text = "Prisen er 10.5 kr. Det er billigt."
        result = splitter.split_text(text)
        assert result == ["Prisen er 10.5 kr.", "Det er billigt."]

    def test_acronyms(self, splitter: DanishSentenceSplitter) -> None:
        """Test handling of sentences with acronyms."""
        text = "Hun arbejder for F.B.I. Det er et spændende job."
        result = splitter.split_text(text)
        assert result == ["Hun arbejder for F.B.I.", "Det er et spændende job."]

    def test_mixed_language(self, splitter: DanishSentenceSplitter) -> None:
        """Test handling of mixed language content."""
        text = "Det er en 'fait accompli'. Vi kan ikke ændre det nu."
        result = splitter.split_text(text)
        assert result == ["Det er en 'fait accompli'.", "Vi kan ikke ændre det nu."]

    def test_web_addresses(self, splitter: DanishSentenceSplitter) -> None:
        """Test handling of web addresses and email addresses."""
        text = "Besøg www.example.com. Send en mail til info@example.com."
        result = splitter.split_text(text)
        assert result == [
            "Besøg www.example.com.",
            "Send en mail til info@example.com.",
        ]

    def test_unconventional_capitalization(
        self, splitter: DanishSentenceSplitter
    ) -> None:
        """Test handling of unconventional capitalization."""
        text = "iPhonen er populær. BREAKING NEWS: En ny model er annonceret."
        result = splitter.split_text(text)
        assert result == [
            "iPhonen er populær.",
            "BREAKING NEWS: En ny model er annonceret.",
        ]
