import pytest
from ragathon.parsers.markdown import MarkdownParser


class TestMarkdownParser:
    def test_empty_document(self) -> None:
        """Test that an empty markdown document is handled correctly."""
        parser = MarkdownParser()
        document = parser.run("")
        assert len(document.sections) == 0

    def test_single_section_document(self) -> None:
        """Test handling of a document with a single section."""
        markdown_content = "# Heading\nText"
        parser = MarkdownParser()
        document = parser.run(markdown_content)
        assert len(document.sections) == 1
        assert document.sections[0].level == 1
        assert document.sections[0].heading == "Heading"
        assert document.sections[0].text == "Text"

    def test_multiple_top_level_sections(self) -> None:
        """Test handling of multiple top-level sections."""
        markdown_content = "# Heading1\nText1\n# Heading2\nText2"
        parser = MarkdownParser()
        document = parser.run(markdown_content)
        assert len(document.sections) == 2
        assert all(section.level == 1 for section in document.sections)
        for i in range(2):
            assert document.sections[i].heading == f"Heading{i+1}"
            assert document.sections[i].text == f"Text{i+1}"

    def test_nested_sections(self) -> None:
        """Test handling of nested sections."""
        markdown_content = "# Heading1\nText1\n## Subheading1.1\nSubtext1.1"
        parser = MarkdownParser()
        document = parser.run(markdown_content)
        assert len(document.sections) == 2
        assert document.sections[0].level == 1
        assert document.sections[0].children == [document.sections[1]]
        assert document.sections[1].level == 2
        assert document.sections[1].heading == "Subheading1.1"
        assert document.sections[1].parent == document.sections[0]

    def test_heading_followed_by_another_heading(self) -> None:
        """Test handling of a heading immediately followed by another heading."""
        markdown_content = "# Heading1\n# Heading2"
        parser = MarkdownParser()
        document = parser.run(markdown_content)
        assert len(document.sections) == 2
        assert document.sections[0].level == 1
        assert document.sections[0].heading == "Heading1"
        assert document.sections[1].level == 1
        assert document.sections[1].heading == "Heading2"

    def test_inconsistent_level_jumps(self) -> None:
        """Test handling of inconsistent level jumps in sections."""
        markdown_content = "# Heading1\nText1\n### Subheading1.1.1\nSubtext1.1.1"
        parser = MarkdownParser()
        with pytest.raises(
            ValueError,
            match="Section ### Subheading1.1.1 is level 3, but must be level 2 or lower.",
        ):
            parser.run(markdown_content)

    def test_section_without_heading(self) -> None:
        """Test handling of a section that starts without a heading."""
        markdown_content = "Text without a heading\n# Heading"
        parser = MarkdownParser()
        with pytest.raises(
            expected_exception=ValueError,
            match="First line of Markdown file must be a heading, but got 'Text without a heading'.",
        ):
            parser.run(markdown_content)

    def test_invalid_heading_levels(self) -> None:
        """Test handling of invalid heading levels."""
        markdown_content = "###### Valid heading\n####### Invalid heading"
        parser = MarkdownParser()
        with pytest.raises(
            expected_exception=ValueError,
            match="First section must be level 1, but got level 6.",
        ):
            parser.run(markdown_content)

    def test_non_sequential_heading_levels(self) -> None:
        """Test handling of non-sequential heading levels."""
        markdown_content = "# Heading1\n### Subheading1.1.1"
        parser = MarkdownParser()
        with pytest.raises(
            expected_exception=ValueError,
            match="Section ### Subheading1.1.1 is level 3, but must be level 2 or lower.",
        ):
            parser.run(markdown_content)
