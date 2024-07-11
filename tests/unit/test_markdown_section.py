from ragathon.data.models import MarkdownSection


class TestMarkdownSection:
    def test_initialization(self) -> None:
        """Test basic initialization of MarkdownSection."""
        section = MarkdownSection(level=1, heading="Heading", text="Text")
        assert section.level == 1
        assert section.heading == "Heading"
        assert section.text == "Text"
        assert section.parent is None
        assert section.children == []

    def test_parent_child_relationship(self) -> None:
        """Test the parent-child relationship between sections."""
        parent = MarkdownSection(level=1, heading="Parent", text="Parent Text")
        child = MarkdownSection(
            level=2, heading="Child", text="Child Text", parent=parent
        )

        assert child.parent is parent
        assert child in parent.children

    def test_entire_text_without_children(self) -> None:
        """Test the entire_text property for a section without children."""
        section = MarkdownSection(level=1, heading="Heading", text="Text")
        expected_text = "# Heading\n\nText\n"
        assert section.entire_text == expected_text

    def test_section_with_no_text(self) -> None:
        """Test the entire_text property for a section with a heading but no text."""
        section = MarkdownSection(level=1, heading="Heading", text="")
        expected_text = "# Heading\n"
        assert section.entire_text == expected_text

    def test_entire_text_if_parent_has_a_single_child(self) -> None:
        parent = MarkdownSection(level=1, heading="Parent", text="Parent text")
        MarkdownSection(level=2, heading="Child", text="Child text", parent=parent)
        expected_text = "# Parent\n\nParent text\n\n## Child\n\nChild text\n"
        assert parent.entire_text == expected_text

    def test_entire_text_if_parent_has_multiple_children(self) -> None:
        parent = MarkdownSection(level=1, heading="Parent", text="Parent text")
        MarkdownSection(level=2, heading="Child 1", text="Child 1 text", parent=parent)
        MarkdownSection(level=2, heading="Child 2", text="Child 2 text", parent=parent)
        expected_text = "# Parent\n\nParent text\n\n## Child 1\n\nChild 1 text\n\n## Child 2\n\nChild 2 text\n"
        assert parent.entire_text == expected_text

    def test_entire_text_if_parent_has_multiple_levels_of_children(self) -> None:
        parent = MarkdownSection(level=1, heading="Parent", text="Parent text")
        child = MarkdownSection(
            level=2, heading="Child", text="Child text", parent=parent
        )
        MarkdownSection(
            level=3, heading="Grandchild", text="Grandchild text", parent=child
        )
        expected_text = "# Parent\n\nParent text\n\n## Child\n\nChild text\n\n### Grandchild\n\nGrandchild text\n"
        assert parent.entire_text == expected_text
