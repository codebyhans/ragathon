import re

from ragathon.data.models import MarkdownDocument, MarkdownSection
from ragathon.utils.strings import generate_id


def add_section_to_document(md_doc: MarkdownDocument, heading: str, text: str) -> None:
    """Add a section to a Markdown document.

    Args:
        heading (str): Heading of the section.
        text (str): Text of the section.

    Raises:
        ValueError: If the heading is not a valid heading.
    """
    level = heading.count("#")
    heading_text = heading.strip("#").strip()

    if len(md_doc.sections) == 0:
        if level != 1:
            raise ValueError(f"First section must be level 1, but got level {level}.")

        section = MarkdownSection(
            id=generate_id(parts=[heading_text]),
            level=level,
            heading=heading_text,
            text=text,
        )
    else:
        prev_section = md_doc.sections[-1]

        if level == 1:
            # New top-level section
            section = MarkdownSection(
                id=generate_id(parts=[heading_text]),
                level=level,
                heading=heading_text,
                text=text,
            )
        elif level == prev_section.level:
            # New section at same level as previous section
            parent = prev_section.parent
            assert parent is not None
            section = MarkdownSection(
                id=generate_id(parts=[parent.id, heading_text]),
                level=level,
                heading=heading_text,
                text=text,
                parent=parent,
            )
        else:
            # New section at lower level than previous section
            if level > prev_section.level + 1:
                raise ValueError(
                    f"Section {heading} is level {level}, but must be level {prev_section.level + 1} or lower."
                )

            parent = prev_section
            section = MarkdownSection(
                id=generate_id(parts=[parent.id, heading_text]),
                level=level,
                heading=heading_text,
                text=text,
                parent=parent,
            )

    md_doc.sections.append(section)


class MarkdownParser:
    """Extracts text from markdown files."""

    def run(self, markdown_content: str) -> MarkdownDocument:
        """Extract text from markdown file."""

        # Initialize variables
        document = MarkdownDocument()
        current_heading = None
        current_text = ""

        cleaned_markdown = markdown_content.strip()
        if len(cleaned_markdown) > 0:
            lines = cleaned_markdown.split("\n")

            # Parse markdown file line by line
            for line in lines:
                header_match = re.match(r"^#+\s", line)

                # If line is a header, save previous section
                if header_match:
                    if current_heading:
                        # Save previous section
                        add_section_to_document(
                            md_doc=document,
                            heading=current_heading,
                            text=current_text,
                        )
                    current_heading = line.strip()
                    current_text = ""
                else:
                    if current_heading is None:
                        raise ValueError(
                            f"First line of Markdown file must be a heading, but got '{line}'."
                        )

                    current_text += f"\n{line}" if len(current_text) > 0 else line

            assert current_heading is not None

            # Save last section
            add_section_to_document(
                md_doc=document,
                heading=current_heading,
                text=current_text,
            )

        return document
