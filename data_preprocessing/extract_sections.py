import re
import numpy as np
import os
import pickle
import pandas as pd
from data_cleaning import clean_text_content,find_and_replace_urls

def extract_sections_from_docling(path = 'arxiv_papers'):
    """
    Extract section-grouped text from a Docling document.

    Goal:
    Convert Docling text elements into a structure like:

        Section
            ├── paragraph
            ├── paragraph
            ├── formula

    Instead of storing paragraphs independently.

    Parameters
    ----------
    s_doc : Docling converted document object
        Output of:
            converter = DocumentConverter()
            s_doc = converter.convert("paper.pdf")

    paper_id : str
        Unique identifier for the paper.

    Returns
    -------
    sections : list
        List of section dictionaries containing grouped content blocks.
    """

    topic_list = os.listdir(path)
    for topic in topic_list:
      parsed_dir = os.path.join('parsed_docs',topic)
      topic_paper_path = [os.path.join(parsed_dir,p) for p in os.listdir(parsed_dir) if p.endswith('.pickle')]
      for docling_obj in topic_paper_path:
        with open(docling_obj,'rb') as f:
          result = pickle.load(f)
        paper_pdf = docling_obj.split('/')[-1].split('.')[0]
        # allowed types for storing information
        ALLOWED_TYPES = {"text", "formula","section_header"}

        # List that will store all extracted sections
        sections = []

        # Tracks the section currently being populated
        current_section = None

        # Incremental id for each section encountered
        section_id = 0

        # Iterate through Docling text elements in the order they appear in the document
        for idx, item in enumerate(result.texts):

            # Clean the text content by stripping leading/trailing whitespace
            text_content = item.text.strip()

            # Skip empty blocks that may appear due to PDF parsing artifacts
            if not text_content:
                continue

            # Extract page number from provenance information
            # Docling stores page info inside the "prov" field
            page_number = item.prov[0].page_no if item.prov else None

            # Determine the type of text block (e.g., text, section_header, caption)
            # This helps differentiate headers from normal paragraphs
            block_type = item.label.value if item.label else "unknown"

            # -------------------------------------------------------------
            # If we encounter a SECTION HEADER
            # -------------------------------------------------------------
            if block_type == "section_header":

                # If a section is already open, save it before starting a new one
                if current_section is not None:
                    sections.append(current_section)

                # Increment section counter
                section_id += 1
                text_content = clean_text_content(text_content)
                text_content = text_content.replace('-',' ')
                text_content = re.sub('[^a-zA-Z_ ]','',text_content).strip()
                # Initialize a new section container
                current_section = {
                    "paper_id": paper_pdf,                 # document identifier
                    "section_id": section_id,             # sequential section id
                    "section_title": text_content,        # title of the section
                    "section_level": getattr(item, "level", None),  # header hierarchy level
                    "page_start": page_number,            # page where section starts
                    "page_end": page_number,              # updated as we add content
                    "content_blocks": []                  # list of paragraphs/blocks in this section
                }

                continue

            if block_type not in ALLOWED_TYPES:
                continue
            # -------------------------------------------------------------
            # Handle content appearing before the first section header
            # Example: abstract or introduction paragraphs
            # -------------------------------------------------------------
            if current_section is None:

                # Create a default "preamble" section
                current_section = {
                    "paper_id": paper_pdf,
                    "section_id": section_id,
                    "section_title": "preamble",
                    "section_level": 0,
                    "page_start": page_number,
                    "page_end": page_number,
                    "content_blocks": []
                }

            # -------------------------------------------------------------
            # Add the current text block to the active section
            # -------------------------------------------------------------
            elif current_section is not None:
              if block_type == 'section_header':
                text_content = clean_text_content(text_content)
                # Append block information
                current_section["content_blocks"].append({
                      "block_index": idx,       # original ordering from Docling
                      "page_number": page_number,
                      "type": block_type,       # paragraph / caption / etc.
                      "text": text_content
                  })

                  # Update page_end so we know the full page span of the section
                if page_number is not None:
                      current_section["page_end"] = page_number

              elif len(text_content.split(' ')) > 10:
                  text_content = clean_text_content(text_content)

                  new_matches = find_and_replace_urls(text_content)

                  # Append block information
                  current_section["content_blocks"].append({
                      "block_index": idx,       # original ordering from Docling
                      "page_number": page_number,
                      "type": block_type,       # paragraph / caption / etc.
                      "text": text_content,
                      'urls': [{"context": c, "url": u} for c, u in new_matches if new_matches]
                  })

                  # Update page_end so we know the full page span of the section
                  if page_number is not None:
                      current_section["page_end"] = page_number

        # -------------------------------------------------------------
        # After finishing iteration, append the final section
        # -------------------------------------------------------------
        if current_section is not None:
            sections.append(current_section)

        # Return the list of section containers
        sections_df = pd.DataFrame(sections)
        os.makedirs(f"{parsed_dir}/sections/{paper_pdf}",exist_ok=True)
        sections_df.to_parquet(f"{parsed_dir}/sections/{paper_pdf}/sections.parquet",index=False)
