from docling_parser import parse_pdf_docs
from extract_sections import extract_sections_from_docling
from extract_tables import extract_pdf_tables
from chunk_builder import build_section_chunks,build_table_chunks

def main():

  print("Step 1: Parsing PDFs...")
  parse_pdf_docs(path = 'arxiv_papers')

  print("Step 2: Extracting sections from docling...")
  extract_sections_from_docling(path = 'arxiv_papers')

  print("Step 3: Extracting tables...")
  extract_pdf_tables(path = 'arxiv_papers')

  print("Step 4: Combine Chunks for table and sections...")
  build_section_chunks('parsed_docs')
  build_table_chunks('parsed_docs')

  print("Pipeline completed!")

if __name__ == "__main__":
    main()
