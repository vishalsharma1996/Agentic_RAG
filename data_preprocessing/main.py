from docling_parser import parse_pdf_docs
from extract_sections import extract_sections_from_docling
from extract_tables import extract_pdf_tables

def main():

  print("Step 1: Parsing PDFs...")
  parse_pdf_docs(path = 'arxiv_papers')

  print("Step 2: Extracting sections from docling...")
  extract_sections_from_docling(path = 'arxiv_papers')

  print("Step 3: Extracting tables...")
  extract_pdf_tables(path = 'arxiv_papers')

  print("Pipeline completed!")

if __name__ == "__main__":
    main()
