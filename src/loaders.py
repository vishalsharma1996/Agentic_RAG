
import pandas as pd
from pathlib import Path
import os

def load_paper_chunks(paper_path):

    section_path = paper_path / "chunk.parquet"
    table_path = paper_path / "table_chunk.parquet"

    chunks = []

    if section_path.exists():
        df = pd.read_parquet(section_path)
        chunks.extend(df["text"].tolist())

    if table_path.exists():
        df = pd.read_parquet(table_path)
        chunks.extend(df["text"].tolist())

    return chunks

def load_dataset_chunks(path = 'parsed_docs'):
  topics = os.listdir(path)
  all_chunks = []
  for topic in topics:
    topic_path = os.path.join(path,topic,'chunks_tables')
    root = Path(topic_path)
    for paper_dir in root.iterdir():
      if paper_dir.is_dir():
        chunks = load_paper_chunks(paper_dir)
        if chunks:
          all_chunks.extend(chunks)
  return all_chunks
