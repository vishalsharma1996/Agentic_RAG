
import pandas as pd
from pathlib import Path
import os

def load_paper_chunks(paper_path):

    section_path = paper_path / "chunk.parquet"
    table_path = paper_path / "table.parquet"

    chunks = []
    chunks_registry = {}

    if section_path.exists():
        df = pd.read_parquet(section_path)
        chunks.extend(df["text"].tolist())
        for _, row in df.iterrows():
            chunks_registry[row["text"]] = {
                "chunk_id": row["chunk_id"],
                "metadata": row["metadata"]
            }

    if table_path.exists():
        df = pd.read_parquet(table_path)
        chunks.extend(df["text"].tolist())
        for _, row in df.iterrows():
            chunks_registry[row["text"]] = {
                "chunk_id": row["chunk_id"],
                "metadata": row["metadata"]
            }

    return chunks,chunks_registry

def load_dataset_chunks(path = 'parsed_docs'):
  topics = os.listdir(path)
  all_chunks = []
  all_chunks_registry = {}
  for topic in topics:
    topic_path = os.path.join(path,topic,'chunks_tables')
    root = Path(topic_path)
    for paper_dir in root.iterdir():
      if paper_dir.is_dir():
        chunks,chunks_registry = load_paper_chunks(paper_dir)
        if chunks:
          all_chunks.extend(chunks)
          all_chunks_registry.update(chunks_registry)
  return all_chunks,all_chunks_registry
