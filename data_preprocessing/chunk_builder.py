import pandas as pd
from pathlib import Path
import os

# -----------------------------
# Section chunk builder
# -----------------------------

def build_section_chunks(section_folder_path,sections='sections'):
    topic_list = os.listdir(section_folder_path)
    for topic in topic_list:
      full_path = os.path.join(section_folder_path,topic,sections)
      section_pdf_li = os.listdir(full_path)
      for pdf_sections in section_pdf_li:
        section_parquet_path = os.path.join(full_path,pdf_sections)
        df = pd.read_parquet(f"{section_parquet_path}/sections.parquet")
        chunks = []
        for _, row in df.iterrows():
              section_title = row["section_title"]
              blocks = row["content_blocks"]
              text_blocks = [b["text"] for b in blocks if b["type"] == "text"]
              urls = [b['urls'] for b in blocks]
              merged_text = section_title + " " + " ".join(text_blocks)
              chunks.append({
                  "text": merged_text,
                  "metadata": {
                      "paper_id": row['paper_id'],
                      "source_type": "section",
                      "section_title": section_title,
                      "urls": urls
                  }
              })

        chunk_df = pd.DataFrame(chunks)
        chunk_path = os.path.join(os.path.join(section_folder_path,topic),'chunks_tables',df.iloc[0]['paper_id'])
        os.makedirs(chunk_path,exist_ok=True)
        chunk_df.to_parquet(f"{chunk_path}/chunk.parquet",index=False)

# -----------------------------
# Table chunk builder
# -----------------------------

def build_table_chunks(table_folder_path,tables='tables'):
  topic_list = os.listdir(table_folder_path)
  for topic in topic_list:
      full_path = os.path.join(table_folder_path,topic,tables)
      table_pdf_li = os.listdir(full_path)
      for table_section in table_pdf_li:
        table_parquet_path = os.path.join(full_path,table_section)
        file_path = f"{table_parquet_path}/table_description.parquet"
        if os.path.exists(file_path):
          desc_df = pd.read_parquet(file_path)
          tables_li = []
          for _,row in desc_df.iterrows():
            description = row['table_description']
            caption = row['caption']
            text = f"{caption} {description}"
            tables_li.append({
                    "text": text,
                    "metadata": {
                        "paper_id": row['paper_id'],
                        "source_type": "table",
                        "table_id": row["table_id"],
                        "caption": caption
                        }
                })
          table_df = pd.DataFrame(tables_li)
          table_path = os.path.join(os.path.join(table_folder_path,topic),'chunks_tables',desc_df.iloc[0]['paper_id'])
          #print(table_path)
          os.makedirs(table_path,exist_ok=True)
          table_df.to_parquet(f"{table_path}/table.parquet",index=False)
