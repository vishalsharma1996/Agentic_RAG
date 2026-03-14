import pandas as pd
from pathlib import Path
import os

# -----------------------------
# Section chunk builder
# -----------------------------

def build_section_chunks(section_folder_path, sections='sections'):
    # List all topic folders (e.g., rag, llm, etc.)
    topic_list = os.listdir(section_folder_path)

    for topic in topic_list:
        # Path to sections inside each topic
        full_path = os.path.join(section_folder_path, topic, sections)

        # Each folder corresponds to a paper
        section_pdf_li = os.listdir(full_path)

        for pdf_sections in section_pdf_li:
            # Path containing parquet file for that paper
            section_parquet_path = os.path.join(full_path, pdf_sections)

            # Load section parquet
            df = pd.read_parquet(f"{section_parquet_path}/sections.parquet")

            chunks = []

            # Iterate over each section row
            for _, row in df.iterrows():
                section_title = row["section_title"]

                # content_blocks contain structured blocks like text, tables, figures
                blocks = row["content_blocks"]

                # Extract only text blocks
                text_blocks = [b["text"] for b in blocks if b["type"] == "text"]

                # Extract urls from blocks (if present)
                urls = [b['urls'] for b in blocks]

                # Merge section title with text blocks to form a single chunk
                merged_text = section_title + " " + " ".join(text_blocks)

                # Build chunk record
                chunks.append({
                    "chunk_id": f"{topic}_{row['paper_id']}_sec_{row['section_id']}",
                    "text": merged_text,
                    "metadata": {
                        "paper_id": row['paper_id'],
                        "source_type": "section",   # indicates section-based chunk
                        "section_title": section_title,
                        "urls": urls
                    }
                })

            # Convert chunks to dataframe
            chunk_df = pd.DataFrame(chunks)

            # Output directory for section chunks
            chunk_path = os.path.join(
                os.path.join(section_folder_path, topic),
                'chunks_tables',
                df.iloc[0]['paper_id']
            )

            # Create directory if it does not exist
            os.makedirs(chunk_path, exist_ok=True)

            # Save chunks to parquet
            chunk_df.to_parquet(f"{chunk_path}/chunk.parquet", index=False)


# -----------------------------
# Table chunk builder
# -----------------------------

def build_table_chunks(table_folder_path, tables='tables'):

    # List topic folders
    topic_list = os.listdir(table_folder_path)

    for topic in topic_list:

        # Path to tables directory for the topic
        full_path = os.path.join(table_folder_path, topic, tables)

        # Each folder corresponds to a paper
        table_pdf_li = os.listdir(full_path)

        for table_section in table_pdf_li:

            # Path to table parquet files
            table_parquet_path = os.path.join(full_path, table_section)

            # Table description file path
            file_path = f"{table_parquet_path}/table_description.parquet"

            # Process only if table description exists
            if os.path.exists(file_path):

                desc_df = pd.read_parquet(file_path)

                tables_li = []

                # Iterate through table descriptions
                for _, row in desc_df.iterrows():

                    description = row['table_description']
                    caption = row['caption']

                    # Combine caption and description into chunk text
                    text = f"{caption} {description}"

                    tables_li.append({
                        "chunk_id": f"{topic}_{row['paper_id']}_table_{row['table_id']}",
                        "text": text,
                        "metadata": {
                            "paper_id": row['paper_id'],
                            "source_type": "table",   # indicates table-based chunk
                            "table_id": row["table_id"],
                            "caption": caption
                        }
                    })

                # Convert to dataframe
                table_df = pd.DataFrame(tables_li)

                # Output path for table chunks
                table_path = os.path.join(
                    os.path.join(table_folder_path, topic),
                    'chunks_tables',
                    desc_df.iloc[0]['paper_id']
                )

                # Create directory if needed
                os.makedirs(table_path, exist_ok=True)

                # Save table chunks
                table_df.to_parquet(f"{table_path}/table.parquet", index=False)
