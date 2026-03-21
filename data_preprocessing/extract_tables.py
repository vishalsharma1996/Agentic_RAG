import numpy as np
import pandas as pd
import re
import os
import pickle
from data_cleaning import clean_text_content,normalize_column_values,normalize_header_token
from table_normalization import merge_duplicate_columns,is_null,is_repeating_row

"""
Table Extraction Pipeline for Research Paper PDFs

This module extracts tables from parsed Docling document objects and converts them
into structured formats suitable for downstream RAG systems.

Outputs produced for each paper:

1. Raw tables saved as parquet files
2. Table metadata (caption, page number, table id)
3. Flattened semantic table descriptions used for retrieval pipelines
"""

def extract_pdf_tables(path="arxiv_papers"):
    """
    Main pipeline to extract and normalize tables from parsed Docling documents.
    """

    # Iterate over topic folders containing parsed documents
    topic_list = os.listdir(path)

    for topic in topic_list:

        # Directory where parsed Docling objects are stored
        parsed_dir = os.path.join("parsed_docs", topic)

        # List of parsed Docling document objects
        topic_paper_path = [
            os.path.join(parsed_dir, p)
            for p in os.listdir(parsed_dir)
            if p.endswith(".pickle")
        ]

        # Process each parsed document
        for docling_obj in topic_paper_path:

            # Load parsed Docling document object
            with open(docling_obj, "rb") as f:
                result = pickle.load(f)

            # Extract paper id from filename
            paper_pdf = docling_obj.split("/")[-1].split(".")[0]

            # Lists to store metadata and flattened table descriptions
            table_df_li = []
            table_flatten_df_li = []

            # Create directory to store table outputs
            os.makedirs(f"{parsed_dir}/tables/{paper_pdf}", exist_ok=True)

            # Iterate through tables detected in the document
            for table_idx, table in enumerate(result.tables):

                # Variables used to reconstruct hierarchical metric prefixes
                current_prefix_parts = []
                last_row_was_repeating = False
                rows_to_keep = []
                prefix_column = []

                table_description = ""

                # Extract table caption if available
                caption_ref = table.captions[0] if table.captions else []

                # Extract page number where table appears
                page_number = table.prov[0].page_no if table.prov else None

                if caption_ref:
                    table_description = clean_text_content(
                        caption_ref.resolve(result).text
                    )
                    table_description = normalize_column_values(table_description)

                # Convert Docling table object to pandas DataFrame
                table_df: pd.DataFrame = table.export_to_dataframe(doc=result)

                # Normalize column headers
                headers = table_df.columns.tolist()
                new_headers = [normalize_header_token(str(header)) for header in headers]
                table_df.columns = new_headers

                # Merge duplicate columns if table extraction produced duplicates
                table_df = merge_duplicate_columns(table_df)

                # Normalize cell values for each column
                for col in new_headers:
                    col_val = table_df[col].values.tolist()
                    new_col_val = [
                        normalize_column_values(str(val)) for val in col_val
                    ]
                    table_df[col] = new_col_val

                df = table_df.copy()

                # Detect repeating rows which often represent metric prefixes
                for idx, row in df.iterrows():

                    # Check if row is a repeating prefix row
                    if is_repeating_row(row):

                        value = str(row.dropna().iloc[0]).strip()

                        if last_row_was_repeating:
                            current_prefix_parts.append(value)
                        else:
                            current_prefix_parts = [value]

                        last_row_was_repeating = True
                        continue

                    # Otherwise it is a data row
                    last_row_was_repeating = False

                    rows_to_keep.append(row)

                    # Store prefix context if available
                    prefix_column.append(
                        " ".join(current_prefix_parts) if current_prefix_parts else None
                    )

                # Construct cleaned dataframe
                new_df = pd.DataFrame(rows_to_keep).reset_index(drop=True)

                # Insert prefix column at first position
                new_df.insert(0, "Metric_Prefix", prefix_column)

                if not new_df.empty:

                    # Combine prefix with metric name for better semantic meaning
                    col_value = new_df.columns.tolist()[1]

                    new_df[col_value] = new_df.apply(
                        lambda x: x.Metric_Prefix + " associated with " + x[col_value]
                        if x.Metric_Prefix is not None and x[col_value] is not None
                        else x[col_value],
                        axis=1,
                    )

                    # Remove temporary prefix column
                    new_df.drop("Metric_Prefix", axis=1, inplace=True)

                    # Convert all cells to scalar values (avoid parquet serialization issues)
                    new_df = new_df.applymap(
                        lambda x: str(x) if not np.array(pd.isna(x)).any() else x
                    )

                    # Save normalized table as parquet
                    new_df.to_parquet(
                        f"{parsed_dir}/tables/{paper_pdf}/{table_idx}.parquet",
                        index=False,
                    )

                    # Store metadata for this table
                    table_metadata = {
                        "paper_id": paper_pdf,
                        "table_id": table_idx,
                        "caption": table_description,
                        "page_number": page_number,
                        "file": f"{table_idx}.parquet",
                    }

                    table_df_li.append(pd.DataFrame(table_metadata, index=[0]))

                    # Convert rows into flattened semantic text descriptions
                    chunks = []

                    for _, row in new_df.iterrows():

                        entity = row[0]
                        values = []

                        for col in df.columns[1:]:
                            val = row[col]

                            if val is not None and val != "-" and val != "- -":
                                values.append(f"{col} is {val}")

                        if (
                            entity is not None
                            and entity != ""
                            and entity != "- -"
                            and entity != "-"
                        ):
                            text = f"{entity} has the following values " + ", ".join(
                                values
                            )
                            chunks.append(text)

                    # Store flattened row descriptions
                    table_flatten_metadata = pd.DataFrame(
                        {"table_description": chunks}
                    )

                    table_flatten_metadata["paper_id"] = paper_pdf
                    table_flatten_metadata["table_id"] = table_idx
                    table_flatten_metadata["caption"] = table_description
                    table_flatten_metadata["page_number"] = page_number

                    table_flatten_df_li.append(table_flatten_metadata)

            # Save metadata for all tables in the paper
            if table_df_li:

                table_metadata_df = pd.concat(table_df_li).reset_index()

                table_flatten_df = pd.concat(table_flatten_df_li).reset_index()

                table_metadata_df.to_parquet(
                    f"{parsed_dir}/tables/{paper_pdf}/table_metadata.parquet",
                    index=False,
                )

                table_flatten_df.to_parquet(
                    f"{parsed_dir}/tables/{paper_pdf}/table_description.parquet",
                    index=False,
                )
