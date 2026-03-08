import numpy as np
import pandas as pd
import re
import os
import pickle
from data_cleaning import clean_text_content,normalize_column_values,normalize_header_token
from table_normalization import merge_duplicate_columns,is_null,is_repeating_row

def extract_pdf_tables(path = 'arxiv_papers'):
  topic_list = os.listdir(path)
  for topic in topic_list:
    parsed_dir = os.path.join('parsed_docs',topic)
    topic_paper_path = [os.path.join(parsed_dir,p) for p in os.listdir(parsed_dir) if p.endswith('.pickle')]
    for docling_obj in topic_paper_path:
      with open(docling_obj,'rb') as f:
        result = pickle.load(f)
      paper_pdf = docling_obj.split('/')[-1].split('.')[0]
      table_df_li = []
      table_flatten_df_li = []
      os.makedirs(f"{parsed_dir}/tables/{paper_pdf}",exist_ok=True)
      for table_idx, table in enumerate(result.tables):
              current_prefix_parts = []
              last_row_was_repeating = False
              rows_to_keep = []
              prefix_column = []
              table_description = ''
              caption_ref = table.captions[0] if table.captions else []
              page_number = table.prov[0].page_no if table.prov else None
              if caption_ref:
                table_description = clean_text_content(caption_ref.resolve(result).text)
              table_df: pd.DataFrame = table.export_to_dataframe(doc=result)
              headers = table_df.columns.tolist()
              new_headers = [normalize_header_token(str(header)) for header in headers]
              table_df.columns = new_headers
              table_df = merge_duplicate_columns(table_df)
              for col in new_headers:
                col_val = table_df[col].values.tolist()
                new_col_val = [normalize_column_values(str(val)) for val in col_val]
                table_df[col] = new_col_val
              df = table_df.copy()
              for idx, row in df.iterrows():
                if is_repeating_row(row):
                  value = str(row.dropna().iloc[0]).strip()

                  if last_row_was_repeating:
                      current_prefix_parts.append(value)
                  else:
                      current_prefix_parts = [value]

                  last_row_was_repeating = True
                  continue

                # Data row
                last_row_was_repeating = False

                rows_to_keep.append(row)
                prefix_column.append(" ".join(current_prefix_parts) if current_prefix_parts else None)

              # Build new dataframe
              new_df = pd.DataFrame(rows_to_keep).reset_index(drop=True)
              new_df.insert(0, "Metric_Prefix", prefix_column)
              if not new_df.empty:
                col_value = new_df.columns.tolist()[1]
                new_df[col_value] = new_df.apply(lambda x: x.Metric_Prefix + ' associated ' + x[col_value] if x.Metric_Prefix is not None and x[col_value] is not None else x[col_value],axis=1)
                new_df.drop('Metric_Prefix',axis=1,inplace=True)
                new_df = new_df.applymap(lambda x: str(x) if not np.array(pd.isna(x)).any() else x)
                new_df.to_parquet(f"{parsed_dir}/tables/{paper_pdf}/{table_idx}.parquet",index=False)
                table_metadata = {'paper_id':paper_pdf,'table_id':table_idx,
                                  'caption':table_description,'page_number':page_number,
                                  'file':f'{table_idx}.parquet'}
                table_df_li.append(pd.DataFrame(table_metadata,index=[0]))
                chunks = []
                for _,row in new_df.iterrows():
                  entity = row[0]
                  text = ''
                  values = []
                  for col in df.columns[1:]:
                    val = row[col]
                    if val is not None and val != '-' and val!='- -':
                      values.append(f"{col} is {val}")
                  if entity is not None and entity != '' and entity != '- -' and entity != '-':
                    text = f"{entity} has the following values " + ", ".join(values)
                    chunks.append(text)
                table_flatten_metadata = pd.DataFrame({'table_description':chunks})
                table_flatten_metadata['paper_id'] = paper_pdf
                table_flatten_metadata['table_id'] = table_idx
                table_flatten_metadata['caption'] = table_description
                table_flatten_metadata['page_number'] = page_number
                table_flatten_df_li.append(table_flatten_metadata)
      if table_df_li:
        table_metadata_df = pd.concat(table_df_li).reset_index()
        table_flatten_df = pd.concat(table_flatten_df_li).reset_index()
        table_metadata_df.to_parquet(f"{parsed_dir}/tables/{paper_pdf}/table_metadata.parquet",index=False)
        table_flatten_df.to_parquet(f"{parsed_dir}/tables/{paper_pdf}/table_description.parquet",index=False)
