import os
import datetime
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
from pydantic import TypeAdapter

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import (
    ThreadedPdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.threaded_standard_pdf_pipeline import ThreadedStandardPdfPipeline
from docling.utils.profiling import ProfilingItem
from docling.document_converter import DocumentConverter
from concurrent.futures import ThreadPoolExecutor

# Define the GPU pipeline for Docling
pipeline_options = ThreadedPdfPipelineOptions(
        accelerator_options=AcceleratorOptions(
            device=AcceleratorDevice.CUDA,
        ),
        ocr_batch_size=4,
        layout_batch_size=64,
        table_batch_size=4,
    )
pipeline_options.do_ocr = False

doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=ThreadedStandardPdfPipeline,
                pipeline_options=pipeline_options,
            )
        }
    )

# Loads model to the GPU
doc_converter.initialize_pipeline(InputFormat.PDF)

def process_pdf(path):
  try:
    result = doc_converter.convert(path)
    return result
  except Exception as e:
    print(f'Failed paper: {path}')
    return path

def parse_pdf_docs(path='arxiv_papers'):
  dir_list = os.listdir(path)
  for topic in dir_list:
    output_dir = os.path.join("parsed_docs", topic)
    os.makedirs(output_dir, exist_ok=True)
    topic_path = os.path.join(path,topic)
    paper_list = os.listdir(topic_path)
    path_list = [os.path.join(topic_path,p) for p in paper_list]
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
      results = list(executor.map(process_pdf,path_list))
    total_runtime = time.time() - start_time
    failed_paper_path = [r for r in results if isinstance(r,str)]
    print(f'Total Runtime for processing {len(paper_list)} papers related to {topic} is {total_runtime:.2f} seconds')
    print(f'Failed papers for {topic}: {len(failed_paper_path)}')
    for paper_path,result in zip(path_list,results):
      if isinstance(result,str):
        continue
      paper = paper_path.split('/')[-1].replace('.','_')
      with open(f'{output_dir}/{paper}.pickle','wb') as f:
        pickle.dump(result.document,f)
