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

"""
Docling PDF Parsing Pipeline

This module performs GPU-accelerated parsing of research paper PDFs using
Docling's threaded pipeline. The purpose of this stage is to convert raw
PDF documents into structured Docling document objects that preserve the
layout and semantic structure of the paper.

The parsed document objects are serialized and stored as pickle files so
that downstream preprocessing stages (section extraction, table extraction,
chunking, etc.) can operate without repeatedly parsing the original PDFs.

Artifacts Generated
-------------------
parsed_docs/<topic>/<paper_id>.pickle

Each pickle file contains a serialized Docling document object that includes:
- pages
- layout blocks
- tables
- captions
- structural metadata

Pipeline Steps
--------------
1. Configure Docling threaded GPU pipeline.
2. Load models onto the GPU.
3. Iterate through topic folders containing PDFs.
4. Parse PDFs in parallel using ThreadPoolExecutor.
5. Serialize parsed document objects to disk for reuse.

Why This Stage Is Important
---------------------------
PDF parsing is typically the most computationally expensive part of
document ingestion pipelines. By caching parsed Docling objects,
subsequent preprocessing steps can run significantly faster.
"""


# ---------------------------------------------------------------------------
# Configure GPU-accelerated Docling pipeline
# ---------------------------------------------------------------------------

# Define pipeline options for threaded PDF processing
pipeline_options = ThreadedPdfPipelineOptions(
    accelerator_options=AcceleratorOptions(
        device=AcceleratorDevice.CUDA,  # Enable GPU acceleration
    ),
    ocr_batch_size=4,      # OCR batch size (unused here since OCR disabled)
    layout_batch_size=64,  # Batch size for layout detection
    table_batch_size=4,    # Batch size for table detection
)

# Disable OCR since research PDFs usually contain machine-readable text
pipeline_options.do_ocr = False


# Create document converter with the configured pipeline
doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=ThreadedStandardPdfPipeline,
            pipeline_options=pipeline_options,
        )
    }
)


# ---------------------------------------------------------------------------
# Initialize the pipeline (loads models onto the GPU)
# ---------------------------------------------------------------------------

doc_converter.initialize_pipeline(InputFormat.PDF)


# ---------------------------------------------------------------------------
# Helper function to parse a single PDF
# ---------------------------------------------------------------------------

def process_pdf(path):
    """
    Parse a single PDF using Docling.

    Parameters
    ----------
    path : str
        Path to the PDF file.

    Returns
    -------
    result : Docling conversion result OR str
        Returns conversion result if successful,
        otherwise returns the path to indicate failure.
    """

    try:
        result = doc_converter.convert(path)
        return result

    except Exception:
        # Log failed PDF path so it can be retried later
        print(f"Failed paper: {path}")
        return path


# ---------------------------------------------------------------------------
# Main parsing pipeline
# ---------------------------------------------------------------------------

def parse_pdf_docs(path="arxiv_papers"):
    """
    Parse research paper PDFs and cache structured Docling documents.

    Parameters
    ----------
    path : str
        Root directory containing topic folders of PDFs.

    Example Directory Structure
    ---------------------------
    arxiv_papers/
        rag/
            paper1.pdf
            paper2.pdf
        agents/
            paper3.pdf
    """

    # Iterate over topic folders
    dir_list = os.listdir(path)

    for topic in dir_list:

        # Create output directory for parsed documents
        output_dir = os.path.join("parsed_docs", topic)
        os.makedirs(output_dir, exist_ok=True)

        # Path to topic-specific PDFs
        topic_path = os.path.join(path, topic)

        # List all PDF files
        paper_list = os.listdir(topic_path)

        # Construct absolute paths to each paper
        path_list = [os.path.join(topic_path, p) for p in paper_list]

        # Measure runtime for performance monitoring
        start_time = time.time()

        # Parse PDFs in parallel using thread pool
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(process_pdf, path_list))

        total_runtime = time.time() - start_time

        # Identify failed conversions
        failed_paper_path = [r for r in results if isinstance(r, str)]

        print(
            f"Total Runtime for processing {len(paper_list)} papers related to {topic} "
            f"is {total_runtime:.2f} seconds"
        )

        print(f"Failed papers for {topic}: {len(failed_paper_path)}")

        # Save successful parsed documents
        for paper_path, result in zip(path_list, results):

            if isinstance(result, str):
                # Skip failed papers
                continue

            # Normalize filename for storage
            paper = paper_path.split("/")[-1].replace(".", "_")

            # Serialize Docling document object
            with open(f"{output_dir}/{paper}.pickle", "wb") as f:
                pickle.dump(result.document, f)
