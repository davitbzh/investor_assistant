import os
import PyPDF2
import pandas as pd

from functions.pdf_preprocess import (
    download_stanford_ai_reports, 
    process_pdf_file,
    extract_year_regex
)
from functions.text_preprocess import process_text_data

import config

def get_reports_df():
    ai_report_text = []    
    
    download_stanford_ai_reports(urls = config.REPORTS_URLS, ai_report_text = ai_report_text, pdfs_path = config.DOWNLOAD_PATH)
    
    # Create a DataFrame
    columns = ["file_name", "file_link", "page_number", "text"]
    ai_report_text_df = pd.DataFrame(
        data=ai_report_text,
        columns=columns,
    )
    
    ai_report_text_df["year"] = ai_report_text_df["file_name"].apply(extract_year_regex).astype(str).astype(int)
    ai_report_text_df
    ai_report_text_df["timestamp"] = pd.to_datetime(ai_report_text_df["year"].astype(str) + "-12-31")
    
    # Process text data using the process_text_data function
    ai_report_text_processed_df = process_text_data(ai_report_text_df)
    ai_report_text_processed_df["source"] = "stanford_report"
    
    # Reorder columns and rename
    ai_report_text_processed_df  = ai_report_text_processed_df[['file_name', 'file_link', 'source', 'page_number', 'paragraph', 'text', 'year', 'timestamp']]
    ai_report_text_processed_df.columns = ['name', 'url', 'source', 'page_number', 'paragraph', 'text', 'year', 'timestamp']

    return ai_report_text_processed_df