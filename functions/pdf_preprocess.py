import os
import re
import requests
import PyPDF2
from typing import List, Dict, Union
from pathlib import Path


def download_pdf(url, output_path):
    """
    Downloads a PDF from a given URL and saves it to output_path.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded PDF: {output_path}")
    except requests.RequestException as e:
        print(f"Failed to download {url}. Error: {e}")

def download_stanford_ai_reports(urls: List[str], ai_report_text: List, pdfs_path: str = 'data') -> List[str]:
    """
    Download 2022, 2023, and 2024 Stanford AI Index reports.
    (Check official Stanford AI Index website for correct or updated links)
    """
    
    for url in urls:
        filename = url.split('/')[-1]
    
        # Output filenames
        pdf_file = f"{pdfs_path}/{filename}"
    
        # Initialize a list to store information about new files
        print('⛳️ Dowloading Stanford ai index report...')
        # Download each
        download_pdf(url, pdf_file)
        
        process_pdf_file(ai_report_text, pdf_file, url)

    return ai_report_text



def process_pdf_file(document_text: List,
                     pdf_path: str, url: str) -> List:
    """
    Process content of a PDF file and append information to the document_text list.

    Parameters:
    - file_info (Dict): Information about the PDF file.
    - document_text (List): List containing document information.
    - pdf_path (str): Path to the folder containing PDF files (default is 'data/').

    Returns:
    - List: Updated document_text list.
    """    
    if pdf_path.split('.')[-1] == 'pdf':
        file_path = Path(pdf_path)
        file_title = file_path.stem        
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        pages_amount = len(pdf_reader.pages)
        print(f'Amount of pages: {pages_amount} in {file_title}')
            
        for i, page in enumerate(pdf_reader.pages):
            document_text.append([file_title, url, i+1, page.extract_text()])
    return document_text

def extract_year_regex(filename):
    pattern = re.compile(r'^(?P<start>\d{4})|(?P<end>\d{4})')
    match = pattern.search(filename)
    if match:
        # If 'start' group is matched, return it; else return 'end'
        return match.group('start') or match.group('end')
    return None
