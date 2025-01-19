import os
import pandas as pd

from functions.company_data import gather_portfolio_data
from functions.text_preprocess import process_text_data

import config

def get_portfolio_df():
    portfolio_data = gather_portfolio_data(config.EQT_X_COMPANY_URLS)
    columns = ["company_name", "url", "source", "text"]
    portfolio_data_df = pd.DataFrame(
        data=portfolio_data,
        columns=columns,
    )
    portfolio_data_df["year"] = 2024
    portfolio_data_df["timestamp"] = pd.to_datetime(portfolio_data_df["year"].astype(str) + "-12-31")
    
    portfolio_data_filtered = process_text_data(portfolio_data_df)
    portfolio_data_filtered["page_number"] = 1
    
    # Reorder columns and rename
    portfolio_data_filtered = portfolio_data_filtered[['company_name', 'url', 'source', 'page_number', 'paragraph', 'text', 'year', 'timestamp']]
    portfolio_data_filtered.columns = ['name', 'url', 'source', 'page_number', 'paragraph', 'text', 'year', 'timestamp']
    return portfolio_data_filtered
    