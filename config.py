import torch

# The local directory path where downloaded data will be saved.
DOWNLOAD_PATH = "data"

REPORTS_URLS = [
        "https://aiindex.stanford.edu/wp-content/uploads/2022/03/2022-AI-Index-Report_Master.pdf",
        "https://aiindex.stanford.edu/wp-content/uploads/2023/04/HAI_AI-Index-Report_2023.pdf",
        "https://aiindex.stanford.edu/wp-content/uploads/2024/05/HAI_AI-Index-Report-2024.pdf"        
    ]

EQT_X_COMPANY_URLS = {
    "AMCS": "https://www.amcsgroup.com/",
    "Avetta": "https://www.avetta.com/",
    "Billtrust": "https://www.billtrust.com/",
    "Dechra_Pharmaceuticals": "https://eqtgroup.com/current-portfolio/dechra-pharmaceuticals/",
    "Hantverksdata": "https://www.hantverksdata.se/",
    "UTA": "https://eqtgroup.com/current-portfolio/uta/",
    "Zeus": "https://eqtgroup.com/current-portfolio/zeus/"
}

# Reranker 
RERANKER = 'BAAI/bge-reranker-large'

# GPT model:
GPT_MODEL = "gpt-4o-mini-2024-07-18"

# The identifier of the pre-trained sentence transformer model for producing sentence embeddings.
MODEL_SENTENCE_TRANSFORMER = 'all-MiniLM-L6-v2'

# The computing device to be used for model inference and training.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# The identifier for the Mistral-7B-Instruct model
MODEL_ID = 'mistralai/Mistral-7B-Instruct-v0.2'
