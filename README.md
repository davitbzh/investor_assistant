# ⚙️ Private AI assistant for investment professionals. 

This project is an AI system for investment professionals that can deep-dive in how AI impacts portfolio companies. It is built on Hopsworks that
  * creates vector embeddings for PDF files from  [Stanford AI Index Report](https://aiindex.stanford.edu/report/), indexes them for retrieval augmented generation (RAG) in Hopsworks Feature Store with Vector Indexing.
  * Based on user input retrieve top-ranked contexts from the Vector datbase and generate response using OpenAI's `gpt-4o-mini-2024-07-18` or Google's `Gemini 1.5 Flash` midels.
  * provides a UI, written in Streamlit/Python, for querying Report PDF and Protfolio company websites that returns answers, citing the page/paragraph/url-to-pdf in its answer.

![Hopsworks Architecture for Private PDFs Indexed for LLMs](../..//images/llm-pdfs-architecture.gif)

## 📖 Feature Pipeline
The Feature Pipeline does the following:

 * Downloads from the [Stanford AI Index Report](https://aiindex.stanford.edu/report/).
 * Downloads data from selected profolio companies
 * Extracts chunks of text from the PDFs/Websites and stores them in a Vector-Index enabled Feature Group in Hopsworks.

## 🏃🏻‍♂️Training Pipeline
This step is optional if you also want to create a fine-tuned model. Currently we opted to use OpenAI's `gpt-4o-mini-2024-07-18` or Google's `Gemini 1.5 Flash` midels

## 🚀 Inference Pipeline
* A chatbot written in Streamlit that answers questions about the Portfolio comapanies based on AI report PDFs company webistes.

## 🕵🏻‍♂️ Prerequisites
1. Create free account on [Hopsworks](https://app.hopsworks.ai/app) and get api key
2. Get either Google Gemini (free teir available) or OpneAI api key
3. Clone repository:
   ```bash
   git clone https://github.com/davitbzh/investor_assistant.git
   cd investor_assistant
   ```
4. Create `.env` file and save Hopsworks and Gemini/OpneAI api keys:
   ```bash
    GEMINI_KEY=your_gemini_api_key
    OPENAI_API_KEY=your_open_ai_api_key
    HOPSWORKS_API_KEY=your_hopsworks_api_key
   ```
5. Run pipeline `feature_pipeline.ipynb`. Note that for similicity we chouse Jupyter notebook here. However for production environments you can convert this to python script and schedule with prefered cadence.
6. For inference run:
   ```bash
    streamlit run ./app.py
   ```
