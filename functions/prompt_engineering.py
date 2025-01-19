import os
import requests
import json_repair

from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker
from hsfs import feature_view

def get_reranker(reranker_model: str) -> FlagReranker:
    reranker = FlagReranker(
        reranker_model, 
        use_fp16=True,
    ) 
    return reranker
    
def get_source(neighbors: List[Tuple[str, str, int, int]]) -> str:
    """
    Generates a formatted string for the sources of the provided context.

    Args:
        neighbors (List[Tuple[str, str, int, int]]): List of tuples representing document information.

    Returns:
        str: Formatted string containing document names, links, pages, and paragraphs.
    """
    return '\n\nReferences:\n' + '\n'.join(
        [
            f' - {neighbor[0]}({neighbor[1]}): Page: {neighbor[2]}, Paragraph: {neighbor[3]}' 
            for neighbor 
            in neighbors
        ]
    )

def get_context(neighbors: List[Tuple[str]]) -> str:
    """
    Generates a formatted string for the context based on the provided neighbors.

    Args:
        neighbors (List[Tuple[str]]): List of tuples representing context information.

    Returns:
        str: Formatted string containing context information.
    """
    return '\n\n'.join([neighbor[-1] for neighbor in neighbors])



def get_neighbors(query: str, sentence_transformer: SentenceTransformer, feature_view, k: int = 10) -> List[Tuple[str, float]]:
    """
    Get the k closest neighbors for a given query using sentence embeddings.

    Parameters:
    - query (str): The input query string.
    - sentence_transformer (SentenceTransformer): The sentence transformer model.
    - feature_view (FeatureView): The feature view for retrieving neighbors.
    - k (int, optional): Number of neighbors to retrieve. Default is 10.

    Returns:
    - List[Tuple[str, float]]: A list of tuples containing the neighbor context.
    """
    question_embedding = sentence_transformer.encode(query)

    # Retrieve closest neighbors
    neighbors = feature_view.find_neighbors(
        question_embedding, 
        k=k,
    )

    return neighbors


def rerank(query: str, neighbors: List[str], reranker, k: int = 3) -> List[str]:
    """
    Rerank a list of neighbors based on a reranking model.

    Parameters:
    - query (str): The input query string.
    - neighbors (List[str]): List of neighbor contexts.
    - reranker (Reranker): The reranking model.
    - k (int, optional): Number of top-ranked neighbors to return. Default is 3.

    Returns:
    - List[str]: The top-ranked neighbor contexts after reranking.
    """
    # Compute scores for each context using the reranker
    scores = [reranker.compute_score([query, context[5]]) for context in neighbors]

    combined_data = [*zip(scores, neighbors)]

    # Sort contexts based on the scores in descending order
    sorted_data = sorted(combined_data, key=lambda x: x[0], reverse=True)

    # Return the top-k ranked contexts
    return [context for score, context in sorted_data][:k]

def get_context_and_source(user_query: str, sentence_transformer: SentenceTransformer,
                           feature_view: feature_view.FeatureView, reranker: FlagReranker, year: int = None, k: int = 10) -> Tuple[str, str]:
    """
    Retrieve context and source based on user query using a combination of embedding, feature view, and reranking.

    Parameters:
    - user_query (str): The user's input query string.
    - sentence_transformer (SentenceTransformer): The sentence transformer model.
    - feature_view (FeatureView): The feature view for retrieving neighbors.
    - reranker (Reranker): The reranking model.
    - year: filter to select only findigs of this particular year.
    - k: number of nearest neighbors to find

    Returns:
    - Tuple[str, str]: A tuple containing the retrieved context and source.
    """
    # Retrieve closest neighbors
    neighbors = get_neighbors(
        user_query,
        sentence_transformer,
        feature_view,
        k=k,
    )
    if year is not None:
        neighbors = [i for i in neighbors if i[6]==year]
    
    # Rerank the neighbors to get top-k
    context_reranked = rerank(
        user_query,
        neighbors,
        reranker,
        k=3,
    )

    # Retrieve source
    source = get_source(context_reranked)

    return context_reranked, source


def build_prompt(query, context):
    """
    Build a multi-shot prompt for LLM to provide more detailed and relevant answers.
    """
    # -- MULTI-SHOT EXAMPLES --
    example_q1 = "What does the 2022 AI Index report highlight about global AI investments?"
    example_a1 = (
        "According to the 2022 AI Index report (Pages 20-21), global AI investments increased "
        "significantly, especially in fintech and healthcare sectors."
    )

    example_q2 = "Does the 2023 report mention the impact on any EQT X portfolio companies?"
    example_a2 = (
        "Based on the 2023 AI Index (Chapter 2, Page 35) and the EQT X data from company 'ABC Health', "
        "there is evidence of AI-driven diagnostics improving patient outcomes."
    )

    example_q3 = "How is AI driving new revenue streams in e-commerce according to 2024 data?"
    example_a3 = (
        "In the 2024 AI Index report (Pages 45-46), AI-driven product recommendations have increased "
        "average order value. One EQT X portfolio company specializing in e-commerce solutions saw a 15% "
        "boost in sales from personalized recommendations."
    )

    # Build the context snippet references
    snippet_texts = []
    for c in context:
        ref_str = ""
        # Identify the source in a more explicit, structured format
        if c[2] == "stanford_report":
            ref_str = f"[source: Stanford AI Index, link: {c[1]}, page {c[3]}, paragraph {c[4]}]"
        elif c[2] == "EQT_X_Portfolio":
            ref_str = f"[source: EQT X portfolio, website: {c[1]}]"
        else:
            ref_str = "[source: Unknown or mislabeled context]"

        snippet_texts.append(f"Context snippet:\n{ref_str}\n{c[5]}\n")

    # Combine all snippet texts
    context_snippets = "\n".join(snippet_texts)

    # Construct the improved prompt
    prompt = f"""
You are a knowledgeable assistant referencing the Stanford AI Index Reports (2022, 2023, 2024) 
and EQT X portfolio data to explain how AI impacts EQT X fund's portfolio companies.

EQT X portfolio companies are:
- "AMCS": "https://www.amcsgroup.com/"
- "Avetta": "https://www.avetta.com/"
- "Billtrust": "https://www.billtrust.com/"
- "Dechra_Pharmaceuticals": "https://eqtgroup.com/current-portfolio/dechra-pharmaceuticals/"
- "Hantverksdata": "https://www.hantverksdata.se/"
- "UTA": "https://eqtgroup.com/current-portfolio/uta/"
- "Zeus": "https://eqtgroup.com/current-portfolio/zeus/"

Your objective is to provide a concise but specific answer to the user's query, focusing on:
- References to relevant pages or chapters from the Stanford AI Index if those pages are mentioned.
- Details about which EQT X portfolio company is involved if the snippet provides such data.

Below are multiple example Q&A pairs showing the style of referencing:

EXAMPLE Q1:
{example_q1}

EXAMPLE A1:
{example_a1}

EXAMPLE Q2:
{example_q2}

EXAMPLE A2:
{example_a2}

EXAMPLE Q3:
{example_q3}

EXAMPLE A3:
{example_a3}

Now, here are the context snippets from your knowledge base (Stanford reports and/or EQT X data):

{context_snippets}

USER QUESTION: {query}

Please follow these steps:
1. Identify which snippet(s) might be relevant to the question.
2. Summarize any key information from those snippets that addresses the question.
3. Provide a concise, direct answer with references to specific pages/companies if the snippets mention them.
4. If the snippets only partially answer the question, provide a best-effort summary and note any gaps.
5. If absolutely no snippet addresses the user’s question, you may reply: "I don't know based on the provided context."

Remember: Use the provided context whenever possible. 
If the question extends beyond the provided snippets, highlight that 
the context does not fully answer the question.
"""
    return prompt

def get_answer_from_gemini(query: str, context: str, source: str, api_key: str):
    """
    Calls the Google Gemini REST API with the given prompt_text.
    Returns the response as a string (if parsing is successful) or a raw JSON fallback.
    
    Example usage:
        response = call_gemini_api("Explain how AI works")
        print("Gemini response:", response)
    """
    prompt_text = build_prompt(query, context)
    
    # Gemini Endpoint 
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"

    # Gemini payload structure
    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt_text}
                ]
            }
        ]
    }
    
    # Set headers
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        # Make the POST request
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        # Parse the JSON response
        response_data = response.json()
        
        try:
            return response_data['candidates'][0]["content"]["parts"][0]["text"]  + "\n\n" + source
        except (KeyError, IndexError):
            # If the structure is different, you might need to dig deeper or return the full object.
            return response_data

    except requests.RequestException as e:
        # Handle network or HTTP errors
        print(f"Gemini API request failed: {e}")
        return None


# Function to query the OpenAI API
def get_answer_from_gpt(query: str, context: str, source: str, gpt_model: str, client) -> str:
    # Build the prompt
    prompt = build_prompt(query, context)

    # Create a chatbot
    completion = client.chat.completions.create(
        model=gpt_model,
        # Pre-define conversation messages for the possible roles 
        messages=[
            {"role": "user", "content": prompt},
        ]
    )
    response = json_repair.loads(completion.choices[0].message.content) + "\n\n" + str(source)
        
    return response

