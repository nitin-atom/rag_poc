import json
import os
from openai import OpenAI
from pinecone import Pinecone

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

INDEX_NAME = "rag-chatbot"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
SIMILARITY_THRESHOLD = 0.35


def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return response.data[0].embedding


def query_similar(question: str, top_k: int = 3) -> tuple[list[dict], bool]:
    index = pc.Index(INDEX_NAME)
    query_embedding = get_embedding(question)

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    contexts = []
    max_score = 0.0
    for match in results.matches:
        score = match.score
        max_score = max(max_score, score)
        contexts.append({
            "text": match.metadata.get("text", ""),
            "source_url": match.metadata.get("source_url", ""),
            "title": match.metadata.get("title", ""),
            "score": score
        })

    has_relevant_results = max_score >= SIMILARITY_THRESHOLD
    print(f"[DEBUG] Query: '{question}' | Max score: {max_score:.4f} | Above threshold: {has_relevant_results}")
    return contexts, has_relevant_results


def generate_answer(question: str, contexts: list[dict], has_relevant_results: bool) -> dict:
    context_text = "\n\n".join([
        f"[Source: {ctx['title']}]\n{ctx['text']}"
        for ctx in contexts
    ])

    sources = list({ctx["source_url"] for ctx in contexts if ctx["source_url"]})
    source_titles = {ctx["source_url"]: ctx["title"] for ctx in contexts if ctx["source_url"]}

    system_prompt = """You are a helpful assistant that answers questions about Atom's domain selling platform.
Use the provided context to answer questions accurately and concisely.
When you use information from the context, naturally incorporate references by mentioning the source article title.
If the context doesn't contain enough information to answer the question, say so honestly.
Keep your answers clear and to the point.

IMPORTANT: You must respond with a JSON object in this exact format:
{
    "answer": "Your helpful response here",
    "context_was_useful": true or false
}

Set "context_was_useful" to true ONLY if the provided context actually helped you answer the question.
Set it to false if:
- The question is nonsensical, gibberish, or not a real question
- The question is unrelated to the context (e.g., weather, general knowledge, random topics)
- The context doesn't contain information relevant to answering the question
- You had to answer without using the context at all"""

    user_prompt = f"""Context:
{context_text}

Question: {question}

Please provide a helpful answer based on the context above (if relevant).
Respond with a JSON object containing "answer" and "context_was_useful" fields."""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        max_tokens=500,
        response_format={"type": "json_object"}
    )

    response_text = response.choices[0].message.content

    try:
        parsed = json.loads(response_text)
        answer = parsed.get("answer", response_text)
        context_was_useful = parsed.get("context_was_useful", False)
    except json.JSONDecodeError:
        answer = response_text
        context_was_useful = False

    print(f"[DEBUG] LLM context_was_useful: {context_was_useful} | has_relevant_results: {has_relevant_results}")
    should_include_citations = has_relevant_results and context_was_useful
    print(f"[DEBUG] should_include_citations: {should_include_citations}")

    citations = []
    if should_include_citations:
        citations = [
            {"title": source_titles.get(url, "Source"), "url": url}
            for url in sources
        ]

    return {
        "answer": answer,
        "citations": citations
    }


def chat(question: str) -> dict:
    contexts, has_relevant_results = query_similar(question)
    return generate_answer(question, contexts, has_relevant_results)
