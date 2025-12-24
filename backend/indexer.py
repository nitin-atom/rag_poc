import os
import time
import httpx
from bs4 import BeautifulSoup
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

INDEX_NAME = "rag-chatbot"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536

ARTICLES = [
    {
        "url": "https://helpdesk.atom.com/en/articles/5918587-how-to-sell-domains-in-wholesale-marketplace",
        "title": "How to Sell Domains in Wholesale Marketplace"
    },
    {
        "url": "https://helpdesk.atom.com/en/articles/2770338-sellers-how-to-update-your-name-servers",
        "title": "Sellers: How to Update Your Name Servers"
    }
]


def create_index_if_not_exists():
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:
        print(f"Creating index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print("Waiting for index to be ready...")
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)
        print("Index ready!")
    else:
        print(f"Index '{INDEX_NAME}' already exists.")


def fetch_article_content(url: str) -> str:
    response = httpx.get(url, follow_redirects=True)
    soup = BeautifulSoup(response.text, "html.parser")

    article = soup.find("article") or soup.find("main") or soup.find("div", class_="article")

    if article:
        for tag in article.find_all(["script", "style", "nav", "footer"]):
            tag.decompose()
        text = article.get_text(separator="\n", strip=True)
    else:
        text = soup.get_text(separator="\n", strip=True)

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n".join(lines)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        if end < text_length:
            space_idx = text.rfind(" ", start, end)
            if space_idx > start:
                end = space_idx

        chunks.append(text[start:end].strip())
        start = end - overlap if end < text_length else text_length

    return [c for c in chunks if c]


def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return response.data[0].embedding


def index_articles():
    create_index_if_not_exists()
    index = pc.Index(INDEX_NAME)

    try:
        index.delete(delete_all=True)
        print("Cleared existing vectors.")
    except Exception:
        print("No existing vectors to clear.")

    all_vectors = []

    for article in ARTICLES:
        print(f"\nProcessing: {article['title']}")
        content = fetch_article_content(article["url"])
        chunks = chunk_text(content)
        print(f"  Created {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            vector_id = f"{article['url'].split('/')[-1]}_{i}"
            embedding = get_embedding(chunk)

            all_vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "text": chunk,
                    "source_url": article["url"],
                    "title": article["title"],
                    "chunk_index": i
                }
            })

    print(f"\nUpserting {len(all_vectors)} vectors...")
    index.upsert(vectors=all_vectors)
    print("Indexing complete!")


if __name__ == "__main__":
    index_articles()
