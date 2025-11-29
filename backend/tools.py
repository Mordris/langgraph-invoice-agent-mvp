import os
from sqlalchemy import create_engine, text
from langchain_openai import OpenAIEmbeddings
import numpy as np

# Database Connection
DB_URL = os.getenv("DATABASE_URL")
engine = create_engine(DB_URL)

# Embeddings Model (Must match ingestion model)
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

def run_sql_query(query: str) -> str:
    """
    Executes a read-only SQL query against the database.
    """
    # Basic safety check (In prod, use a read-only DB user)
    if not query.strip().lower().startswith("select"):
        raise ValueError("Only SELECT queries are allowed.")

    try:
        with engine.connect() as connection:
            result = connection.execute(text(query))
            keys = result.keys()
            rows = result.fetchall()
            
            if not rows:
                return "No results found."
            
            # Format as a string representation of a list of dicts
            result_str = [dict(zip(keys, row)) for row in rows]
            return str(result_str)
            
    except Exception as e:
        return f"SQL Error: {str(e)}"

def semantic_search(query: str, limit: int = 5) -> str:
    """
    Embeds the query and searches invoices and items using pgvector.
    Returns the most relevant text chunks.
    """
    try:
        query_emb = embeddings_model.embed_query(query)
        
        # We search both tables and combine results
        # Using the <=> operator for Cosine Distance
        
        sql = text("""
        WITH combined_search AS (
            -- Search Invoices
            SELECT 
                'Invoice' as type,
                invoice_number as id,
                raw_xml as content,
                embedding <=> :emb as distance
            FROM invoices
            WHERE (embedding <=> :emb) < 0.5 -- Threshold
            
            UNION ALL
            
            -- Search Items
            SELECT 
                'Item' as type,
                description as id,
                description || ' (Price: ' || unit_price || ')' as content,
                embedding <=> :emb as distance
            FROM invoice_items
            WHERE (embedding <=> :emb) < 0.5
        )
        SELECT type, content 
        FROM combined_search
        ORDER BY distance ASC
        LIMIT :limit;
        """)
        
        with engine.connect() as connection:
            results = connection.execute(sql, {"emb": str(query_emb), "limit": limit}).fetchall()
            
        if not results:
            return "No relevant invoices or items found in the semantic search."
            
        formatted = "\n".join([f"[{r[0]}] {r[1]}" for r in results])
        return formatted

    except Exception as e:
        return f"Vector Search Error: {str(e)}"