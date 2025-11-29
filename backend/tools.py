import os
import logging
from sqlalchemy import create_engine, text
from langchain_openai import OpenAIEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_URL = os.getenv("DATABASE_URL")
engine = create_engine(DB_URL)

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

def run_sql_query(query: str) -> str:
    """
    Executes SQL with a safety cap on the output size.
    """
    if not query.strip().lower().startswith("select"):
        return "Error: Only SELECT queries are allowed."

    try:
        with engine.connect() as connection:
            logger.info(f"Executing SQL: {query}")
            result = connection.execute(text(query))
            keys = result.keys()
            rows = result.fetchall()
            
            if not rows:
                return "No results found."
            
            # Convert to list of dicts
            result_list = [dict(zip(keys, row)) for row in rows]
            result_str = str(result_list)
            
            # SAFETY TRUNCATION: Limit to ~15,000 characters (approx 4k tokens)
            # This leaves plenty of room for the LLM's own response.
            MAX_CHARS = 15000
            if len(result_str) > MAX_CHARS:
                return result_str[:MAX_CHARS] + "\n...(Result truncated due to size. Suggest refining the query.)"
            
            return result_str
            
    except Exception as e:
        logger.error(f"SQL Error: {e}")
        return f"SQL Error: {str(e)}"

def semantic_search(query: str, limit: int = 5) -> str:
    # (Same semantic_search function as before)
    try:
        query_emb = embeddings_model.embed_query(query)
        
        sql = text("""
        WITH search_results AS (
            SELECT 
                'Invoice' as type,
                i.invoice_number as id,
                'Invoice ' || i.invoice_number || ' from ' || m.name || ' on ' || i.date || '. Total: $' || i.total_amount as content,
                i.embedding <=> :emb as distance
            FROM invoices i
            JOIN merchants m ON i.merchant_id = m.id
            WHERE (i.embedding <=> :emb) < 0.6
            
            UNION ALL
            
            SELECT 
                'Item' as type,
                ii.description as id,
                ii.description || ' bought from ' || m.name || ' for $' || ii.unit_price as content,
                ii.embedding <=> :emb as distance
            FROM invoice_items ii
            JOIN invoices i ON ii.invoice_id = i.id
            JOIN merchants m ON i.merchant_id = m.id
            WHERE (ii.embedding <=> :emb) < 0.6
        )
        SELECT type, content, distance
        FROM search_results
        ORDER BY distance ASC
        LIMIT :limit;
        """)
        
        with engine.connect() as connection:
            results = connection.execute(sql, {"emb": str(query_emb), "limit": limit}).fetchall()
            
        if not results:
            return "No relevant invoices or items found."
            
        formatted = "\n".join([f"- {r[1]}" for r in results])
        return formatted

    except Exception as e:
        logger.error(f"Vector Error: {e}")
        return f"Vector Search Error: {str(e)}"