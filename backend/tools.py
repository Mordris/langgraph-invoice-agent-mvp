import os
import logging
from sqlalchemy import create_engine, text
from langchain_openai import OpenAIEmbeddings

# Configure Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_URL = os.getenv("DATABASE_URL")
engine = create_engine(DB_URL)

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

def run_sql_query(query: str) -> str:
    """
    Executes a read-only SQL query against the database.
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
            
            # Format as string
            result_str = [dict(zip(keys, row)) for row in rows]
            return str(result_str)
            
    except Exception as e:
        logger.error(f"SQL Error: {e}")
        return f"SQL Error: {str(e)}"

def semantic_search(query: str, limit: int = 5) -> str:
    """
    Embeds the query and searches invoices and items.
    CRITICAL FIX: Joins with merchants table to provide context (Who sold it?).
    """
    try:
        query_emb = embeddings_model.embed_query(query)
        
        # We perform two separate searches (Invoices & Items) and union them.
        # This is often cleaner/faster than a complex CTE for heterogenous data.
        
        sql = text("""
        WITH search_results AS (
            -- 1. Search Invoice Summaries
            SELECT 
                'Invoice' as type,
                i.invoice_number as id,
                -- We include Merchant Name in the result content
                'Invoice ' || i.invoice_number || ' from ' || m.name || ' on ' || i.date || '. Total: $' || i.total_amount as content,
                i.embedding <=> :emb as distance
            FROM invoices i
            JOIN merchants m ON i.merchant_id = m.id
            WHERE (i.embedding <=> :emb) < 0.6
            
            UNION ALL
            
            -- 2. Search Specific Items
            SELECT 
                'Item' as type,
                ii.description as id,
                -- CRITICAL FIX: Include Merchant Name here!
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
            
        # Format the context for the LLM
        formatted = "\n".join([f"- {r[1]}" for r in results])
        logger.info(f"Vector Search found {len(results)} matches.")
        return formatted

    except Exception as e:
        logger.error(f"Vector Error: {e}")
        return f"Vector Search Error: {str(e)}"