import os
import redis
from sqlalchemy import create_engine, inspect

r = redis.Redis.from_url(os.getenv("REDIS_URL"), decode_responses=True)

def get_token_friendly_schema(db_url: str) -> str:
    """
    Returns a minified schema, HIDING heavy columns like raw_xml and embedding.
    """
    cache_key = "db_schema_minified_v2" # Changed key to force refresh
    cached_schema = r.get(cache_key)
    
    if cached_schema:
        return cached_schema

    engine = create_engine(db_url)
    inspector = inspect(engine)
    schema_lines = []

    allowed_tables = {'merchants', 'invoices', 'invoice_items'}
    # Columns to hide from the LLM to prevent context explosion
    blacklisted_cols = {'raw_xml', 'embedding', 'vectors'}
    
    for table_name in inspector.get_table_names():
        if table_name not in allowed_tables:
            continue
            
        columns = []
        for col in inspector.get_columns(table_name):
            col_name = col['name']
            if col_name in blacklisted_cols:
                continue
                
            col_type = str(col['type']).split('(')[0]
            columns.append(f"{col_name}:{col_type}")
        
        schema_lines.append(f"Table {table_name} ({', '.join(columns)})")

    final_schema = "\n".join(schema_lines)
    r.setex(cache_key, 86400, final_schema)
    
    return final_schema