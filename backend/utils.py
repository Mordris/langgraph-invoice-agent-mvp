import os
import redis
from sqlalchemy import create_engine, inspect

# Singleton Redis Connection
# decode_responses=True ensures we get Strings back, not Bytes
r = redis.Redis.from_url(os.getenv("REDIS_URL"), decode_responses=True)

def get_token_friendly_schema(db_url: str) -> str:
    """
    Introspects the database and returns a minified schema string.
    Caches the result in Redis for 24 hours to improve performance.
    
    Format:
    Table merchants(id:UUID, name:TEXT, address:TEXT)
    Table invoices(id:UUID, total_amount:NUMERIC, ...)
    """
    cache_key = "db_schema_minified"
    cached_schema = r.get(cache_key)
    
    if cached_schema:
        return cached_schema

    # If not in cache, generate it
    engine = create_engine(db_url)
    inspector = inspect(engine)
    schema_lines: list[str] = []

    # Filter only relevant tables (exclude vector migrations or system tables if any)
    allowed_tables = {'merchants', 'invoices', 'invoice_items'}
    
    for table_name in inspector.get_table_names():
        if table_name not in allowed_tables:
            continue
            
        columns: list[str] = []
        for col in inspector.get_columns(table_name):
            # Format: "name:type" (e.g., "total_amount:NUMERIC")
            # We strip complex type details to save tokens
            col_type = str(col['type']).split('(')[0]
            columns.append(f"{col['name']}:{col_type}")
        
        schema_lines.append(f"Table {table_name} ({', '.join(columns)})")

    final_schema = "\n".join(schema_lines)
    
    # Cache for 24 hours (86400 seconds)
    r.setex(cache_key, 86400, final_schema)
    
    return final_schema