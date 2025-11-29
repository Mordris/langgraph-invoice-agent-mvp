import os
import random
import uuid
import time
from datetime import datetime, timedelta
import psycopg2
from langchain_openai import OpenAIEmbeddings

# Configuration
DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise ValueError("DATABASE_URL environment variable is missing")

# Initialize Embeddings
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

def get_db_connection():
    """Wait for database to be ready and connect."""
    retries = 10
    while retries > 0:
        try:
            conn = psycopg2.connect(DB_URL)
            conn.autocommit = True
            return conn
        except psycopg2.OperationalError:
            print(f"Database not ready. Retrying... ({retries} left)")
            time.sleep(3)
            retries -= 1
    raise Exception("Could not connect to the database.")

def setup_schema(cur):
    """Enable extensions and create tables."""
    print("Setting up schema...")
    
    # 1. Extensions
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("CREATE EXTENSION IF NOT EXISTS vectorscale;") # For high-performance indexing

    # 2. Tables
    # Merchants
    cur.execute("""
        CREATE TABLE IF NOT EXISTS merchants (
            id UUID PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT,
            address TEXT
        );
    """)

    # Invoices (Head)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS invoices (
            id UUID PRIMARY KEY,
            merchant_id UUID REFERENCES merchants(id),
            invoice_number TEXT,
            date DATE NOT NULL,
            total_amount NUMERIC(10, 2),
            tax_amount NUMERIC(10, 2),
            currency TEXT DEFAULT 'USD',
            embedding vector(1536) -- Summary embedding
        );
    """)

    # Invoice Items (Lines)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS invoice_items (
            id UUID PRIMARY KEY,
            invoice_id UUID REFERENCES invoices(id),
            description TEXT NOT NULL,
            quantity INTEGER,
            unit_price NUMERIC(10, 2),
            total_line_amount NUMERIC(10, 2),
            embedding vector(1536) -- Item description embedding
        );
    """)
    print("Schema setup complete.")

def seed_data(cur):
    """Generate and insert synthetic data."""
    
    # Check if data already exists to prevent duplicates
    cur.execute("SELECT COUNT(*) FROM invoices")
    count = cur.fetchone()[0]
    if count > 0:
        print(f"Database already contains {count} invoices. Skipping seed.")
        return

    print("Generating synthetic data...")

    # 1. Mock Merchants
    merchants_data = [
        ("Samsung Electronics", "Electronics", "123 Tech Park, Seoul"),
        ("Apple Store", "Electronics", "5th Avenue, NY"),
        ("Starbucks", "Food & Beverage", "Market Street, SF"),
        ("Amazon Web Services", "Cloud Services", "Seattle, WA"),
        ("Uber", "Transportation", "Market St, SF"),
        ("Wallmart", "Retail", "Bentonville, AR")
    ]
    
    merchant_ids = []
    for m_name, m_cat, m_addr in merchants_data:
        m_id = str(uuid.uuid4())
        merchant_ids.append(m_id)
        cur.execute(
            "INSERT INTO merchants (id, name, category, address) VALUES (%s, %s, %s, %s)",
            (m_id, m_name, m_cat, m_addr)
        )

    # 2. Mock Items Pool
    tech_items = [
        ("Galaxy S24 Ultra", 1200.00), ("USB-C Cable", 15.00), ("Monitor 27 inch", 300.00),
        ("Mechanical Keyboard", 150.00), ("Mouse Pad", 20.00)
    ]
    food_items = [
        ("Latte", 5.50), ("Sandwich", 12.00), ("Bagel", 3.00), ("Cold Brew", 6.00)
    ]
    cloud_items = [
        ("EC2 Instance Usage", 50.00), ("S3 Storage", 12.50), ("RDS Database", 75.00)
    ]
    
    # 3. Generate Invoices
    for _ in range(50): # Generate 50 invoices
        m_idx = random.randint(0, len(merchants_data) - 1)
        m_id = merchant_ids[m_idx]
        m_name = merchants_data[m_idx][0]
        
        # Determine items based on merchant type
        if "Electronics" in merchants_data[m_idx][1]:
            pool = tech_items
        elif "Food" in merchants_data[m_idx][1]:
            pool = food_items
        elif "Cloud" in merchants_data[m_idx][1]:
            pool = cloud_items
        else:
            pool = tech_items + food_items # Generic
            
        # Select random items
        num_items = random.randint(1, 5)
        selected_items = random.choices(pool, k=num_items)
        
        inv_id = str(uuid.uuid4())
        inv_date = datetime.now() - timedelta(days=random.randint(0, 60))
        inv_num = f"INV-{random.randint(1000, 9999)}"
        
        running_total = 0
        items_summary = []

        # Insert Items
        for item_name, price in selected_items:
            qty = random.randint(1, 3)
            line_total = price * qty
            running_total += line_total
            
            # Embed Item
            item_text = f"{item_name} purchased from {m_name}"
            item_emb = embeddings_model.embed_query(item_text)
            
            cur.execute(
                """INSERT INTO invoice_items 
                   (id, invoice_id, description, quantity, unit_price, total_line_amount, embedding) 
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (str(uuid.uuid4()), inv_id, item_name, qty, price, line_total, item_emb)
            )
            items_summary.append(f"{qty}x {item_name}")

        # Insert Invoice
        tax = running_total * 0.10 # 10% tax
        total_with_tax = running_total + tax
        
        # Create a rich semantic summary for the invoice
        inv_summary_text = f"Invoice {inv_num} from {m_name} on {inv_date.strftime('%Y-%m-%d')}. \
                             Items: {', '.join(items_summary)}. \
                             Total: ${total_with_tax:.2f}. Category: {merchants_data[m_idx][1]}."
        
        inv_emb = embeddings_model.embed_query(inv_summary_text)
        
        cur.execute(
            """INSERT INTO invoices 
               (id, merchant_id, invoice_number, date, total_amount, tax_amount, embedding) 
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (inv_id, m_id, inv_num, inv_date, total_with_tax, tax, inv_emb)
        )
        
        print(f"Created Invoice {inv_num} for {m_name}")

    # 4. Create Index (pgvectorscale)
    # Using StreamingDiskANN index for performance on large datasets (mocked here)
    print("Creating Vector Indexes...")
    cur.execute("""
        CREATE INDEX IF NOT EXISTS invoice_embedding_idx 
        ON invoices 
        USING diskann (embedding);
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS item_embedding_idx 
        ON invoice_items 
        USING diskann (embedding);
    """)
    
    print("Data Seeding Complete.")

if __name__ == "__main__":
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        setup_schema(cur)
        seed_data(cur)
        cur.close()
    finally:
        conn.close()