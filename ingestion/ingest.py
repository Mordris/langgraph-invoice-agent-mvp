import os
import random
import uuid
import time
from datetime import datetime, timedelta
import psycopg2
from langchain_openai import OpenAIEmbeddings
from lxml import etree
import copy

# Configuration
DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise ValueError("DATABASE_URL environment variable is missing")

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Official UBL 2.1 Namespaces
NS = {
    'cbc': 'urn:oasis:names:specification:ubl:schema:xsd:CommonBasicComponents-2',
    'cac': 'urn:oasis:names:specification:ubl:schema:xsd:CommonAggregateComponents-2',
    'inv': 'urn:oasis:names:specification:ubl:schema:xsd:Invoice-2'
}

def get_db_connection():
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
    print("Setting up schema...")
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("CREATE EXTENSION IF NOT EXISTS vectorscale;")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS merchants (
            id UUID PRIMARY KEY,
            name TEXT NOT NULL,
            address TEXT
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS invoices (
            id UUID PRIMARY KEY,
            merchant_id UUID REFERENCES merchants(id),
            invoice_number TEXT,
            date DATE NOT NULL,
            total_amount NUMERIC(10, 2),
            tax_amount NUMERIC(10, 2),
            raw_xml TEXT,
            embedding vector(1536)
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS invoice_items (
            id UUID PRIMARY KEY,
            invoice_id UUID REFERENCES invoices(id),
            description TEXT NOT NULL,
            quantity INTEGER,
            unit_price NUMERIC(10, 2),
            total_line_amount NUMERIC(10, 2),
            embedding vector(1536)
        );
    """)
    print("Schema setup complete.")

def create_invoice_from_template(template_tree, data):
    """
    Takes the real UBL XML tree and modifies it with new data.
    """
    root = copy.deepcopy(template_tree)
    
    # Helper to set text
    def set_text(xpath, value, element=root):
        nodes = element.xpath(xpath, namespaces=NS)
        if nodes:
            nodes[0].text = str(value)
            
    # 1. Header Info
    set_text('//cbc:ID', data['invoice_number'])
    set_text('//cbc:IssueDate', data['date'])
    
    # 2. Supplier (Merchant)
    set_text('//cac:AccountingSupplierParty/cac:Party/cac:PartyName/cbc:Name', data['merchant_name'])
    set_text('//cac:AccountingSupplierParty/cac:Party/cac:PostalAddress/cbc:StreetName', data['merchant_address'])
    
    # 3. Totals
    set_text('//cac:TaxTotal/cbc:TaxAmount', f"{data['tax']:.2f}")
    set_text('//cac:LegalMonetaryTotal/cbc:PayableAmount', f"{data['total']:.2f}")
    set_text('//cac:LegalMonetaryTotal/cbc:TaxInclusiveAmount', f"{data['total']:.2f}")
    set_text('//cac:LegalMonetaryTotal/cbc:TaxExclusiveAmount', f"{data['total'] - data['tax']:.2f}")

    # 4. Items (This is tricky in XML, we remove existing and add new)
    # Find the parent of InvoiceLine (which is the root Invoice)
    # Remove existing lines from template
    for line in root.xpath('//cac:InvoiceLine', namespaces=NS):
        line.getparent().remove(line)
        
    # Create new lines
    # We cheat a bit by parsing a mini-template for the line or building it manually.
    # For MVP speed, let's use a string template for lines and append them as elements.
    for idx, item in enumerate(data['items']):
        line_xml = f"""
        <cac:InvoiceLine xmlns:cac="urn:oasis:names:specification:ubl:schema:xsd:CommonAggregateComponents-2"
                         xmlns:cbc="urn:oasis:names:specification:ubl:schema:xsd:CommonBasicComponents-2">
            <cbc:ID>{idx + 1}</cbc:ID>
            <cbc:InvoicedQuantity unitCode="EA">{item['qty']}</cbc:InvoicedQuantity>
            <cbc:LineExtensionAmount currencyID="USD">{item['line_total']:.2f}</cbc:LineExtensionAmount>
            <cac:Item>
                <cbc:Description>{item['name']}</cbc:Description>
            </cac:Item>
            <cac:Price>
                <cbc:PriceAmount currencyID="USD">{item['price']:.2f}</cbc:PriceAmount>
            </cac:Price>
        </cac:InvoiceLine>
        """
        line_element = etree.fromstring(line_xml)
        root.append(line_element)
        
    return etree.tostring(root, pretty_print=True).decode('utf-8')

def parse_ubl_invoice(xml_content):
    """
    Parses a UBL XML string using proper namespaces.
    """
    root = etree.fromstring(xml_content.encode('utf-8'))
    
    def get_text(xpath, element=root):
        res = element.xpath(xpath, namespaces=NS)
        return res[0].text if res else None

    return {
        'invoice_number': get_text('//cbc:ID'),
        'date': get_text('//cbc:IssueDate'),
        'merchant_name': get_text('//cac:AccountingSupplierParty/cac:Party/cac:PartyName/cbc:Name'),
        'merchant_address': get_text('//cac:AccountingSupplierParty/cac:Party/cac:PostalAddress/cbc:StreetName'),
        'total_amount': float(get_text('//cac:LegalMonetaryTotal/cbc:PayableAmount') or 0),
        'tax_amount': float(get_text('//cac:TaxTotal/cbc:TaxAmount') or 0),
        'items': [
            {
                'description': get_text('cac:Item/cbc:Description', line),
                'quantity': int(float(get_text('cbc:InvoicedQuantity', line) or 0)),
                'unit_price': float(get_text('cac:Price/cbc:PriceAmount', line) or 0),
                'total_line_amount': float(get_text('cbc:LineExtensionAmount', line) or 0)
            }
            for line in root.xpath('//cac:InvoiceLine', namespaces=NS)
        ]
    }

def process_invoices(cur):
    # Load Master Template
    with open("templates/ubl_2.1_sample.xml", "rb") as f:
        master_template = etree.parse(f).getroot()

    merchants = [
        ("Samsung Electronics", "123 Tech Park"), 
        ("Apple Store", "5th Avenue NY"), 
        ("Starbucks", "Market St SF"), 
        ("Amazon AWS", "Seattle WA")
    ]
    products = [
        ("Samsung S24 Ultra", 1200.00), ("MacBook Pro", 2500.00), 
        ("Latte", 5.50), ("AWS EC2", 50.00), ("USB-C Cable", 15.00)
    ]

    print("Generating & Ingesting Invoices from Real UBL Template...")
    
    for _ in range(50):
        # 1. Prepare Data
        m_name, m_addr = random.choice(merchants)
        selected_items = random.choices(products, k=random.randint(1, 4))
        
        items_data = []
        running_total = 0
        for p_name, p_price in selected_items:
            qty = random.randint(1, 3)
            l_tot = p_price * qty
            running_total += l_tot
            items_data.append({'name': p_name, 'price': p_price, 'qty': qty, 'line_total': l_tot})
            
        tax = running_total * 0.10
        total = running_total + tax
        
        # 2. Generate XML (The "Download" step)
        raw_xml = create_invoice_from_template(master_template, {
            'invoice_number': f"INV-{random.randint(10000, 99999)}",
            'date': (datetime.now() - timedelta(days=random.randint(0, 90))).strftime('%Y-%m-%d'),
            'merchant_name': m_name,
            'merchant_address': m_addr,
            'tax': tax,
            'total': total,
            'items': items_data
        })
        
        # 3. Parse XML (The Ingestion Step)
        parsed = parse_ubl_invoice(raw_xml)
        
        # 4. Store in DB
        # Merchant
        m_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, parsed['merchant_name']))
        cur.execute(
            "INSERT INTO merchants (id, name, address) VALUES (%s, %s, %s) ON CONFLICT (id) DO NOTHING",
            (m_id, parsed['merchant_name'], parsed['merchant_address'])
        )

        # Invoice
        inv_id = str(uuid.uuid4())
        items_str = ", ".join([f"{i['quantity']}x {i['description']}" for i in parsed['items']])
        summary = f"Invoice {parsed['invoice_number']} from {parsed['merchant_name']}. Date: {parsed['date']}. Total: ${parsed['total_amount']}. Items: {items_str}."
        inv_emb = embeddings_model.embed_query(summary)
        
        cur.execute(
            """INSERT INTO invoices 
               (id, merchant_id, invoice_number, date, total_amount, tax_amount, raw_xml, embedding)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
            (inv_id, m_id, parsed['invoice_number'], parsed['date'], parsed['total_amount'], parsed['tax_amount'], raw_xml, inv_emb)
        )

        # Items
        for item in parsed['items']:
            item_text = f"{item['description']} from {parsed['merchant_name']} at ${item['unit_price']}"
            item_emb = embeddings_model.embed_query(item_text)
            cur.execute(
                """INSERT INTO invoice_items 
                   (id, invoice_id, description, quantity, unit_price, total_line_amount, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (str(uuid.uuid4()), inv_id, item['description'], item['quantity'], item['unit_price'], item['total_line_amount'], item_emb)
            )

    # Indexes
    cur.execute("CREATE INDEX IF NOT EXISTS invoice_emb_idx ON invoices USING diskann (embedding);")
    cur.execute("CREATE INDEX IF NOT EXISTS item_emb_idx ON invoice_items USING diskann (embedding);")
    print("Ingestion Complete.")

if __name__ == "__main__":
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        setup_schema(cur)
        cur.execute("SELECT COUNT(*) FROM invoices")
        if cur.fetchone()[0] == 0:
            process_invoices(cur)
        else:
            print("Data exists.")
        cur.close()
    finally:
        conn.close()