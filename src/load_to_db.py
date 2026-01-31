"""
Load structured JSON data into MySQL Database with Enhanced Schema.
Supports full pipeline artifacts including Docling tables, OCR verification, and Figure captions.

Usage:
    python src/load_to_db.py --doc-name "2023_HDEC_Report" [--init-db]
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pymysql
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database Configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "esg_reports") # Default DB name updated
DB_PORT = int(os.getenv("DB_PORT", 3306))

DEFAULT_INPUT_DIR = Path("data/pages_structured")

# Regex for parsing cell values (number + unit)
# Matches patterns like: "1,234.56", "1,234.56 tCO2eq", "45%", "-12.5"
# Group 1: Number part (with commas)
# Group 3: Unit part (optional, remainder)
NUMBER_PATTERN = re.compile(r"^([-+]?[\d,]+(?:\.\d+)?)\s*(.*)$")
UNIT_DIRECT_PATTERN = re.compile(r"단위[:：]\s*([^\s)]+)")
UNIT_PAREN_PATTERN = re.compile(r"\(단위[:：]?\s*([^)]+)\)")
UNIT_SUFFIX_PATTERN = re.compile(r"\(([^)\d]{1,10})\)\s*$")
UNIT_VALID_CHARS = re.compile(r"[A-Za-z가-힣%°℃/·]")


def get_connection():
    try:
        conn = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=DB_PORT,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
        )
        return conn
    except pymysql.MySQLError as e:
        print(f"Error connecting to MySQL: {e}")
        print("Please check your .env file and ensure MySQL is properly configured.")
        sys.exit(1)


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA-256 hash of a file to detect duplicates."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def parse_filename(filename: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Parse filename to extract Year and Company Name.
    Expected format: [Year]_[Company]_Report.pdf or similar.
    Example: 2023_HDEC_Report.pdf -> (HDEC, 2023)
    """
    # Regex to find year (4 digits) and assuming structure
    match = re.search(r"(\d{4})_([^_]+)", filename)
    if match:
        year = int(match.group(1))
        company = match.group(2)
        return company, year
    
    # Fallback: try to find just a year
    year_match = re.search(r"20\d{2}", filename)
    year = int(year_match.group(0)) if year_match else None
    return "Unknown", year


def extract_unit_hint(text: str | None) -> Optional[str]:
    if not text:
        return None
    text = text.strip()
    match = UNIT_PAREN_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    match = UNIT_DIRECT_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    match = UNIT_SUFFIX_PATTERN.search(text)
    if match:
        candidate = match.group(1).strip()
        if candidate and len(candidate) <= 12 and UNIT_VALID_CHARS.search(candidate):
            return candidate
    return None


def infer_table_units(cells: List[List[dict]]) -> tuple[Optional[str], dict[int, str]]:
    global_unit: Optional[str] = None
    column_units: dict[int, str] = {}
    for row in cells[:3]:  # 첫 몇 행만 검사
        for cell in row:
            text = (cell.get("text") or "").strip()
            if not text:
                continue
            unit_hint = extract_unit_hint(text)
            if not unit_hint:
                continue
            col_idx = cell.get("col")
            if "단위" in text and (not cell.get("column_header") or len(text) <= len(unit_hint) + 6):
                if not global_unit:
                    global_unit = unit_hint
                continue
            if cell.get("column_header") and col_idx is not None and col_idx not in column_units:
                column_units[col_idx] = unit_hint
    return global_unit, column_units


def parse_cell_value(text: str, fallback_unit: Optional[str] = None) -> Tuple[Optional[float], Optional[str], str]:
    """
    Parse cell text to extract numeric value and unit.
    Returns: (numeric_value, unit, content_type)
    content_type is enum: 'text', 'number', 'date' (simple detection)
    """
    if not text:
        return None, None, 'text'

    text = text.strip()
    match = NUMBER_PATTERN.match(text)
    if match:
        num_str = match.group(1).replace(",", "")
        unit = match.group(2).strip()
        # Truncate unit if too long (likely not a real unit if > 50 chars)
        if len(unit) > 50:
            unit = unit[:50]
            
        try:
            val = float(num_str)
            resolved_unit = unit or fallback_unit
            return val, resolved_unit, 'number'
        except ValueError:
            pass
            
    return None, None, 'text'


def init_database(conn):
    """Create tables if they don't exist based on the new schema."""
    print("Initializing database schema...")
    with conn.cursor() as cursor:
        # 1. Documents
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INT AUTO_INCREMENT PRIMARY KEY,
                filename VARCHAR(255) NOT NULL UNIQUE,
                company_name VARCHAR(100),
                report_year INT,
                file_hash VARCHAR(64) UNIQUE,
                total_pages INT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_company_year (company_name, report_year),
                INDEX idx_file_hash (file_hash)
            )
        """)

        # 2. Pages
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pages (
                id INT AUTO_INCREMENT PRIMARY KEY,
                doc_id INT NOT NULL,
                page_no INT NOT NULL,
                visual_density FLOAT DEFAULT 0.0,
                has_tables BOOLEAN DEFAULT FALSE,
                has_figures BOOLEAN DEFAULT FALSE,
                needs_review BOOLEAN DEFAULT FALSE,
                full_markdown LONGTEXT,
                image_path VARCHAR(255),
                FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE,
                UNIQUE KEY idx_doc_page (doc_id, page_no)
            )
        """)

        # 3. Doc Tables (Refined Schema)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS doc_tables (
                id INT AUTO_INCREMENT PRIMARY KEY,
                page_id INT NOT NULL,
                doc_id INT NOT NULL,
                page_no INT NOT NULL,
                table_index INT NOT NULL,
                title VARCHAR(500),
                bbox_json JSON,
                diff_data JSON,
                image_path VARCHAR(255),
                FOREIGN KEY (page_id) REFERENCES pages(id) ON DELETE CASCADE,
                FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE,
                UNIQUE KEY idx_page_table (page_id, table_index),
                INDEX idx_table_doc (doc_id),
                INDEX idx_table_doc_page (doc_id, page_no)
            )
        """)

        # 4. Table Cells (Enhanced with numeric parsing)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS table_cells (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                table_id INT NOT NULL,
                doc_id INT NOT NULL,
                row_idx INT NOT NULL,
                col_idx INT NOT NULL,
                content TEXT,
                content_type ENUM('text', 'number', 'date') DEFAULT 'text',
                numeric_value DECIMAL(20, 4),
                unit VARCHAR(100),
                row_span INT DEFAULT 1,
                col_span INT DEFAULT 1,
                is_header BOOLEAN DEFAULT FALSE,
                bbox_json JSON,
                FOREIGN KEY (table_id) REFERENCES doc_tables(id) ON DELETE CASCADE,
                FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE,
                INDEX idx_cell_table (table_id),
                INDEX idx_cell_position (doc_id, row_idx, col_idx),
                INDEX idx_cell_numeric (numeric_value)
            )
        """)
        
        # 5. Doc Figures
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS doc_figures (
                id INT AUTO_INCREMENT PRIMARY KEY,
                page_id INT NOT NULL,
                doc_id INT NOT NULL,
                page_no INT NOT NULL,
                caption TEXT,
                description TEXT,
                image_path VARCHAR(255),
                bbox_json JSON,
                FOREIGN KEY (page_id) REFERENCES pages(id) ON DELETE CASCADE,
                FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE,
                INDEX idx_figure_doc (doc_id),
                INDEX idx_figure_page (page_id),
                INDEX idx_figure_doc_page (doc_id, page_no)
            )
        """)
    conn.commit()
    
    # Schema Migration (Run after creation to ensure columns exist)
    with conn.cursor() as cursor:
        def _ensure_column(table, col, definition):
            cursor.execute(f"SHOW COLUMNS FROM {table} LIKE '{col}'")
            if not cursor.fetchone():
                print(f"Migrating: Adding '{col}' to '{table}'...")
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col} {definition}")

        _ensure_column("doc_tables", "page_no", "INT NOT NULL DEFAULT 0")
        _ensure_column("doc_tables", "image_path", "VARCHAR(255)")
        _ensure_column("doc_figures", "page_no", "INT NOT NULL DEFAULT 0")
        
    conn.commit()
    print("Schema initialized and verified.")


def insert_document(conn, doc_name: str, input_dir: Path) -> int:
    """Insert or retrieve document metadata including parsing company/year."""
    # Try to find the original PDF to compute hash (optional, but good practice)
    # Here we assume doc_name might be the filename stem. 
    # To keep it simple, we use the doc_name as filename for now, 
    # or infer real filename if possible. 
    # But input_dir is the structured output dir.
    
    # Parse metadata from doc_name (assuming it's a file stem e.g., 2023_HDEC_Report)
    company, year = parse_filename(doc_name)
    filename = f"{doc_name}.pdf" # Best guess
    
    # Calculate rudimentary hash from doc_name if real file missing, 
    # or just use doc_name uniqueness.
    # We will skip real file hash for now to avoid dependency on input folder path here.
    file_hash = hashlib.sha256(doc_name.encode()).hexdigest()

    with conn.cursor() as cursor:
        cursor.execute("SELECT id FROM documents WHERE filename = %s", (filename,))
        row = cursor.fetchone()
        if row:
            return row["id"]
        
        cursor.execute("""
            INSERT INTO documents (filename, company_name, report_year, file_hash) 
            VALUES (%s, %s, %s, %s)
        """, (filename, company, year, file_hash))
        conn.commit()
        return cursor.lastrowid


def load_file_content(path: Path) -> Optional[str]:
    if path and path.exists():
        return path.read_text(encoding="utf-8")
    return None


def load_json_file(path: Path) -> Any:
    if path and path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def load_page(conn, doc_id: int, page_dir: Path):
    page_json_path = page_dir / "page.json"
    if not page_json_path.exists():
        return

    data = load_json_file(page_json_path)
    page_no = data.get("page_number")
    markdown = data.get("markdown", "")
    visual_density = data.get("visual_density", 0.0)
    needs_review = data.get("needs_visual_review", False)
    

    image_rel_path = data.get("page_image_path")
    
    tables_list = data.get("tables", [])
    figures_list = data.get("figures", [])
    
    has_tables = len(tables_list) > 0
    has_figures = len(figures_list) > 0


    with conn.cursor() as cursor:
        # Upsert Page
        sql = """
            INSERT INTO pages (doc_id, page_no, visual_density, has_tables, has_figures, needs_review, full_markdown, image_path)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                visual_density = VALUES(visual_density),
                has_tables = VALUES(has_tables),
                has_figures = VALUES(has_figures),
                needs_review = VALUES(needs_review),
                full_markdown = VALUES(full_markdown),
                image_path = VALUES(image_path)
        """
        cursor.execute(sql, (doc_id, page_no, visual_density, has_tables, has_figures, needs_review, markdown, image_rel_path))
        
        cursor.execute("SELECT id FROM pages WHERE doc_id=%s AND page_no=%s", (doc_id, page_no))
        page_id = cursor.fetchone()["id"]

        # Process Tables (with index)
        for idx, tbl_meta in enumerate(tables_list, 1):
            load_table(conn, doc_id, page_id, page_no, page_dir, tbl_meta, idx)
            
        # Process Figures (with inferred index logic inside)
        for fig_meta in figures_list:
            load_figure(conn, doc_id, page_id, page_no, page_dir, fig_meta)
    
    conn.commit()
    print(f"Loaded Page {page_no} (ID: {page_id})")


def load_table(conn, doc_id: int, page_id: int, page_no: int, page_dir: Path, tbl_meta: Dict[str, Any], table_index: int):
    table_id_str = tbl_meta.get("id") # table_001
    title = tbl_meta.get("title")
    bbox = tbl_meta.get("bbox")
    
    # Resolve Paths
    tables_dir = page_dir / "tables"
    json_path = tables_dir / f"{table_id_str}.json"
    diff_path = tables_dir / f"{table_id_str}.diff.json"
    image_rel_path = tbl_meta.get("image_path")

    # Load Data
    # refined logic: if we have a VLM re-parsed result (indicated by ocr_path ending in .vlm.json), use it as the source of truth.
    ocr_path = tbl_meta.get("ocr_path")
    
    # If ocr_path points to a VLM result, prefer it
    if ocr_path and ocr_path.endswith(".vlm.json"):
        # ocr_path is relative to structured_dir usually, but let's check
        # In structured_extract/ocr, we save it relative to structured_dir.
        # page_dir is structured_dir/page_XXXX.
        # ocr_path might be "page_XXXX/tables/table_001.vlm.json"
        
        # We need to construct full path.
        # page_dir.parent should be structured_dir
        # But clearer to just use page_dir / "tables" and filename if we know it?
        # Actually ocr_path in page.json is typically "page_0017/tables/table_001.vlm.json".
        vlm_full_path = page_dir.parent / ocr_path
        if vlm_full_path.exists():
            print(f"    [Info] Using VLM result for table {table_id_str}")
            # VLM outcome is list of cells, but we need to wrap it in "cells" key if load_table expects {"cells": [...]}
            # My table_vlm.py returns {"cells": [...]}. So it matches.
            table_data = load_json_file(vlm_full_path)
        else:
             # Fallback
             print(f"    [Warn] VLM path {vlm_full_path} missing, falling back to Docling.")
             table_data = load_json_file(json_path)
    else:
        table_data = load_json_file(json_path)

    if not table_data:
        return

    diff_data = load_json_file(diff_path)
    diff_json_str = json.dumps(diff_data) if diff_data else None
    
    cells = table_data.get("cells", [])
    global_unit, column_units = infer_table_units(cells)

    with conn.cursor() as cursor:
        # Upsert Table Meta
        cursor.execute("""
            INSERT INTO doc_tables (page_id, doc_id, page_no, table_index, title, bbox_json, diff_data, image_path)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                title = VALUES(title),
                bbox_json = VALUES(bbox_json),
                diff_data = VALUES(diff_data),
                image_path = VALUES(image_path),
                page_no = VALUES(page_no)
        """, (page_id, doc_id, page_no, table_index, title, json.dumps(bbox), diff_json_str, image_rel_path))
        
        # Get Table ID
        cursor.execute("SELECT id FROM doc_tables WHERE page_id=%s AND table_index=%s", (page_id, table_index))
        db_table_id = cursor.fetchone()["id"]

        # Re-insert Cells
        cursor.execute("DELETE FROM table_cells WHERE table_id = %s", (db_table_id,))
        
        if not cells:
            return
            
        insert_data = []
        for row_list in cells:
            for cell in row_list:
                content = cell.get("text", "")
                col_idx = cell.get("col")
                fallback_unit = column_units.get(col_idx) if col_idx is not None else None
                if not fallback_unit:
                    fallback_unit = global_unit
                numeric_val, unit, c_type = parse_cell_value(content, fallback_unit)
                
                insert_data.append((
                    db_table_id,
                    doc_id, # Denormalized FK
                    cell.get("row"),
                    cell.get("col"),
                    content,
                    c_type,
                    numeric_val,
                    unit,
                    cell.get("row_span", 1),
                    cell.get("col_span", 1),
                    cell.get("column_header", False) or cell.get("row_header", False),
                    None # bbox_json is NULL for now
                ))

        if insert_data:
            cursor.executemany("""
                INSERT INTO table_cells 
                (table_id, doc_id, row_idx, col_idx, content, content_type, numeric_value, unit, row_span, col_span, is_header, bbox_json)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, insert_data)


def load_figure(conn, doc_id: int, page_id: int, page_no: int, page_dir: Path, fig_meta: Dict[str, Any]):
    figure_id_str = fig_meta.get("id") # figure_001
    caption = fig_meta.get("caption")
    bbox = fig_meta.get("bbox")
    image_rel_path = fig_meta.get("image_path")
    
    # Check for description
    figures_dir = page_dir / "figures"
    desc_path = figures_dir / f"{figure_id_str}.desc.md"
    description = load_file_content(desc_path)
    
    with conn.cursor() as cursor:
        cursor.execute("""
            INSERT INTO doc_figures (page_id, doc_id, page_no, bbox_json, caption, description, image_path)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                caption = VALUES(caption),
                description = VALUES(description),
                image_path = VALUES(image_path),
                bbox_json = VALUES(bbox_json),
                page_no = VALUES(page_no)
        """, (page_id, doc_id, page_no, json.dumps(bbox), caption, description, image_rel_path))


def main():
    parser = argparse.ArgumentParser(description="Load extracted JSON data into MySQL")
    parser.add_argument("--doc-name", type=str, required=True, help="Document name (used as ID/Filename stem)")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory containing page_XXXX folders")
    parser.add_argument("--init-db", action="store_true", help="Initialize database schema (create tables)")
    
    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"Input directory not found: {args.input_dir}")
        return

    conn = get_connection()
    try:
        if args.init_db:
            init_database(conn)

        doc_id = insert_document(conn, args.doc_name, args.input_dir)
        print(f"Processing Document: {args.doc_name} (ID: {doc_id})")

        page_dirs = sorted([d for d in args.input_dir.iterdir() if d.is_dir() and d.name.startswith("page_")])
        if not page_dirs:
            print("No page directories found. Run structured_extract.py first.")
            return

        for page_dir in page_dirs:
            load_page(conn, doc_id, page_dir)

        # Update total_pages count
        total_extracted = len(page_dirs)
        with conn.cursor() as cursor:
            cursor.execute("UPDATE documents SET total_pages = %s WHERE id = %s", (total_extracted, doc_id))
        conn.commit()

        print("\nSuccess! Data loaded into database.")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
