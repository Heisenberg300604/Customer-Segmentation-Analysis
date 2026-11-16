"""OPTIONAL: load the segmented data into PostgreSQL for the SQL / Power BI path.

The Streamlit app does NOT need a database. This script is only for running
sql/02_business_queries.sql or connecting Power BI.

Setup (one time):
    pip install -r requirements-db.txt
    export DATABASE_URL="postgresql+psycopg2://USER:PASSWORD@localhost:5432/DBNAME"

Run:
    python -m src.load_to_postgres
"""
from __future__ import annotations

import os
import sys

import pandas as pd

from src import config


def main() -> None:
    url = os.environ.get("DATABASE_URL")
    if not url:
        sys.exit("Set the DATABASE_URL environment variable first, e.g.\n"
                 '  export DATABASE_URL="postgresql+psycopg2://user:pass@localhost:5432/db"')
    if not config.CLUSTERED_CSV_PATH.exists():
        sys.exit("Run `python -m src.train` first to create clustered_customers.csv.")

    from sqlalchemy import create_engine, text  # imported here: optional dependency

    engine = create_engine(url)
    ddl = (config.ROOT_DIR / "sql" / "01_create_table.sql").read_text()
    df = pd.read_csv(config.CLUSTERED_CSV_PATH)
    with engine.begin() as conn:
        # Drop comment lines, then run each ';'-separated statement.
        sql = "\n".join(line for line in ddl.splitlines()
                        if not line.strip().startswith("--"))
        for statement in sql.split(";"):
            if statement.strip():
                conn.execute(text(statement))
        df.to_sql("customer", conn, if_exists="append", index=False)
    print(f"Loaded {len(df):,} rows into table 'customer'.")


if __name__ == "__main__":
    main()
