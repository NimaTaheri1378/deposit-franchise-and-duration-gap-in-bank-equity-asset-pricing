from __future__ import annotations

import argparse

import wrds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tables", nargs="+")
    parser.add_argument("--patterns", nargs="+", required=True)
    args = parser.parse_args()
    conn = wrds.Connection()
    try:
        for full in args.tables:
            schema, table = full.split(".", 1)
            clauses = []
            for pat in args.patterns:
                safe = pat.replace("'", "''")
                clauses.append(f"lower(column_name) like lower('%%{safe}%%')")
            rows = conn.raw_sql(
                f"""
                select column_name, data_type
                from information_schema.columns
                where table_schema = '{schema}'
                  and table_name = '{table}'
                  and ({' or '.join(clauses)})
                order by column_name
                """
            )
            print(f"\nTABLE {full}")
            print(rows.to_string(index=False) if not rows.empty else "NO_MATCH")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

