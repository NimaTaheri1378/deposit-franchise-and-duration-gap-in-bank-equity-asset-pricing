from __future__ import annotations

import argparse

import wrds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tables", nargs="+", help="Fully qualified schema.table names")
    args = parser.parse_args()

    conn = wrds.Connection()
    try:
        for full in args.tables:
            schema, table = full.split(".", 1)
            cols = conn.raw_sql(
                f"""
                select column_name, data_type, ordinal_position
                from information_schema.columns
                where table_schema = '{schema}'
                  and table_name = '{table}'
                order by ordinal_position
                """
            )
            print(f"\nTABLE {full} COUNT {len(cols)}")
            print(cols[["column_name", "data_type"]].to_string(index=False))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

