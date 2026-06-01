from __future__ import annotations

import argparse

import wrds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schemas", nargs="+", required=True)
    parser.add_argument("--limit", type=int, default=120)
    args = parser.parse_args()

    conn = wrds.Connection()
    try:
        for schema in args.schemas:
            tabs = conn.raw_sql(
                f"""
                select table_name
                from information_schema.tables
                where table_schema = '{schema}'
                  and table_type = 'BASE TABLE'
                order by table_name
                """
            )
            print(f"\nSCHEMA {schema} COUNT {len(tabs)}")
            print(tabs["table_name"].head(args.limit).to_string(index=False))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

