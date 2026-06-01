from __future__ import annotations

import argparse

import wrds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("codes", nargs="+")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()
    conn = wrds.Connection()
    try:
        for code in args.codes:
            safe = code.replace("'", "''")
            rows = conn.raw_sql(
                f"""
                select reporting_form, schedule, variable_name, start_date, end_date,
                       item_name, "table"
                from bank_all.wrds_bank_reg_vars
                where item_code = '{safe}'
                order by variable_name, start_date
                limit {int(args.limit)}
                """
            )
            print(f"\nITEM_CODE {code} COUNT_SHOWN {len(rows)}")
            if not rows.empty:
                print(rows.to_string(index=False))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

