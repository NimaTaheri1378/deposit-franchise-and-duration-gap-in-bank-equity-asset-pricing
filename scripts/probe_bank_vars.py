from __future__ import annotations

import argparse

import wrds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--terms", nargs="+", required=True)
    parser.add_argument("--limit", type=int, default=80)
    args = parser.parse_args()
    conn = wrds.Connection()
    try:
        for term in args.terms:
            safe = term.replace("'", "''")
            q = f"""
                select reporting_form, schedule, variable_name, mnemonic, item_code,
                       start_date, end_date, item_name, "table"
                from bank_all.wrds_bank_reg_vars
                where lower(item_name) like lower('%%{safe}%%')
                   or lower(mnemonic) like lower('%%{safe}%%')
                   or lower(variable_name) like lower('%%{safe}%%')
                order by start_date desc nulls last, variable_name
                limit {int(args.limit)}
            """
            rows = conn.raw_sql(q)
            print(f"\nTERM {term} COUNT_SHOWN {len(rows)}")
            if rows.empty:
                continue
            print(rows.to_string(index=False))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
