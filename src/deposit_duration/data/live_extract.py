from __future__ import annotations

from functools import reduce
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from deposit_duration.utils.manifest import write_manifest


def _connect_wrds() -> Any:
    try:
        import wrds  # type: ignore
    except ImportError as exc:
        raise RuntimeError("The wrds package is required for live extraction. Install .[wrds].") from exc
    return wrds.Connection()


def _sql_date(value: str) -> str:
    if not value.replace("-", "").isdigit() or len(value) != 10:
        raise ValueError(f"Expected YYYY-MM-DD date, got {value!r}")
    return value


def _existing_columns(conn: Any, schema: str, table: str) -> set[str]:
    cols = conn.raw_sql(
        f"""
        select column_name
        from information_schema.columns
        where table_schema = '{schema}'
          and table_name = '{table}'
        """
    )
    return set(cols["column_name"].str.lower())


def _pull_table(
    conn: Any,
    schema: str,
    table: str,
    date_col: str,
    start: str,
    end: str,
    columns: list[str],
    linked_only: bool = True,
    rssd_filter_sql: str | None = None,
) -> pd.DataFrame:
    existing = _existing_columns(conn, schema, table)
    keep = [c for c in columns if c.lower() in existing]
    if date_col not in keep:
        keep = [date_col, *keep]
    if "rssd9001" not in keep and "rssd9001" in existing:
        keep = ["rssd9001", *keep]
    keep = list(dict.fromkeys(keep))
    if len(keep) <= 2:
        return pd.DataFrame()
    link_filter = ""
    if "rssd9001" in existing:
        if rssd_filter_sql is not None:
            link_filter = f"and rssd9001 in ({rssd_filter_sql})"
        elif linked_only:
            link_filter = "and rssd9001 in (select distinct rssd9001 from bank_all.wrds_bank_crsp_link)"
    sql = f"""
        select {", ".join(keep)}
        from {schema}.{table}
        where {date_col} between '{start}' and '{end}'
        {link_filter}
    """
    out = conn.raw_sql(sql)
    out.columns = [c.lower() for c in out.columns]
    value_cols = [c for c in out.columns if c not in {"rssd9001", date_col.lower(), "rssdsubmissiondate"}]
    for col in value_cols:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].replace({4948807: np.nan, -4948807: np.nan})
    return out


def _pull_holding_table(
    conn: Any,
    table: str,
    start: str,
    end: str,
    columns: list[str],
) -> pd.DataFrame:
    out = _pull_table(
        conn,
        "bank_all",
        table,
        "rssd9999",
        start,
        end,
        ["rssd9001", "rssd9999", *columns],
        linked_only=True,
    )
    if out.empty:
        return out
    out = out.rename(columns={"rssd9999": "wrdsreportdate"})
    return out


def _offspring_filter_sql(start: str, end: str) -> str:
    return f"""
        select distinct id_rssd_offspring
        from bank_all.wrds_struct_relationships
        where id_rssd_parent in (
            select distinct rssd9001 from bank_all.wrds_bank_crsp_link where rssd9001 is not null
        )
          and id_rssd_offspring is not null
          and date_start <= '{end}'
          and coalesce(date_end, date '9999-12-31') >= '{start}'
          and (ctrl_ind = 1 or pct_equity >= 50)
    """


def _coalesce(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return pd.Series(np.nan, index=df.index)
    return df[existing].bfill(axis=1).iloc[:, 0]


def _merge_components(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise ValueError("No bank regulatory component frames were extracted.")
    for idx, frame in enumerate(frames):
        frames[idx] = frame.drop_duplicates(["rssd9001", "wrdsreportdate"])
    return reduce(
        lambda left, right: left.merge(right, on=["rssd9001", "wrdsreportdate"], how="outer"),
        frames,
    )


def _canonical_bank_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["report_date"] = pd.to_datetime(df["wrdsreportdate"])
    df["rssd9001"] = pd.to_numeric(df["rssd9001"], errors="coerce").astype("Int64")

    out = pd.DataFrame({"rssd9001": df["rssd9001"], "report_date": df["report_date"]})
    out["total_assets"] = _coalesce(
        df,
        [
            "bhck2170",
            "bhca2170",
            "bhct2170",
            "bhcp2170",
            "rcon2170",
            "rcfd2170",
            "rcoa2170",
            "rcfa2170",
        ],
    )
    out["total_deposits"] = _coalesce(df, ["rcon2200"])
    out["uninsured_deposits"] = _coalesce(df, ["rconl197"])
    out["brokered_deposits"] = _coalesce(df, ["rcon2365"])
    large_time_sum_cols = [c for c in ["rconhk12", "rconhk13", "rconhk14", "rconhk15"] if c in df.columns]
    large_time_sum = df[large_time_sum_cols].sum(axis=1, min_count=1) if large_time_sum_cols else pd.Series(np.nan, index=df.index)
    out["large_time_deposits"] = _coalesce(df, ["rconj474", "bhcbj474", "bhodj474"]).fillna(large_time_sum)
    out["noninterest_deposits"] = _coalesce(df, ["rcon6631", "rcfn6631", "bhdm6631", "bhfn6631"])
    out["securities_htm"] = _coalesce(df, ["bhck1754", "rcon1754", "rcfd1754"])
    out["securities_htm_fair"] = _coalesce(df, ["bhck1771", "rcon1771", "rcfd1771"])
    out["securities_afs"] = _coalesce(df, ["bhck1772", "rcon1772", "rcfd1772"])
    out["tier1_capital"] = _coalesce(
        df,
        [
            "bhcap859",
            "bhcwp859",
            "rcoap859",
            "rcfwp859",
            "rcfap859",
            "rcoa8274",
            "rcfa8274",
            "rcon8274",
            "rcfd8274",
            "bhck3210",
            "rcon3210",
            "rcfd3210",
        ],
    )
    out["cre_loans"] = _coalesce(df, ["bhckjj05", "rconjj05", "rcfdjj05", "rcon2746"])
    out["total_loans"] = _coalesce(df, ["bhck2122", "bhdm2122", "rcon2122", "rcfd2122"])
    nontrading = _coalesce(df, ["bhck8725", "rcon8725", "rcfd8725"])
    trading = _coalesce(df, ["bhcka126", "rcona126", "rcfda126"])
    out["interest_rate_derivatives"] = nontrading.fillna(0) + trading.fillna(0)
    out.loc[nontrading.isna() & trading.isna(), "interest_rate_derivatives"] = np.nan
    out["unrealized_losses"] = out["securities_htm_fair"] - out["securities_htm"]
    return out


def _build_bank_quarterly(raw: pd.DataFrame, link: pd.DataFrame, rel: pd.DataFrame | None = None) -> pd.DataFrame:
    out = _canonical_bank_features(raw)
    if rel is not None and not rel.empty:
        rel = rel.copy()
        rel["date_start"] = pd.to_datetime(rel["date_start"], errors="coerce").fillna(pd.Timestamp("1900-01-01"))
        rel["date_end"] = pd.to_datetime(rel["date_end"], errors="coerce").fillna(pd.Timestamp("2099-12-31"))
        out = out.merge(rel, left_on="rssd9001", right_on="id_rssd_offspring", how="inner")
        out = out[(out["report_date"] >= out["date_start"]) & (out["report_date"] <= out["date_end"])]
        sort_cols = [c for c in ["reln_lvl", "pct_equity"] if c in out.columns]
        if sort_cols:
            out = out.sort_values(sort_cols, ascending=[True, False][: len(sort_cols)])
        out = out.drop_duplicates(["id_rssd_parent", "id_rssd_offspring", "report_date"])
        out = out.drop(columns=["rssd9001", "id_rssd_offspring", "date_start", "date_end"])
        out = out.rename(columns={"id_rssd_parent": "rssd9001"})
        numeric_cols = [
            c
            for c in out.columns
            if c not in {"rssd9001", "report_date", "ctrl_ind", "pct_equity", "reln_lvl"}
        ]
        out = out.groupby(["rssd9001", "report_date"], as_index=False)[numeric_cols].sum(min_count=1)

    link = link.copy()
    link["dt_start"] = pd.to_datetime(link["dt_start"], errors="coerce").fillna(pd.Timestamp("1900-01-01"))
    link["dt_end"] = pd.to_datetime(link["dt_end"], errors="coerce").fillna(pd.Timestamp("2099-12-31"))
    out = out.merge(link[["rssd9001", "permco", "name", "inst_type", "dt_start", "dt_end"]], on="rssd9001", how="left")
    out = out[(out["report_date"] >= out["dt_start"]) & (out["report_date"] <= out["dt_end"])]
    out = out.drop(columns=["dt_start", "dt_end"]).drop_duplicates(["permco", "report_date"])
    return out


def _combine_holding_and_call(holding: pd.DataFrame, call: pd.DataFrame) -> pd.DataFrame:
    keys = ["permco", "report_date"]
    merged = holding.merge(call, on=keys, how="outer", suffixes=("_holding", "_call"))
    passthrough = ["rssd9001", "name", "inst_type"]
    value_cols = [
        "total_assets",
        "total_deposits",
        "uninsured_deposits",
        "brokered_deposits",
        "large_time_deposits",
        "noninterest_deposits",
        "securities_htm",
        "securities_htm_fair",
        "securities_afs",
        "unrealized_losses",
        "tier1_capital",
        "cre_loans",
        "total_loans",
        "interest_rate_derivatives",
    ]
    out = merged[keys].copy()
    for col in passthrough:
        out[col] = _coalesce(merged, [f"{col}_holding", f"{col}_call", col])
    for col in value_cols:
        holding_first = col not in {
            "total_deposits",
            "uninsured_deposits",
            "brokered_deposits",
            "noninterest_deposits",
        }
        choices = [f"{col}_holding", f"{col}_call"] if holding_first else [f"{col}_call", f"{col}_holding"]
        out[col] = _coalesce(merged, choices)
    return out.dropna(subset=["permco"]).drop_duplicates(keys)


def _build_stock_monthly(conn: Any, start: str, end: str) -> pd.DataFrame:
    factors = _pull_factors(conn, start, end)
    sql = f"""
        select m.permno, m.permco, m.mthcaldt as month, m.mthret as ret,
               m.mthprc as price, m.shrout as shares_out, m.mthvol as volume,
               m.mthcap as market_cap, m.siccd, m.ticker
        from crsp_a_stock.msf_v2 as m
        inner join (
            select distinct permco from bank_all.wrds_bank_crsp_link where permco is not null
        ) as b
          on m.permco = b.permco
        where m.mthcaldt between '{start}' and '{end}'
          and m.securitytype = 'EQTY'
          and m.sharetype = 'NS'
          and m.usincflg = 'Y'
    """
    stock = conn.raw_sql(sql)
    stock.columns = [c.lower() for c in stock.columns]
    stock["month"] = pd.to_datetime(stock["month"]).dt.to_period("M").dt.to_timestamp("M")
    stock = stock.merge(factors[["month", "rf", "mktrf"]], on="month", how="left")
    for c in ["ret", "price", "shares_out", "volume", "market_cap", "rf", "mktrf"]:
        if c in stock:
            stock[c] = pd.to_numeric(stock[c], errors="coerce")
    rf = stock["rf"].copy()
    if rf.abs().median(skipna=True) > 0.02:
        rf = rf / 100.0
    stock["excess_ret"] = stock["ret"] - rf.fillna(0)
    stock = stock.sort_values(["permno", "month"])
    market = stock.groupby("month", as_index=False)["excess_ret"].mean().rename(columns={"excess_ret": "bank_market_ret"})
    stock = stock.merge(market, on="month", how="left")
    stock["market_beta"] = (
        stock.groupby("permno", group_keys=False)
        .apply(_rolling_beta)
        .reset_index(level=0, drop=True)
    )
    return stock


def _rolling_beta(g: pd.DataFrame) -> pd.Series:
    cov = g["excess_ret"].rolling(24, min_periods=12).cov(g["bank_market_ret"])
    var = g["bank_market_ret"].rolling(24, min_periods=12).var()
    return (cov / var).replace([np.inf, -np.inf], np.nan)


def _pull_factors(conn: Any, start: str, end: str) -> pd.DataFrame:
    ff = conn.raw_sql(
        f"""
        select date as month, mktrf, smb, hml, rf, umd
        from ff_all.factors_monthly
        where date between '{start}' and '{end}'
        """
    )
    ff.columns = [c.lower() for c in ff.columns]
    ff["month"] = pd.to_datetime(ff["month"]).dt.to_period("M").dt.to_timestamp("M")
    return ff


def _build_macro(conn: Any, start: str, end: str) -> pd.DataFrame:
    yld = conn.raw_sql(
        f"""
        select qdate as date, yield1, yield2, yield5
        from crsp_a_treasuries.fbyld
        where qdate between '{start}' and '{end}'
        """
    )
    yld.columns = [c.lower() for c in yld.columns]
    yld["month"] = pd.to_datetime(yld["date"]).dt.to_period("M").dt.to_timestamp("M")
    if yld.empty or pd.to_datetime(yld["date"]).max() < pd.to_datetime(end) - pd.Timedelta(days=60):
        macro = _fred_macro(start, end)
    else:
        macro = yld.sort_values("date").groupby("month", as_index=False).tail(1)
        for c in ["yield1", "yield2", "yield5"]:
            macro[c] = pd.to_numeric(macro[c], errors="coerce")
        macro["rate_2y"] = macro["yield2"]
        macro["rate_2y_change"] = macro["rate_2y"].diff()
        macro["term_spread"] = macro["yield5"] - macro["yield1"]

    vix = conn.raw_sql(
        f"""
        select date, vix
        from cboe_all.cboe
        where date between '{start}' and '{end}'
        """
    )
    vix.columns = [c.lower() for c in vix.columns]
    vix["month"] = pd.to_datetime(vix["date"]).dt.to_period("M").dt.to_timestamp("M")
    vix = vix.sort_values("date").groupby("month", as_index=False).tail(1)
    macro = macro.merge(vix[["month", "vix"]], on="month", how="left")
    return macro[["month", "rate_2y_change", "term_spread", "vix"]]


def _fred_macro(start: str, end: str) -> pd.DataFrame:
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO,DGS2,DGS10"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    from io import StringIO

    rates = pd.read_csv(StringIO(resp.text))
    rates = rates.rename(columns={"observation_date": "date"})
    rates["date"] = pd.to_datetime(rates["date"])
    rates = rates[(rates["date"] >= start) & (rates["date"] <= end)]
    for col in ["DGS3MO", "DGS2", "DGS10"]:
        rates[col] = pd.to_numeric(rates[col].replace(".", np.nan), errors="coerce")
    rates["month"] = rates["date"].dt.to_period("M").dt.to_timestamp("M")
    macro = rates.sort_values("date").groupby("month", as_index=False).tail(1)
    macro["rate_2y_change"] = macro["DGS2"].diff()
    macro["term_spread"] = macro["DGS10"] - macro["DGS2"]
    return macro[["month", "rate_2y_change", "term_spread"]]


def _pull_parent_child_relationships(conn: Any, start: str, end: str) -> pd.DataFrame:
    rel = conn.raw_sql(
        f"""
        select id_rssd_parent, id_rssd_offspring, ctrl_ind, pct_equity, reln_lvl, date_start, date_end
        from bank_all.wrds_struct_relationships
        where id_rssd_parent in (
            select distinct rssd9001 from bank_all.wrds_bank_crsp_link where rssd9001 is not null
        )
          and id_rssd_offspring is not null
          and date_start <= '{end}'
          and coalesce(date_end, date '9999-12-31') >= '{start}'
          and (ctrl_ind = 1 or pct_equity >= 50)
        """
    )
    rel.columns = [c.lower() for c in rel.columns]
    return rel


def extract_live_raw(
    out_dir: str | Path,
    start: str = "2004-01-01",
    end: str = "2026-03-31",
) -> dict[str, Path]:
    start = _sql_date(start)
    end = _sql_date(end)
    out_dir = Path(out_dir)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    conn = _connect_wrds()
    try:
        link = conn.raw_sql(
            """
            select rssd9001, permco, name, inst_type, dt_start, dt_end
            from bank_all.wrds_bank_crsp_link
            where permco is not null
              and rssd9001 is not null
            """
        )
        link.columns = [c.lower() for c in link.columns]
        link["dt_end"] = pd.to_datetime(link["dt_end"], errors="coerce").fillna(pd.Timestamp("2099-12-31"))
        rel = _pull_parent_child_relationships(conn, start, end)
        offspring_filter = _offspring_filter_sql(start, end)

        call_components = [
            _pull_table(
                conn,
                "bank_all",
                "wrds_call_rcon_2",
                "wrdsreportdate",
                start,
                end,
                [
                    "rssd9001",
                    "wrdsreportdate",
                    "rssdsubmissiondate",
                    "rcon2170",
                    "rcon2200",
                    "rcon2365",
                    "rconj474",
                    "rcon6631",
                    "rcon1771",
                    "rcon1772",
                    "rcon2122",
                    "rcon3210",
                    "rcon8274",
                    "rcon8725",
                    "rcona126",
                ],
                rssd_filter_sql=offspring_filter,
            ),
            _pull_table(
                conn,
                "bank_all",
                "wrds_call_rcon_1",
                "wrdsreportdate",
                start,
                end,
                ["rssd9001", "wrdsreportdate", "rconl197", "rcon1754", "rconjj05", "rconhk12", "rconhk13", "rconhk14", "rconhk15"],
                rssd_filter_sql=offspring_filter,
            ),
            _pull_table(
                conn,
                "bank_all",
                "wrds_call_rcfd_2",
                "wrdsreportdate",
                start,
                end,
                [
                    "rssd9001",
                    "wrdsreportdate",
                    "rcfd2170",
                    "rcfd1771",
                    "rcfd1772",
                    "rcfd2122",
                    "rcfd3210",
                    "rcfd8274",
                    "rcfd8725",
                    "rcfda126",
                ],
                rssd_filter_sql=offspring_filter,
            ),
            _pull_table(
                conn,
                "bank_all",
                "wrds_call_rcfd_1",
                "wrdsreportdate",
                start,
                end,
                ["rssd9001", "wrdsreportdate", "rcfd1754", "rcfdjj05"],
                rssd_filter_sql=offspring_filter,
            ),
            _pull_table(
                conn,
                "bank_all",
                "wrds_call_rcoa_1",
                "wrdsreportdate",
                start,
                end,
                ["rssd9001", "wrdsreportdate", "rcoa2170", "rcoa8274", "rcoap859"],
                rssd_filter_sql=offspring_filter,
            ),
            _pull_table(
                conn,
                "bank_all",
                "wrds_call_rcfa_1",
                "wrdsreportdate",
                start,
                end,
                ["rssd9001", "wrdsreportdate", "rcfa2170", "rcfa8274", "rcfap859"],
                rssd_filter_sql=offspring_filter,
            ),
            _pull_table(
                conn,
                "bank_all",
                "wrds_call_rcfw_1",
                "wrdsreportdate",
                start,
                end,
                ["rssd9001", "wrdsreportdate", "rcfwp859"],
                rssd_filter_sql=offspring_filter,
            ),
        ]
        holding_components = [
            _pull_holding_table(
                conn,
                "wrds_holding_bhck_1",
                start,
                end,
                ["bhck1754", "bhckjj05"],
            ),
            _pull_holding_table(
                conn,
                "wrds_holding_bhck_2",
                start,
                end,
                ["bhck1771", "bhck1772", "bhck2122", "bhck2170", "bhck3210", "bhck8725", "bhcka126"],
            ),
            _pull_holding_table(
                conn,
                "wrds_holding_other_1",
                start,
                end,
                ["bhca2170", "bhct2170", "bhcap859", "bhcwp859", "bhdm2122", "bhdm6631", "bhfn6631", "bhcbj474", "bhodj474"],
            ),
        ]
        call_raw = _merge_components(call_components)
        call_quarterly = _build_bank_quarterly(call_raw, link, rel=rel)
        holding_raw = _merge_components(holding_components)
        holding_quarterly = _build_bank_quarterly(holding_raw, link, rel=None)
        bank_quarterly = _combine_holding_and_call(holding_quarterly, call_quarterly)
        stock = _build_stock_monthly(conn, start, end)
        macro = _build_macro(conn, start, end)
        factors = _pull_factors(conn, start, end)

        paths = {
            "bank_link": raw_dir / "bank_link.parquet",
            "relationships": raw_dir / "bank_relationships.parquet",
            "call_quarterly": raw_dir / "call_quarterly_parent_agg.parquet",
            "holding_quarterly": raw_dir / "holding_quarterly.parquet",
            "bank_quarterly": raw_dir / "bank_quarterly.parquet",
            "stock_monthly": raw_dir / "stock_monthly.parquet",
            "macro_monthly": raw_dir / "macro_monthly.parquet",
            "factors_monthly": raw_dir / "factors_monthly.parquet",
        }
        link.to_parquet(paths["bank_link"], index=False)
        rel.to_parquet(paths["relationships"], index=False)
        call_quarterly.to_parquet(paths["call_quarterly"], index=False)
        holding_quarterly.to_parquet(paths["holding_quarterly"], index=False)
        bank_quarterly.to_parquet(paths["bank_quarterly"], index=False)
        stock.to_parquet(paths["stock_monthly"], index=False)
        macro.to_parquet(paths["macro_monthly"], index=False)
        factors.to_parquet(paths["factors_monthly"], index=False)
        write_manifest(
            out_dir / "live_extract_manifest.json",
            {
                "kind": "live_extract",
                "start": start,
                "end": end,
                "paths": {k: str(v) for k, v in paths.items()},
                "rows": {
                    "bank_link": len(link),
                    "relationships": len(rel),
                    "call_quarterly": len(call_quarterly),
                    "holding_quarterly": len(holding_quarterly),
                    "bank_quarterly": len(bank_quarterly),
                    "stock_monthly": len(stock),
                    "macro_monthly": len(macro),
                    "factors_monthly": len(factors),
                },
                "wrds_tables": [
                    "bank_all.wrds_bank_crsp_link",
                    "bank_all.wrds_call_rcon_1",
                    "bank_all.wrds_call_rcon_2",
                    "bank_all.wrds_call_rcfd_1",
                    "bank_all.wrds_call_rcfd_2",
                    "bank_all.wrds_call_rcoa_1",
                    "bank_all.wrds_call_rcfa_1",
                    "bank_all.wrds_call_rcfw_1",
                    "crsp_a_stock.msf_v2",
                    "crsp_a_treasuries.fbyld",
                    "cboe_all.cboe",
                    "ff_all.factors_monthly",
                ],
            },
        )
        return paths
    finally:
        conn.close()
