from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deposit_duration.utils.config import project_config, write_yaml
from deposit_duration.utils.manifest import write_manifest


SAFE_NAME = re.compile(r"^[a-zA-Z0-9_]+$")


ROLE_KEYWORDS: dict[str, list[str]] = {
    "bank_call_reports": [
        "call",
        "ffiec",
        "rc",
        "rcon",
        "rcfd",
        "rssd",
        "cert",
        "depos",
        "asset",
        "loan",
    ],
    "holding_company_financials": ["y9", "bhc", "holding", "parent", "rssd", "bhck"],
    "bank_structure": ["structure", "hierarchy", "entity", "parent", "relationship", "rssd"],
    "bank_linking": ["link", "crosswalk", "permno", "gvkey", "cusip", "rssd", "cert"],
    "crsp_monthly_stock": ["msf", "monthly", "ret", "permno", "shrout", "prc", "dlret"],
    "crsp_daily_stock": ["dsf", "daily", "ret", "permno", "vol", "prc"],
    "crsp_names": ["msenames", "names", "ncusip", "ticker", "siccd"],
    "ccm_link": ["ccm", "link", "lnk", "gvkey", "permno", "linkdt", "linkenddt"],
    "treasury": ["treasury", "yield", "tfz", "maturity", "bond", "rate"],
    "fama_french": ["ff", "factor", "mktrf", "smb", "hml", "mom"],
    "macro": ["fred", "frb", "series", "macro", "rate"],
    "options": ["option", "opprcd", "impl", "volatility", "delta"],
}


@dataclass(frozen=True)
class TableProfile:
    schema: str
    table: str
    columns: list[str]

    @property
    def key(self) -> str:
        return f"{self.schema}.{self.table}"


def _safe_schema_list(schemas: list[str]) -> list[str]:
    safe = []
    for schema in schemas:
        if not SAFE_NAME.match(schema):
            raise ValueError(f"Unsafe schema name in config: {schema!r}")
        safe.append(schema)
    return safe


def _sql_in(values: list[str]) -> str:
    return ", ".join(f"'{v}'" for v in values)


def _score(profile: TableProfile, keywords: list[str]) -> int:
    haystack = " ".join([profile.schema, profile.table, *profile.columns]).lower()
    return sum(haystack.count(k.lower()) for k in keywords)


def _connect_wrds() -> Any:
    try:
        import wrds  # type: ignore
    except ImportError as exc:
        raise RuntimeError("The wrds package is required for schema-audit. Install .[wrds].") from exc
    return wrds.Connection()


def run_schema_audit(
    config_path: str | Path = "configs/project.yml",
    out_yaml: str | Path = "configs/schema_map.yml",
    manifest_path: str | Path = "data_manifest/schema_audit.json",
    max_tables_per_library: int = 500,
    max_columns_per_table: int = 80,
) -> dict[str, Any]:
    cfg = project_config(config_path)
    schemas = _safe_schema_list(cfg["wrds"]["candidate_libraries"])
    db = _connect_wrds()
    try:
        visible = set(db.list_libraries())
        target_schemas = [s for s in schemas if s in visible]

        tables_sql = f"""
            select table_schema, table_name
            from information_schema.tables
            where table_schema in ({_sql_in(target_schemas)})
              and table_type = 'BASE TABLE'
            order by table_schema, table_name
        """
        tables = db.raw_sql(tables_sql)
        table_rows = tables.to_dict("records")

        selected_table_rows = []
        per_schema_count: dict[str, int] = defaultdict(int)
        for row in table_rows:
            schema = row["table_schema"]
            if per_schema_count[schema] >= max_tables_per_library:
                continue
            selected_table_rows.append(row)
            per_schema_count[schema] += 1

        selected_names = [(r["table_schema"], r["table_name"]) for r in selected_table_rows]
        profiles: list[TableProfile] = []
        if selected_names:
            where = " or ".join(
                f"(table_schema = '{schema}' and table_name = '{table}')"
                for schema, table in selected_names
            )
            cols_sql = f"""
                select table_schema, table_name, column_name, ordinal_position
                from information_schema.columns
                where {where}
                order by table_schema, table_name, ordinal_position
            """
            cols = db.raw_sql(cols_sql)
            grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
            for row in cols.to_dict("records"):
                key = (row["table_schema"], row["table_name"])
                if len(grouped[key]) < max_columns_per_table:
                    grouped[key].append(row["column_name"])
            profiles = [
                TableProfile(schema=schema, table=table, columns=grouped.get((schema, table), []))
                for schema, table in selected_names
            ]

        role_candidates: dict[str, list[dict[str, Any]]] = {}
        for role, keywords in ROLE_KEYWORDS.items():
            scored = []
            for profile in profiles:
                score = _score(profile, keywords)
                if score > 0:
                    scored.append(
                        {
                            "schema": profile.schema,
                            "table": profile.table,
                            "score": score,
                            "columns": profile.columns[:30],
                        }
                    )
            role_candidates[role] = sorted(scored, key=lambda x: x["score"], reverse=True)[:25]

        result: dict[str, Any] = {
            "status": "audited",
            "libraries": {
                schema: {
                    "available": schema in visible,
                    "table_count_seen": int((tables["table_schema"] == schema).sum())
                    if target_schemas
                    else 0,
                }
                for schema in schemas
            },
            "role_candidates": role_candidates,
            "selected_tables": {
                role: candidates[0] if candidates else None
                for role, candidates in role_candidates.items()
            },
            "notes": [
                "Selected tables are heuristic first choices. Inspect before full extraction.",
                "No table data were pulled by this audit; only metadata were queried.",
            ],
        }
        write_yaml(out_yaml, result)
        write_manifest(
            manifest_path,
            {
                "kind": "schema_audit",
                "visible_library_count": len(visible),
                "target_libraries": schemas,
                "available_target_libraries": target_schemas,
                "profiled_table_count": len(profiles),
                "schema_map": str(out_yaml),
            },
        )
        return result
    finally:
        try:
            db.close()
        except Exception:
            pass
