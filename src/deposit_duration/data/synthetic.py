from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from deposit_duration.utils.manifest import write_manifest


def make_synthetic_raw(out_dir: str | Path, seed: int = 1378) -> dict[str, Path]:
    out_dir = Path(out_dir)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    n_banks = 80
    banks = pd.DataFrame(
        {
            "bank_id": np.arange(1, n_banks + 1),
            "permno": np.arange(10001, 10001 + n_banks),
            "gvkey": [f"{i:06d}" for i in range(1, n_banks + 1)],
            "name": [f"Synthetic Bank {i:03d}" for i in range(1, n_banks + 1)],
        }
    )

    quarters = pd.period_range("2012Q1", "2025Q4", freq="Q").to_timestamp("Q")
    months = pd.period_range("2012-04", "2026-03", freq="M").to_timestamp("M")

    q_rows = []
    for _, bank in banks.iterrows():
        bank_size = rng.lognormal(mean=8.5, sigma=0.7)
        franchise = rng.normal(0, 1)
        duration_tilt = rng.normal(0, 1)
        for qdate in quarters:
            asset_growth = 1 + rng.normal(0.008, 0.025)
            bank_size *= max(asset_growth, 0.85)
            deposits = bank_size * rng.uniform(0.55, 0.9)
            uninsured = deposits * np.clip(0.25 + 0.12 * franchise + rng.normal(0, 0.08), 0.02, 0.85)
            securities = bank_size * np.clip(0.18 + 0.08 * duration_tilt + rng.normal(0, 0.05), 0.02, 0.65)
            htm = securities * np.clip(0.45 + 0.15 * duration_tilt + rng.normal(0, 0.08), 0.0, 0.95)
            afs = securities - htm
            capital = bank_size * np.clip(0.09 + rng.normal(0, 0.015), 0.04, 0.18)
            cre = bank_size * np.clip(0.18 + rng.normal(0, 0.08), 0.02, 0.6)
            q_rows.append(
                {
                    "bank_id": bank.bank_id,
                    "permno": bank.permno,
                    "gvkey": bank.gvkey,
                    "report_date": qdate,
                    "total_assets": bank_size,
                    "total_deposits": deposits,
                    "uninsured_deposits": uninsured,
                    "brokered_deposits": deposits * np.clip(rng.beta(1.5, 12), 0, 0.5),
                    "large_time_deposits": deposits * np.clip(rng.beta(2.0, 10), 0, 0.6),
                    "noninterest_deposits": deposits
                    * np.clip(0.28 - 0.08 * franchise + rng.normal(0, 0.06), 0.02, 0.8),
                    "securities_afs": afs,
                    "securities_htm": htm,
                    "unrealized_losses": -securities
                    * np.clip(0.01 + 0.012 * duration_tilt + rng.normal(0, 0.01), -0.02, 0.12),
                    "tier1_capital": capital,
                    "cre_loans": cre,
                    "total_loans": bank_size * np.clip(0.58 + rng.normal(0, 0.08), 0.15, 0.9),
                    "interest_rate_derivatives": bank_size
                    * np.clip(0.03 - 0.02 * duration_tilt + rng.normal(0, 0.025), 0, 0.35),
                }
            )
    quarterly = pd.DataFrame(q_rows)

    m_rows = []
    macro = pd.DataFrame({"month": months})
    macro["rate_2y_change"] = rng.normal(0, 0.22, len(macro))
    macro["term_spread"] = rng.normal(0.8, 0.9, len(macro))
    macro["vix"] = np.clip(rng.normal(20, 7, len(macro)), 10, 65)
    macro.loc[(macro["month"] >= "2023-03-31") & (macro["month"] <= "2023-05-31"), "vix"] += 18
    macro.loc[(macro["month"] >= "2022-01-31") & (macro["month"] <= "2023-10-31"), "rate_2y_change"] += 0.25
    factors = pd.DataFrame({"month": months})
    factors["mktrf"] = rng.normal(0.006, 0.045, len(factors))
    factors["smb"] = rng.normal(0.001, 0.025, len(factors))
    factors["hml"] = rng.normal(0.001, 0.030, len(factors))
    factors["rmw"] = rng.normal(0.001, 0.020, len(factors))
    factors["cma"] = rng.normal(0.001, 0.020, len(factors))
    factors["umd"] = rng.normal(0.002, 0.035, len(factors))

    for _, bank in banks.iterrows():
        price = rng.uniform(12, 90)
        shares = rng.lognormal(mean=4.2, sigma=0.7)
        bank_beta = rng.normal(1.0, 0.25)
        for month in months:
            q = quarterly[(quarterly.bank_id == bank.bank_id) & (quarterly.report_date <= month)].tail(1)
            if q.empty:
                continue
            qrow = q.iloc[0]
            mac = macro.loc[macro.month == month].iloc[0]
            frag = qrow.uninsured_deposits / qrow.total_deposits
            dur = (qrow.securities_htm + 0.5 * qrow.securities_afs) / qrow.total_assets
            shock_loss = -0.015 * frag * dur * max(mac.rate_2y_change, 0) * 10
            alpha = 0.002 + 0.006 * dur * frag - 0.003 * qrow.cre_loans / qrow.total_assets
            ret = alpha + shock_loss + rng.normal(0, 0.075) + 0.01 * bank_beta
            price = max(price * (1 + ret), 1.0)
            m_rows.append(
                {
                    "permno": bank.permno,
                    "month": month,
                    "ret": ret,
                    "excess_ret": ret - 0.002,
                    "price": price,
                    "shares_out": shares,
                    "volume": rng.lognormal(12.0, 0.8),
                    "market_beta": bank_beta,
                }
            )
    monthly = pd.DataFrame(m_rows)

    paths = {
        "banks": raw_dir / "banks.parquet",
        "quarterly": raw_dir / "bank_quarterly.parquet",
        "monthly": raw_dir / "stock_monthly.parquet",
        "macro": raw_dir / "macro_monthly.parquet",
        "factors": raw_dir / "factors_monthly.parquet",
    }
    banks.to_parquet(paths["banks"], index=False)
    quarterly.to_parquet(paths["quarterly"], index=False)
    monthly.to_parquet(paths["monthly"], index=False)
    macro.to_parquet(paths["macro"], index=False)
    factors.to_parquet(paths["factors"], index=False)
    write_manifest(out_dir / "manifest.json", {"kind": "synthetic_raw", "paths": {k: str(v) for k, v in paths.items()}})
    return paths
