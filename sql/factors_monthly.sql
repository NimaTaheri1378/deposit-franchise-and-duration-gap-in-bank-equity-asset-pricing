-- Fama-French monthly factors from WRDS.
-- Parameters: :start_date, :end_date.

select date as month, mktrf, smb, hml, rf, umd
from ff_all.factors_monthly
where date between :start_date and :end_date;
