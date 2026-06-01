-- Treasury state variables from CRSP Treasuries.
-- Parameters: :start_date, :end_date.

select qdate as date, yield1, yield2, yield5
from crsp_a_treasuries.fbyld
where qdate between :start_date and :end_date;
