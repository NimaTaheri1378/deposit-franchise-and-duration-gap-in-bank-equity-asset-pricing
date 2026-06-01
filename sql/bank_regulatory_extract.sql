-- Bank regulatory components for listed-bank parent aggregation.
-- Parameters: :start_date, :end_date.

select rssd9001, wrdsreportdate, rssdsubmissiondate,
       rcon2170, rcon2200, rcon2365, rconj474, rcon6631,
       rcon1771, rcon1772, rcon2122, rcon3210, rcon8274,
       rcon8725, rcona126
from bank_all.wrds_call_rcon_2
where wrdsreportdate between :start_date and :end_date
  and rssd9001 in (
      select distinct id_rssd_offspring
      from bank_all.wrds_struct_relationships
      where id_rssd_parent in (
          select distinct rssd9001
          from bank_all.wrds_bank_crsp_link
          where rssd9001 is not null
      )
        and id_rssd_offspring is not null
        and date_start <= :end_date
        and coalesce(date_end, date '9999-12-31') >= :start_date
        and (ctrl_ind = 1 or pct_equity >= 50)
  );
