-- CRSP monthly listed-bank stock panel.
-- Parameters: :start_date, :end_date.

select m.permno, m.permco, m.mthcaldt as month, m.mthret as ret,
       m.mthprc as price, m.shrout as shares_out, m.mthvol as volume,
       m.mthcap as market_cap, m.siccd, m.ticker
from crsp_a_stock.msf_v2 as m
inner join (
    select distinct permco
    from bank_all.wrds_bank_crsp_link
    where permco is not null
) as b
  on m.permco = b.permco
where m.mthcaldt between :start_date and :end_date
  and m.securitytype = 'EQTY'
  and m.sharetype = 'NS'
  and m.usincflg = 'Y';
