# SQL Extract Templates

These files document the WRDS queries used by the live extraction code. They
are templates, not standalone credentials or raw outputs. The Python extractor
still performs schema-aware column filtering before execution because WRDS table
coverage differs across subscriptions.

The public repository should keep these templates, while raw WRDS outputs stay
under ignored `artifacts/`, `data/`, or cluster scratch folders.
