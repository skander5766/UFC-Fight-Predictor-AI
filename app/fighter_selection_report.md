# Fighter selection audit

Application-only membership and status filtering; no historical training records are removed.

Divisions come from fights_dated.csv and retain observation dates. Queries with as_of_date use strictly earlier observations and exclude undated rows. Historical division names do not guarantee identical weight limits across eras. Generic tournament and Superfight labels remain unspecified; no division is inferred from body weight.

No reliable retirement/current-roster field was found in the local CSV files below. stats_match_status and date_match_status describe joins, not fighter status. Unknown status remains None. A future separate evidence adapter requires an explicit status, verification flag, source reference, effective date and verification date. Unknowns are included by default; this is not a verified active roster.

## Unique fighter membership counts (all observed dates, including undated)

| Division | Fighters |
|---|---:|
| Bantamweight | 348 |
| Catch Weight | 151 |
| Featherweight | 407 |
| Flyweight | 165 |
| Heavyweight | 327 |
| Light Heavyweight | 309 |
| Lightweight | 588 |
| Middleweight | 472 |
| Open Weight | 98 |
| Super Heavyweight | 2 |
| Unspecified historical division | 15 |
| Welterweight | 571 |
| Women's Bantamweight | 102 |
| Women's Featherweight | 26 |
| Women's Flyweight | 118 |
| Women's Strawweight | 129 |

## Multi-division examples

- Conor McGregor: Featherweight, Lightweight, Welterweight
- Georges St-Pierre: Middleweight, Welterweight
- Amanda Nunes: Women's Bantamweight, Women's Featherweight

## Inspected local CSV files

- `data/processed/fights_clean.csv`: 8,936 rows; status-like columns: ['stats_match_status']; explicit status-label hits: [].
- `data/processed/fights_dated.csv`: 8,936 rows; status-like columns: ['stats_match_status', 'date_match_status']; explicit status-label hits: [].
- `data/processed/prefight_features.csv`: 8,726 rows; status-like columns: []; explicit status-label hits: [].
- `data/processed/prefight_features_activity.csv`: 8,726 rows; status-like columns: []; explicit status-label hits: [].
- `data/processed/prefight_features_enriched.csv`: 8,726 rows; status-like columns: []; explicit status-label hits: [].
- `data/raw/fighters.csv`: 4,494 rows; status-like columns: []; explicit status-label hits: [].
- `data/raw/fights.csv`: 8,701 rows; status-like columns: []; explicit status-label hits: [].
- `data/raw/ufc_fight_results.csv`: 8,936 rows; status-like columns: []; explicit status-label hits: [].
- `data/raw/ufc_fight_stats.csv`: 42,012 rows; status-like columns: []; explicit status-label hits: [].

## Complete raw results labels and counts

Counts are source rows, not summed across duplicated processed snapshots. Normalization changes only the application catalog.

| Raw label | Rows | Application division |
|---|---:|---|
| Lightweight Bout | 1434 | Lightweight |
| Welterweight Bout | 1351 | Welterweight |
| Middleweight Bout | 1120 | Middleweight |
| Featherweight Bout | 859 | Featherweight |
| Bantamweight Bout | 773 | Bantamweight |
| Heavyweight Bout | 728 | Heavyweight |
| Light Heavyweight Bout | 712 | Light Heavyweight |
| Flyweight Bout | 403 | Flyweight |
| Women's Strawweight Bout | 362 | Women's Strawweight |
| Women's Flyweight Bout | 268 | Women's Flyweight |
| Women's Bantamweight Bout | 232 | Women's Bantamweight |
| Open Weight Bout | 101 | Open Weight |
| Catch Weight Bout | 84 | Catch Weight |
| UFC Light Heavyweight Title Bout | 57 | Light Heavyweight |
| UFC Welterweight Title Bout | 50 | Welterweight |
| UFC Heavyweight Title Bout | 44 | Heavyweight |
| UFC Middleweight Title Bout | 40 | Middleweight |
| UFC Lightweight Title Bout | 37 | Lightweight |
| UFC Flyweight Title Bout | 28 | Flyweight |
| UFC Bantamweight Title Bout | 25 | Bantamweight |
| UFC Featherweight Title Bout | 22 | Featherweight |
| Women's Featherweight Bout | 22 | Women's Featherweight |
| UFC Women's Strawweight Title Bout | 21 | Women's Strawweight |
| UFC Women's Bantamweight Title Bout | 20 | Women's Bantamweight |
| UFC Women's Flyweight Title Bout | 15 | Women's Flyweight |
| UFC Interim Heavyweight Title Bout | 10 | Heavyweight |
| UFC Women's Featherweight Title Bout | 7 | Women's Featherweight |
| UFC Superfight Championship Bout | 5 | Unspecified historical division |
| UFC Interim Lightweight Title Bout | 4 | Lightweight |
| UFC Interim Featherweight Title Bout | 4 | Featherweight |
| UFC Interim Bantamweight Title Bout | 4 | Bantamweight |
| UFC Interim Welterweight Title Bout | 3 | Welterweight |
| Ultimate Fighter 33 Welterweight Tournament Title Bout | 2 | Welterweight |
| UFC Interim Middleweight Title Bout | 2 | Middleweight |
| UFC Interim Light Heavyweight Title Bout | 2 | Light Heavyweight |
| Ultimate Fighter 33 Flyweight Tournament Title Bout | 1 | Flyweight |
| Road to UFC 3 Bantamweight Tournament TitleBout | 1 | Bantamweight |
| Road to UFC 3 Flyweight Tournament TitleBout | 1 | Flyweight |
| Road to UFC 3 Women's Strawweight Tournament TitleBout | 1 | Women's Strawweight |
| Ultimate Fighter 32 Middleweight Tournament Title Bout | 1 | Middleweight |
| Ultimate Fighter 32 Featherweight Tournament Title Bout | 1 | Featherweight |
| Road To UFC 1 Lightweight Tournament Title Bout | 1 | Lightweight |
| Road To UFC 1 Featherweight Tournament Title Bout | 1 | Featherweight |
| Road To UFC 1 Bantamweight Tournament Title Bout | 1 | Bantamweight |
| Road to UFC 1 Flyweight Tournament Title Bout | 1 | Flyweight |
| UFC Interim Flyweight Title Bout | 1 | Flyweight |
| Ultimate Fighter 28 Heavyweight Tournament Title Bout | 1 | Heavyweight |
| Ultimate Fighter 28 Women's Featherweight Tournament Title Bout | 1 | Women's Featherweight |
| Ultimate Fighter 27 Lightweight Tournament Title Bout | 1 | Lightweight |
| Ultimate Fighter 27 Featherweight Tournament Title Bout | 1 | Featherweight |
| Ultimate Fighter 25 Welterweight Tournament Title Bout | 1 | Welterweight |
| Ultimate Fighter Latin America 3 Lightweight Tournament Title Bout | 1 | Lightweight |
| Ultimate Fighter 23 Light Heavyweight Tournament Title Bout | 1 | Light Heavyweight |
| Ultimate Fighter 23 Women's Strawweight Tournament Title Bout | 1 | Women's Strawweight |
| Ultimate Fighter 22 Lightweight Tournament Title Bout | 1 | Lightweight |
| Ultimate Fighter Latin America 2 Welterweight Tournament Title Bout | 1 | Welterweight |
| Ultimate Fighter Latin America 2 Lightweight Tournament Title Bout | 1 | Lightweight |
| Ultimate Fighter Brazil 4 Lightweight Tournament Title Bout | 1 | Lightweight |
| Ultimate Fighter Brazil 4 Bantamweight Tournament Title Bout | 1 | Bantamweight |
| Ultimate Fighter 21 Welterweight Tournament Title Bout | 1 | Welterweight |
| Ultimate Fighter Latin America Featherweight Tournament Title Bout | 1 | Featherweight |
| Ultimate Fighter Latin America Bantamweight Tournament Title Bout | 1 | Bantamweight |
| Ultimate Fighter China Featherweight Tournament Title Bout | 1 | Featherweight |
| Ultimate Fighter 19 Light Heavyweight Tournament Title Bout | 1 | Light Heavyweight |
| Ultimate Fighter 19 Middleweight Tournament Title Bout | 1 | Middleweight |
| Ultimate Fighter Brazil 3 Heavyweight Tournament Title Bout | 1 | Heavyweight |
| Ultimate Fighter Brazil 3 Middleweight Tournament Title Bout | 1 | Middleweight |
| TUF Nations Canada vs. Australia Middleweight Tournament Title Bout | 1 | Middleweight |
| TUF Nations Canada vs. Australia Welterweight Tournament Title Bout | 1 | Welterweight |
| Ultimate Fighter China Welterweight Tournament Title Bout | 1 | Welterweight |
| Ultimate Fighter 18 Women's Bantamweight Tournament Title Bout | 1 | Women's Bantamweight |
| Ultimate Fighter 18 Bantamweight Tournament Title Bout | 1 | Bantamweight |
| Ultimate Fighter Brazil 2 Welterweight Tournament Title Bout | 1 | Welterweight |
| Ultimate Fighter 17 Middleweight Tournament Title Bout | 1 | Middleweight |
| Ultimate Fighter 16 Welterweight Tournament Title Bout | 1 | Welterweight |
| Ultimate Fighter Australia vs. UK Welterweight Tournament Title Bout | 1 | Welterweight |
| Ultimate Fighter Australia vs. UK Lightweight Tournament Title Bout | 1 | Lightweight |
| Ultimate Fighter Brazil 1 Middleweight Tournament Title Bout | 1 | Middleweight |
| Ultimate Fighter Brazil 1 Featherweight Tournament Title Bout | 1 | Featherweight |
| Ultimate Fighter 15 Lightweight Tournament Title Bout | 1 | Lightweight |
| Ultimate Fighter 14 Featherweight Tournament Title Bout | 1 | Featherweight |
| Ultimate Fighter 14 Bantamweight Tournament Title Bout | 1 | Bantamweight |
| Ultimate Fighter 13 Welterweight Tournament Title Bout | 1 | Welterweight |
| Ultimate Fighter 12 Lightweight Tournament Title Bout | 1 | Lightweight |
| Ultimate Fighter 11 Middleweight Tournament Title Bout | 1 | Middleweight |
| Ultimate Fighter 10 Heavyweight Tournament Title Bout | 1 | Heavyweight |
| Ultimate Fighter 9 Welterweight Tournament Title Bout | 1 | Welterweight |
| Ultimate Fighter 9 Lightweight Tournament Title Bout | 1 | Lightweight |
| Ultimate Fighter 8 Lightweight Tournament Title Bout | 1 | Lightweight |
| Ultimate Fighter 8 Light Heavyweight Tournament Title Bout | 1 | Light Heavyweight |
| Ultimate Fighter 7 Middleweight Tournament Title Bout | 1 | Middleweight |
| Ultimate Fighter 6 Welterweight Tournament Title Bout | 1 | Welterweight |
| Ultimate Fighter 5 Lightweight Tournament Title Bout | 1 | Lightweight |
| Ultimate Fighter 4 Welterweight Tournament Title Bout | 1 | Welterweight |
| Ultimate Fighter 4 Middleweight Tournament Title Bout | 1 | Middleweight |
| Ultimate Fighter 3 Light Heavyweight Tournament Title Bout | 1 | Light Heavyweight |
| Ultimate Fighter 3 Middleweight Tournament Title Bout | 1 | Middleweight |
| Ultimate Fighter 2 Heavyweight Tournament Title Bout | 1 | Heavyweight |
| Ultimate Fighter 2 Welterweight Tournament Title Bout | 1 | Welterweight |
| Ultimate Fighter 1 Light Heavyweight Tournament Title Bout | 1 | Light Heavyweight |
| Ultimate Fighter 1 Middleweight Tournament Title Bout | 1 | Middleweight |
| Super Heavyweight Bout | 1 | Super Heavyweight |
| Ultimate Japan 2 Heavyweight Tournament Title Bout | 1 | Heavyweight |
| UFC 17 Middleweight Tournament Title Bout | 1 | Middleweight |
| Ultimate Japan Heavyweight Tournament Title Bout | 1 | Heavyweight |
| UFC 15 Heavyweight Tournament Title Bout | 1 | Heavyweight |
| UFC 14 Heavyweight Tournament Title Bout | 1 | Heavyweight |
| UFC 14 Middleweight Tournament Title Bout | 1 | Middleweight |
| UFC 13 Heavyweight Tournament Title Bout | 1 | Heavyweight |
| UFC 13 Lightweight Tournament Title Bout | 1 | Lightweight |
| Ultimate Ultimate '96 Tournament Title Bout | 1 | Unspecified historical division |
| UFC 10 Tournament Title Bout | 1 | Unspecified historical division |
| UFC 8 Tournament Title Bout | 1 | Unspecified historical division |
| Ultimate Ultimate '95 Tournament Title Bout | 1 | Unspecified historical division |
| UFC 7 Tournament Title Bout | 1 | Unspecified historical division |
| UFC 6 Tournament Title Bout | 1 | Unspecified historical division |
| UFC 5 Tournament Title Bout | 1 | Unspecified historical division |
| UFC 4 Tournament Title Bout | 1 | Unspecified historical division |
| UFC 3 Tournament Title Bout | 1 | Unspecified historical division |
| UFC 2 Tournament Title Bout | 1 | Unspecified historical division |

## Independent fights.csv labels

| Raw label | Rows |
|---|---:|
| Lightweight | 1460 |
| Welterweight | 1393 |
| Middleweight | 1153 |
| Featherweight | 861 |
| Bantamweight | 781 |
| Heavyweight | 772 |
| Light Heavyweight | 750 |
| Flyweight | 415 |
| Women's Strawweight | 368 |
| Women's Flyweight | 273 |
| Women's Bantamweight | 246 |
| Open Weight | 117 |
| Catch Weight | 81 |
| Women's Featherweight | 30 |
| Super Heavyweight | 1 |
