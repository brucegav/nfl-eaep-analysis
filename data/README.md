# Data

## Required Downloads

The following files are **not included** in this repository due to size and licensing. 
Download them and place them in the `data/` directory before running the notebook.

### FiveThirtyEight NFL ELO Dataset
- **File:** `nfl_elo.csv`
- **Source:** https://github.com/fivethirtyeight/data/tree/master/nfl-elo
- **Description:** Game-by-game ELO ratings and forecasts for every NFL game back 
  to 1920. Primary dataset for all ELO and EAEP computations.

- **File:** `nfl_games.csv`  
- **Source:** https://github.com/fivethirtyeight/nfl-elo-game
- **Description:** Historical NFL game scores back to 1920. Used for computing 
  ELO ratings for seasons 2022-2025 beyond the dataset's cutoff.

---

## Included Data

All files below are manually curated from 
[Pro Football Reference](https://www.pro-football-reference.com) 
and are included in the repository.

### `coaching_tenures/`
Head coaching tenure data for all 32 current NFL franchises. One CSV per franchise.
Includes coaches with 20+ games coached to filter out interim coaches.

### `draft/`
NFL Draft pick data from 1960-1979 collected from Pro Football Reference.
Draft data from 1980-2025 is loaded programmatically via `nflreadpy`.

### `quarterbacks/`
Franchise quarterback history for selected franchises used in QB analysis and modeling.
Includes `franchise_qb_history.csv` used for the homegrown vs. acquired QB analysis.

### `nfl_standings_2021_2025.csv`
NFL franchise win/loss records from 2022-2025 seasons. Used for real-world 
validation of the sustained decline prediction model.

### `playoff_seasons.csv`
Playoff appearance seasons for the Buffalo Bills, Tennessee Titans, and 
Washington Commanders. Used in the peak era analysis.