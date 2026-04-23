import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import playergamelog

PLAYER_ID = 2544  # LeBron James
SEASON = "2024-25"
OUTPUT_PATH = Path(__file__).parents[3] / "data" / "raw" / "lebron_smoke.parquet"


def fetch_gamelog(player_id: int, season: str) -> pd.DataFrame:
    response = playergamelog.PlayerGameLog(
        player_id=player_id,
        season=season,
        season_type_all_star="Regular Season",
    )
    time.sleep(0.6)  # nba_api rate-limit pattern — keep even for single calls
    return response.get_data_frames()[0]


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = fetch_gamelog(PLAYER_ID, SEASON)
    df.to_parquet(OUTPUT_PATH, index=False)

    null_counts = df.isnull().sum()
    print(f"Shape:       {df.shape}")
    print(f"Columns:     {df.columns.tolist()}")
    print(f"Null counts:\n{null_counts[null_counts > 0] if null_counts.any() else '  (none)'}")
    print(f"Date range:  {df['GAME_DATE'].min()}  →  {df['GAME_DATE'].max()}")


if __name__ == "__main__":
    main()
