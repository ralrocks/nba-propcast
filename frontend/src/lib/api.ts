const BASE = (import.meta.env.VITE_API_URL as string | undefined) ?? "http://localhost:8000"

export interface Player {
  player_id: number
  player_name: string
}

export interface Prediction {
  player_id: number
  stat: string
  point_estimate: number
  ci_low: number
  ci_high: number
  p_over: number | null
  n_games: number
  latest_game_date: string
}

export interface StatMetrics {
  brier: number
  clv: number
  mae: number
  roi: number
  n_bets: number
  n_samples: number
  mode: string
}

export type BacktestResults = Record<string, StatMetrics>

async function _json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    throw new Error(`${res.status}: ${text}`)
  }
  return res.json() as Promise<T>
}

export async function searchPlayers(q: string): Promise<Player[]> {
  return _json(await fetch(`${BASE}/players/search?q=${encodeURIComponent(q)}&limit=10`))
}

export async function predict(
  player_id: number,
  stat: string,
  line?: number,
): Promise<Prediction> {
  return _json(
    await fetch(`${BASE}/predict/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ player_id, stat, line: line ?? null }),
    }),
  )
}

export async function getBacktestResults(): Promise<BacktestResults> {
  return _json(await fetch(`${BASE}/backtest/results`))
}

export async function runBacktest(mode: "sim" | "live"): Promise<BacktestResults> {
  return _json(await fetch(`${BASE}/backtest/run?mode=${mode}`, { method: "POST" }))
}
