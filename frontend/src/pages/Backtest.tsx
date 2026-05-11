import { useEffect, useState } from "react"
import { Loader2, RefreshCw } from "lucide-react"
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"
import { type BacktestResults, getBacktestResults } from "@/lib/api"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import { Skeleton } from "@/components/ui/skeleton"

const STAT_LABELS: Record<string, string> = {
  pts: "Points",
  reb: "Rebounds",
  ast: "Assists",
  fg3m: "3-Pointers",
}

function fmt(n: number, decimals = 3) {
  return n.toFixed(decimals)
}

function MetricsTable({ results }: { results: BacktestResults }) {
  const stats = ["pts", "reb", "ast", "fg3m"].filter((s) => s in results)
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b text-muted-foreground">
            <th className="pb-2 text-left font-medium">Stat</th>
            <th className="pb-2 text-right font-medium">Brier ↓</th>
            <th className="pb-2 text-right font-medium">CLV</th>
            <th className="pb-2 text-right font-medium">MAE ↓</th>
            <th className="pb-2 text-right font-medium">ROI</th>
            <th className="pb-2 text-right font-medium">Bets</th>
            <th className="pb-2 text-right font-medium">N</th>
          </tr>
        </thead>
        <tbody>
          {stats.map((s) => {
            const m = results[s]
            const clvPositive = m.clv > 0
            return (
              <tr key={s} className="border-b last:border-0">
                <td className="py-3 font-medium">{STAT_LABELS[s] ?? s}</td>
                <td className="py-3 text-right tabular-nums">{fmt(m.brier)}</td>
                <td className={`py-3 text-right tabular-nums font-medium ${clvPositive ? "text-green-600 dark:text-green-400" : "text-muted-foreground"}`}>
                  {clvPositive ? "+" : ""}{fmt(m.clv)}
                </td>
                <td className="py-3 text-right tabular-nums">{fmt(m.mae, 2)}</td>
                <td className={`py-3 text-right tabular-nums ${m.roi > 0 ? "text-green-600 dark:text-green-400" : "text-red-500"}`}>
                  {m.roi > 0 ? "+" : ""}{(m.roi * 100).toFixed(1)}%
                </td>
                <td className="py-3 text-right tabular-nums text-muted-foreground">{m.n_bets.toLocaleString()}</td>
                <td className="py-3 text-right tabular-nums text-muted-foreground">{m.n_samples.toLocaleString()}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function CLVChart({ results }: { results: BacktestResults }) {
  const data = ["pts", "reb", "ast", "fg3m"]
    .filter((s) => s in results)
    .map((s) => ({
      stat: STAT_LABELS[s] ?? s,
      clv: parseFloat((results[s].clv * 100).toFixed(2)),
    }))

  return (
    <ResponsiveContainer width="100%" height={180}>
      <BarChart data={data} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
        <XAxis dataKey="stat" tick={{ fontSize: 12 }} className="text-muted-foreground" />
        <YAxis tick={{ fontSize: 12 }} unit="%" className="text-muted-foreground" />
        <Tooltip
          formatter={(v) => [`${Number(v).toFixed(2)}%`, "CLV"]}
          contentStyle={{ fontSize: 12 }}
        />
        <Bar dataKey="clv" radius={[4, 4, 0, 0]}>
          {data.map((d) => (
            <Cell key={d.stat} fill={d.clv > 0 ? "hsl(142 71% 45%)" : "hsl(var(--muted-foreground))"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

export default function BacktestPage() {
  const [results, setResults] = useState<BacktestResults | null>(null)
  const [loading, setLoading] = useState(true)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [mode, setMode] = useState<"sim" | "live">("sim")

  useEffect(() => {
    getBacktestResults()
      .then(setResults)
      .catch(() => setResults(null))
      .finally(() => setLoading(false))
  }, [])

  async function handleRun() {
    setRunning(true)
    setError(null)
    try {
      const res = await getBacktestResults()
      setResults(res)
    } catch (e) {
      setError("No backtest results available yet.")
    } finally {
      setRunning(false)
    }
  }

  const isSimulated = results && Object.values(results)[0]?.mode === "sim"

  return (
    <div className="container max-w-2xl py-10">
      <div className="mb-8">
        <h1 className="text-2xl font-bold">Backtest Dashboard</h1>
        <p className="text-muted-foreground text-sm mt-1">
          Model performance vs DraftKings closing lines · Brier, CLV, MAE, flat-bet ROI
        </p>
      </div>

      <div className="flex items-center gap-3 mb-6">
        <div className="flex rounded-md border overflow-hidden">
          {(["sim", "live"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`px-4 py-1.5 text-sm font-medium transition-colors ${
                mode === m
                  ? "bg-primary text-primary-foreground"
                  : "bg-background text-muted-foreground hover:bg-accent"
              }`}
            >
              {m === "sim" ? "Simulated" : "Live DK"}
            </button>
          ))}
        </div>
        <Button
          size="sm"
          variant="outline"
          onClick={handleRun}
          disabled={running}
          className="gap-2"
        >
          {running ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <RefreshCw className="h-4 w-4" />
          )}
          {running ? "Loading…" : "Refresh"}
        </Button>
      </div>

      {isSimulated && (
        <div className="mb-4 rounded-md border border-yellow-500/30 bg-yellow-500/10 px-4 py-2 text-sm text-yellow-700 dark:text-yellow-400">
          <strong>Simulated lines</strong> — synthetic ±0.5 noise around season rolling avg.
          Switch to <em>Live DK</em> once 30+ days of closing lines are collected.
        </div>
      )}

      {error && (
        <div className="mb-4 rounded-md border border-destructive/50 bg-destructive/10 p-4 text-sm text-destructive">
          {error}
        </div>
      )}

      {loading && (
        <div className="space-y-3">
          <Skeleton className="h-8 w-full" />
          <Skeleton className="h-8 w-full" />
          <Skeleton className="h-8 w-full" />
          <Skeleton className="h-8 w-full" />
        </div>
      )}

      {!loading && !results && !error && (
        <div className="rounded-md border bg-muted/40 py-12 text-center">
          <p className="text-muted-foreground text-sm">No results yet.</p>
          <p className="text-muted-foreground text-xs mt-1">Click "Run backtest" to generate.</p>
        </div>
      )}

      {results && (
        <>
          <Card>
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="text-base">Metrics</CardTitle>
                {isSimulated && <Badge variant="secondary">SIMULATED</Badge>}
              </div>
              <CardDescription className="text-xs">
                Brier &lt; 0.25 = better than random · CLV &gt; 0 = positive market edge ·
                ROI at p_over &gt; 52% threshold
              </CardDescription>
            </CardHeader>
            <CardContent>
              <MetricsTable results={results} />
            </CardContent>
          </Card>

          <Card className="mt-4">
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Closing Line Value (CLV)</CardTitle>
              <CardDescription className="text-xs">
                Mean edge over vig-free market probability. The gold standard for model value.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <CLVChart results={results} />
            </CardContent>
          </Card>

          <Separator className="my-6" />
          <p className="text-xs text-muted-foreground">
            Season 2023-24 · {Object.values(results)[0]?.n_samples.toLocaleString() ?? "?"} prop/game pairs ·
            Trained with time-series CV (no data leakage) · Isotonic calibration applied
          </p>
        </>
      )}
    </div>
  )
}
