import { useEffect, useRef, useState } from "react"
import { Loader2, Search, TrendingUp } from "lucide-react"
import { type Player, type Prediction, predict, searchPlayers } from "@/lib/api"
import { useDebounce } from "@/hooks/useDebounce"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Separator } from "@/components/ui/separator"
import { Skeleton } from "@/components/ui/skeleton"

const STATS = [
  { value: "pts", label: "Points" },
  { value: "reb", label: "Rebounds" },
  { value: "ast", label: "Assists" },
  { value: "fg3m", label: "3-Pointers" },
] as const

type Stat = (typeof STATS)[number]["value"]

function CIBar({ low, est, high }: { low: number; est: number; high: number }) {
  const min = Math.max(0, low - 2)
  const max = high + 2
  const range = max - min
  const lowPct = ((low - min) / range) * 100
  const highPct = ((high - min) / range) * 100
  const estPct = ((est - min) / range) * 100

  return (
    <div className="mt-4">
      <div className="flex justify-between text-xs text-muted-foreground mb-1">
        <span>{low.toFixed(1)}</span>
        <span className="font-medium text-foreground">{est.toFixed(1)}</span>
        <span>{high.toFixed(1)}</span>
      </div>
      <div className="relative h-3 rounded-full bg-muted overflow-hidden">
        <div
          className="absolute h-full bg-primary/20 rounded-full"
          style={{ left: `${lowPct}%`, width: `${highPct - lowPct}%` }}
        />
        <div
          className="absolute h-full w-1 bg-primary rounded-full -translate-x-1/2"
          style={{ left: `${estPct}%` }}
        />
      </div>
      <p className="text-xs text-muted-foreground mt-1 text-center">80% confidence interval</p>
    </div>
  )
}

function PredictionCard({ result, playerName }: { result: Prediction; playerName: string }) {
  const statLabel = STATS.find((s) => s.value === result.stat)?.label ?? result.stat
  const pOver = result.p_over
  const hasEdge = pOver !== null && pOver > 0.52

  return (
    <Card className="mt-6">
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between">
          <div>
            <CardTitle className="text-lg">{playerName}</CardTitle>
            <CardDescription>{statLabel} projection</CardDescription>
          </div>
          {pOver !== null && (
            <Badge variant={hasEdge ? "default" : "secondary"} className="text-sm">
              {(pOver * 100).toFixed(1)}% over
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <div className="flex items-baseline gap-2">
          <span className="text-5xl font-bold tabular-nums">
            {result.point_estimate.toFixed(1)}
          </span>
          <span className="text-muted-foreground text-sm">{statLabel.toLowerCase()}</span>
        </div>

        <CIBar low={result.ci_low} est={result.point_estimate} high={result.ci_high} />

        <Separator className="my-4" />

        <div className="grid grid-cols-2 gap-3 text-sm">
          <div>
            <p className="text-muted-foreground">Model</p>
            <p className="font-medium">XGBoost + Isotonic cal.</p>
          </div>
          <div>
            <p className="text-muted-foreground">Based on</p>
            <p className="font-medium">{result.n_games} games</p>
          </div>
          <div>
            <p className="text-muted-foreground">Last game</p>
            <p className="font-medium">{result.latest_game_date}</p>
          </div>
          {pOver !== null && (
            <div>
              <p className="text-muted-foreground">P(over line)</p>
              <p className={`font-medium ${hasEdge ? "text-green-600 dark:text-green-400" : ""}`}>
                {(pOver * 100).toFixed(1)}%
                {hasEdge && " ✓ edge"}
              </p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

export default function PredictPage() {
  const [query, setQuery] = useState("")
  const [players, setPlayers] = useState<Player[]>([])
  const [selectedPlayer, setSelectedPlayer] = useState<Player | null>(null)
  const [showDropdown, setShowDropdown] = useState(false)
  const [searching, setSearching] = useState(false)
  const [stat, setStat] = useState<Stat>("pts")
  const [line, setLine] = useState("")
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<Prediction | null>(null)
  const [error, setError] = useState<string | null>(null)

  const debouncedQuery = useDebounce(query, 280)
  const dropdownRef = useRef<HTMLDivElement>(null)

  // Close dropdown on outside click
  useEffect(() => {
    function handler(e: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setShowDropdown(false)
      }
    }
    document.addEventListener("mousedown", handler)
    return () => document.removeEventListener("mousedown", handler)
  }, [])

  // Search players when query changes
  useEffect(() => {
    if (debouncedQuery.length < 2) {
      setPlayers([])
      return
    }
    setSearching(true)
    searchPlayers(debouncedQuery)
      .then((res) => {
        setPlayers(res)
        setShowDropdown(res.length > 0)
      })
      .catch(() => setPlayers([]))
      .finally(() => setSearching(false))
  }, [debouncedQuery])

  function selectPlayer(p: Player) {
    setSelectedPlayer(p)
    setQuery(p.player_name)
    setShowDropdown(false)
    setResult(null)
    setError(null)
  }

  async function handlePredict() {
    if (!selectedPlayer) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const lineVal = line ? parseFloat(line) : undefined
      const res = await predict(selectedPlayer.player_id, stat, lineVal)
      setResult(res)
    } catch (e) {
      setError(e instanceof Error ? e.message : "Prediction failed")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container max-w-xl py-10">
      <div className="mb-8">
        <div className="flex items-center gap-2 mb-1">
          <TrendingUp className="h-5 w-5 text-primary" />
          <h1 className="text-2xl font-bold">PropCast</h1>
        </div>
        <p className="text-muted-foreground text-sm">
          XGBoost projections with 80% confidence intervals and market edge detection
        </p>
      </div>

      {/* Player search */}
      <div className="space-y-4">
        <div className="relative" ref={dropdownRef}>
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            className="pl-9"
            placeholder="Search player…"
            value={query}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
              setQuery(e.target.value)
              setSelectedPlayer(null)
              setResult(null)
            }}
            onFocus={() => players.length > 0 && setShowDropdown(true)}
          />
          {searching && (
            <Loader2 className="absolute right-3 top-1/2 -translate-y-1/2 h-4 w-4 animate-spin text-muted-foreground" />
          )}
          {showDropdown && (
            <div className="absolute z-10 mt-1 w-full rounded-md border bg-popover shadow-md">
              {players.map((p) => (
                <button
                  key={p.player_id}
                  className="w-full px-4 py-2 text-left text-sm hover:bg-accent hover:text-accent-foreground first:rounded-t-md last:rounded-b-md"
                  onMouseDown={() => selectPlayer(p)}
                >
                  {p.player_name}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Stat selector */}
        <div className="flex gap-2">
          {STATS.map((s) => (
            <button
              key={s.value}
              onClick={() => {
                setStat(s.value)
                setResult(null)
              }}
              className={`flex-1 rounded-md border px-3 py-2 text-xs font-medium transition-colors ${
                stat === s.value
                  ? "bg-primary text-primary-foreground border-primary"
                  : "bg-background text-muted-foreground hover:bg-accent hover:text-accent-foreground"
              }`}
            >
              {s.label}
            </button>
          ))}
        </div>

        {/* Optional line */}
        <div className="flex gap-2">
          <Input
            type="number"
            step="0.5"
            min="0"
            placeholder="DK line (optional, e.g. 22.5)"
            value={line}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setLine(e.target.value)}
            className="flex-1"
          />
          <Button
            onClick={handlePredict}
            disabled={!selectedPlayer || loading}
            className="min-w-[110px]"
          >
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : "Predict"}
          </Button>
        </div>
      </div>

      {/* Loading skeleton */}
      {loading && (
        <div className="mt-6 space-y-3">
          <Skeleton className="h-6 w-40" />
          <Skeleton className="h-14 w-28" />
          <Skeleton className="h-8 w-full" />
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="mt-6 rounded-md border border-destructive/50 bg-destructive/10 p-4 text-sm text-destructive">
          {error}
        </div>
      )}

      {/* Result */}
      {result && selectedPlayer && (
        <PredictionCard result={result} playerName={selectedPlayer.player_name} />
      )}
    </div>
  )
}
