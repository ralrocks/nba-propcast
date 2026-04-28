import { BrowserRouter, Link, Route, Routes, useLocation } from "react-router-dom"
import BacktestPage from "./pages/Backtest"
import PredictPage from "./pages/Predict"

function Nav() {
  const loc = useLocation()
  return (
    <header className="sticky top-0 z-40 border-b bg-background/95 backdrop-blur">
      <div className="container flex h-14 items-center gap-6">
        <span className="font-bold text-sm tracking-tight">NBA PropCast</span>
        <nav className="flex items-center gap-4">
          <Link
            to="/"
            className={`text-sm transition-colors ${
              loc.pathname === "/"
                ? "text-foreground font-medium"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            Predict
          </Link>
          <Link
            to="/backtest"
            className={`text-sm transition-colors ${
              loc.pathname === "/backtest"
                ? "text-foreground font-medium"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            Backtest
          </Link>
        </nav>
      </div>
    </header>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-background text-foreground">
        <Nav />
        <main>
          <Routes>
            <Route path="/" element={<PredictPage />} />
            <Route path="/backtest" element={<BacktestPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
