import { useState, useEffect, useRef } from 'react'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, BarChart, Bar, ReferenceLine
} from 'recharts'

const API_BASE = import.meta.env.VITE_API_URL || ''

const COLLECTIONS = [
  { slug: 'boredapeyachtclub', label: 'Bored Ape YC', symbol: 'BAYC' },
  { slug: 'cryptopunks', label: 'CryptoPunks', symbol: 'PUNK' },
  { slug: 'mutant-ape-yacht-club', label: 'Mutant Ape YC', symbol: 'MAYC' },
  { slug: 'azuki', label: 'Azuki', symbol: 'AZUKI' },
  { slug: 'pudgypenguins', label: 'Pudgy Penguins', symbol: 'PPG' },
  { slug: 'clonex', label: 'CloneX', symbol: 'CLONE' },
  { slug: 'moonbirds', label: 'Moonbirds', symbol: 'MOON' },
  { slug: 'doodles-official', label: 'Doodles', symbol: 'DOOD' },
]

function useForecasts() {
  const [results, setResults] = useState({})
  const [loading, setLoading] = useState({})
  const [error, setError] = useState(null)

  const forecast = async (collection, horizon = 7) => {
    const key = `${collection}-${horizon}`
    setLoading(l => ({ ...l, [key]: true }))
    setError(null)
    try {
      const res = await fetch(`${API_BASE}/api/forecast`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ collection, horizon, include_chart: true })
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setResults(r => ({ ...r, [key]: data }))
      return data
    } catch (e) {
      setError(e.message)
      return null
    } finally {
      setLoading(l => ({ ...l, [key]: false }))
    }
  }

  return { results, loading, error, forecast }
}

function GlowOrbs() {
  return (
    <>
      <div className="bg-orb" style={{ width: 600, height: 600, background: 'rgba(124,92,252,0.08)', top: '-200px', left: '-200px' }} />
      <div className="bg-orb" style={{ width: 400, height: 400, background: 'rgba(252,92,125,0.06)', bottom: '10%', right: '-100px', animationDelay: '-7s' }} />
      <div className="bg-orb" style={{ width: 300, height: 300, background: 'rgba(92,240,160,0.05)', top: '50%', left: '40%', animationDelay: '-13s' }} />
    </>
  )
}

function ConfidenceRing({ confidence, direction }) {
  const color = direction === 'UP' ? '#2dce89' : '#fc5c7d'
  const pct = Math.round(confidence * 100)
  const r = 42
  const circ = 2 * Math.PI * r
  const dash = (pct / 100) * circ

  return (
    <div className="relative flex items-center justify-center" style={{ width: 110, height: 110 }}>
      <svg width="110" height="110" style={{ transform: 'rotate(-90deg)' }}>
        <circle cx="55" cy="55" r={r} fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="6" />
        <circle
          cx="55" cy="55" r={r} fill="none"
          stroke={color} strokeWidth="6"
          strokeDasharray={`${dash} ${circ}`}
          strokeLinecap="round"
          style={{ transition: 'stroke-dasharray 1.2s ease', filter: `drop-shadow(0 0 6px ${color})` }}
        />
      </svg>
      <div className="absolute flex flex-col items-center">
        <span className="font-bold text-xl" style={{ color, fontFamily: 'Syne', lineHeight: 1 }}>{pct}%</span>
        <span className="text-xs" style={{ color: 'var(--text-muted)', fontFamily: 'Space Mono' }}>conf</span>
      </div>
    </div>
  )
}

function MiniChart({ data }) {
  if (!data || data.length === 0) return null
  const min = Math.min(...data.map(d => d.floor))
  const max = Math.max(...data.map(d => d.floor))
  const isUp = data[data.length - 1]?.floor >= data[0]?.floor

  return (
    <ResponsiveContainer width="100%" height={140}>
      <AreaChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
        <defs>
          <linearGradient id="floorGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={isUp ? '#2dce89' : '#fc5c7d'} stopOpacity={0.35} />
            <stop offset="95%" stopColor={isUp ? '#2dce89' : '#fc5c7d'} stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
        <XAxis
          dataKey="date"
          tick={false}
          axisLine={{ stroke: 'rgba(255,255,255,0.08)' }}
          tickLine={false}
        />
        <YAxis
          domain={[min * 0.9, max * 1.1]}
          tick={{ fontSize: 9, fill: 'rgba(255,255,255,0.3)', fontFamily: 'Space Mono' }}
          width={38}
          axisLine={false}
          tickLine={false}
          tickFormatter={v => `${v.toFixed(1)}Ξ`}
        />
        <Tooltip
          contentStyle={{ background: '#13131a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, fontSize: 11, fontFamily: 'Space Mono' }}
          labelStyle={{ color: 'rgba(255,255,255,0.5)' }}
          itemStyle={{ color: isUp ? '#2dce89' : '#fc5c7d' }}
          formatter={v => [`${v.toFixed(4)} ETH`, 'Floor']}
        />
        <Area
          type="monotone" dataKey="floor"
          stroke={isUp ? '#2dce89' : '#fc5c7d'} strokeWidth={2}
          fill="url(#floorGrad)" dot={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  )
}

function ForecastCard({ result, loading, onRefresh, collection }) {
  const isUp = result?.direction === 'UP'
  const color = isUp ? '#2dce89' : '#fc5c7d'
  const symbol = COLLECTIONS.find(c => c.slug === collection)?.symbol || collection.toUpperCase().slice(0, 5)

  return (
    <div className="card p-5 flex flex-col gap-4 animate-slide-up" style={{ minHeight: 360 }}>
      {/* Header */}
      <div className="flex items-start justify-between gap-2">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="pill pill-neutral text-xs">{symbol}</span>
            {result?.data_source === 'synthetic_demo' && (
              <span className="pill" style={{ background: 'rgba(255,200,50,0.1)', color: '#ffc832', border: '1px solid rgba(255,200,50,0.25)', fontSize: 10 }}>
                DEMO DATA
              </span>
            )}
          </div>
          <h3 className="font-bold text-base leading-tight" style={{ fontFamily: 'Syne' }}>
            {result?.collection_name || collection}
          </h3>
          {result && (
            <div className="text-xs mt-1" style={{ color: 'var(--text-muted)', fontFamily: 'Space Mono' }}>
              Floor: <span style={{ color: 'var(--text)' }}>{result.current_floor_eth}Ξ</span>
            </div>
          )}
        </div>
        {result && (
          <ConfidenceRing confidence={result.confidence} direction={result.direction} />
        )}
      </div>

      {/* Result */}
      {loading && (
        <div className="flex-1 flex items-center justify-center">
          <div style={{ width: 32, height: 32, borderRadius: '50%', border: '3px solid rgba(124,92,252,0.2)', borderTopColor: 'var(--accent)', animation: 'spin 0.8s linear infinite' }} />
        </div>
      )}

      {result && !loading && (
        <>
          {/* Direction + magnitude */}
          <div className="card-inner p-4 flex items-center justify-between">
            <div>
              <div className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>7-day forecast</div>
              <div className="flex items-center gap-3">
                <span className={`pill ${isUp ? 'pill-up' : 'pill-down'} text-sm font-bold`} style={{ fontSize: 14, padding: '5px 14px' }}>
                  {isUp ? '▲' : '▼'} {result.direction}
                </span>
                <span className="text-2xl font-bold" style={{ color, fontFamily: 'Syne' }}>
                  {result.predicted_pct_change > 0 ? '+' : ''}{result.predicted_pct_change}%
                </span>
              </div>
            </div>
            <div className="text-right">
              <div className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>Target</div>
              <div className="font-bold" style={{ fontFamily: 'Syne', fontSize: 18 }}>
                {result.predicted_floor_eth}Ξ
              </div>
            </div>
          </div>

          {/* Chart */}
          {result.chart_data && (
            <MiniChart data={result.chart_data.slice(-60)} />
          )}
        </>
      )}

      {!result && !loading && (
        <div className="flex-1 flex items-center justify-center">
          <button
            onClick={onRefresh}
            className="px-5 py-2.5 rounded-xl font-bold text-sm transition-all"
            style={{ background: 'rgba(124,92,252,0.15)', border: '1px solid rgba(124,92,252,0.3)', color: 'var(--accent)', fontFamily: 'Space Mono', cursor: 'pointer' }}
            onMouseEnter={e => e.target.style.background = 'rgba(124,92,252,0.25)'}
            onMouseLeave={e => e.target.style.background = 'rgba(124,92,252,0.15)'}
          >
            Run Forecast →
          </button>
        </div>
      )}
    </div>
  )
}

function SearchBar({ onSearch, loading }) {
  const [value, setValue] = useState('')
  const [horizon, setHorizon] = useState(7)

  const handleSubmit = (e) => {
    e.preventDefault()
    if (value.trim()) onSearch(value.trim(), horizon)
  }

  return (
    <form onSubmit={handleSubmit} className="flex gap-2 items-center w-full max-w-2xl mx-auto">
      <input
        value={value}
        onChange={e => setValue(e.target.value)}
        placeholder="Collection slug or 0x address (e.g. azuki, boredapeyachtclub)"
        className="flex-1 px-4 py-3 rounded-xl text-sm outline-none transition-all"
        style={{
          background: 'var(--bg-2)', border: '1px solid var(--border)',
          color: 'var(--text)', fontFamily: 'Space Mono', fontSize: 12,
        }}
        onFocus={e => e.target.style.borderColor = 'rgba(124,92,252,0.5)'}
        onBlur={e => e.target.style.borderColor = 'var(--border)'}
      />
      <select
        value={horizon}
        onChange={e => setHorizon(Number(e.target.value))}
        className="px-3 py-3 rounded-xl text-sm outline-none"
        style={{ background: 'var(--bg-2)', border: '1px solid var(--border)', color: 'var(--text)', fontFamily: 'Space Mono', fontSize: 12, cursor: 'pointer' }}
      >
        <option value={7}>7d</option>
        <option value={14}>14d</option>
        <option value={30}>30d</option>
      </select>
      <button
        type="submit"
        disabled={loading || !value.trim()}
        className="px-5 py-3 rounded-xl font-bold text-sm transition-all"
        style={{
          background: loading ? 'rgba(124,92,252,0.2)' : 'var(--accent)',
          color: 'white', border: 'none', fontFamily: 'Space Mono',
          cursor: loading ? 'not-allowed' : 'pointer',
          boxShadow: loading ? 'none' : '0 0 20px rgba(124,92,252,0.4)',
        }}
      >
        {loading ? '...' : 'Forecast'}
      </button>
    </form>
  )
}

function StatBar({ label, value, max, color }) {
  const pct = Math.min((value / max) * 100, 100)
  return (
    <div className="flex items-center gap-3">
      <div className="text-xs w-16 shrink-0" style={{ color: 'var(--text-muted)', fontFamily: 'Space Mono' }}>{label}</div>
      <div className="flex-1 conf-bar">
        <div className="conf-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <div className="text-xs w-12 text-right" style={{ color: 'var(--text)', fontFamily: 'Space Mono' }}>{value}</div>
    </div>
  )
}

function DetailPanel({ result, onClose }) {
  if (!result) return null
  const isUp = result.direction === 'UP'
  const color = isUp ? '#2dce89' : '#fc5c7d'

  const volumeData = result.chart_data?.slice(-30).map(d => ({
    ...d,
    color: d.volume > (result.chart_data.reduce((s, x) => s + x.volume, 0) / result.chart_data.length) ? color : 'rgba(255,255,255,0.2)'
  }))

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: 'rgba(6,6,8,0.85)', backdropFilter: 'blur(12px)' }}
      onClick={onClose}
    >
      <div
        className="card w-full max-w-3xl p-6 animate-slide-up"
        style={{ maxHeight: '90vh', overflowY: 'auto' }}
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-start justify-between mb-6">
          <div>
            <div className="pill pill-neutral mb-2">{result.collection_name}</div>
            <h2 style={{ fontFamily: 'Syne', fontSize: 26, fontWeight: 800 }}>Forecast Detail</h2>
          </div>
          <button onClick={onClose} style={{ color: 'var(--text-muted)', background: 'none', border: 'none', fontSize: 20, cursor: 'pointer' }}>✕</button>
        </div>

        {/* Key metrics */}
        <div className="grid grid-cols-2 gap-3 mb-6" style={{ gridTemplateColumns: 'repeat(4, 1fr)' }}>
          {[
            { label: 'Direction', val: result.direction, style: { color } },
            { label: 'Δ%', val: `${result.predicted_pct_change > 0 ? '+' : ''}${result.predicted_pct_change}%`, style: { color } },
            { label: 'P(Up)', val: `${Math.round(result.prob_up * 100)}%`, style: {} },
            { label: 'Confidence', val: `${Math.round(result.confidence * 100)}%`, style: {} },
          ].map(m => (
            <div key={m.label} className="card-inner p-3 text-center">
              <div className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>{m.label}</div>
              <div className="font-bold text-lg" style={{ fontFamily: 'Syne', ...m.style }}>{m.val}</div>
            </div>
          ))}
        </div>

        {/* Floor chart */}
        <div className="mb-2 text-xs" style={{ color: 'var(--text-muted)' }}>90-DAY FLOOR PRICE (ETH)</div>
        <div style={{ height: 180 }} className="mb-6">
          <MiniChart data={result.chart_data} />
        </div>

        {/* Volume bars */}
        {volumeData && (
          <>
            <div className="mb-2 text-xs" style={{ color: 'var(--text-muted)' }}>30-DAY VOLUME (ETH)</div>
            <div style={{ height: 100 }} className="mb-6">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={volumeData} margin={{ top: 2, right: 4, bottom: 0, left: 0 }}>
                  <XAxis dataKey="date" tick={false} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fontSize: 9, fill: 'rgba(255,255,255,0.3)', fontFamily: 'Space Mono' }} width={36} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{ background: '#13131a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, fontSize: 11, fontFamily: 'Space Mono' }} />
                  <Bar dataKey="volume" fill={color} opacity={0.7} radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </>
        )}

        {/* Address */}
        <div className="text-xs" style={{ color: 'var(--text-muted)', wordBreak: 'break-all' }}>
          Contract: <span style={{ color: 'var(--accent)' }}>{result.collection_address}</span>
        </div>
        <div className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
          Source: {result.data_source} · Generated: {new Date(result.generated_at * 1000).toLocaleString()}
        </div>
        <div className="mt-4 p-3 rounded-xl text-xs" style={{ background: 'rgba(255,200,50,0.05)', border: '1px solid rgba(255,200,50,0.15)', color: 'rgba(255,200,50,0.7)', fontFamily: 'Space Mono' }}>
          ⚠ This is a machine learning forecast for research purposes only. Not financial advice. NFT markets are highly volatile.
        </div>
      </div>
    </div>
  )
}

export default function App() {
  const { results, loading, error, forecast } = useForecasts()
  const [activeCollection, setActiveCollection] = useState(null)
  const [customResult, setCustomResult] = useState(null)
  const [customLoading, setCustomLoading] = useState(false)
  const [initLoaded, setInitLoaded] = useState(false)
  const [detailResult, setDetailResult] = useState(null)

  // Auto-load 4 collections on mount
  useEffect(() => {
    const initial = COLLECTIONS.slice(0, 4)
    const load = async () => {
      for (const col of initial) {
        await forecast(col.slug, 7)
        await new Promise(r => setTimeout(r, 200))
      }
      setInitLoaded(true)
    }
    load()
  }, [])

  const handleSearch = async (collection, horizon) => {
    setCustomLoading(true)
    const res = await forecast(collection, horizon)
    if (res) setCustomResult(res)
    setCustomLoading(false)
  }

  const handleCollectionForecast = (slug) => {
    forecast(slug, 7)
  }

  return (
    <div className="min-h-screen" style={{ position: 'relative' }}>
      <GlowOrbs />

      {/* Nav */}
      <nav className="sticky top-0 z-40 flex items-center justify-between px-6 py-4"
        style={{ background: 'rgba(6,6,8,0.85)', backdropFilter: 'blur(20px)', borderBottom: '1px solid var(--border)' }}>
        <div className="flex items-center gap-3">
          <span style={{ fontSize: 22 }}>◈</span>
          <span style={{ fontFamily: 'Syne', fontWeight: 800, fontSize: 18, letterSpacing: '-0.02em' }}>
            NFT Floor Forecaster
          </span>
          <span className="pill pill-neutral" style={{ fontSize: 10 }}>BETA</span>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-xs" style={{ color: 'var(--text-muted)', fontFamily: 'Space Mono' }}>
            PatchTST + CLIP · Multimodal
          </span>
          <a href="/api/docs" target="_blank" rel="noreferrer"
            className="text-xs px-3 py-1.5 rounded-lg transition-colors"
            style={{ color: 'var(--accent)', background: 'rgba(124,92,252,0.1)', border: '1px solid rgba(124,92,252,0.25)', fontFamily: 'Space Mono', textDecoration: 'none' }}>
            API Docs →
          </a>
        </div>
      </nav>

      <main className="px-6 py-10 max-w-7xl mx-auto">
        {/* Hero */}
        <div className="text-center mb-12 animate-slide-up">
          <div className="pill pill-neutral mx-auto mb-4" style={{ display: 'inline-flex' }}>
            Multimodal AI · Time-series + Vision
          </div>
          <h1 style={{ fontFamily: 'Syne', fontSize: 'clamp(32px, 5vw, 60px)', fontWeight: 800, letterSpacing: '-0.03em', lineHeight: 1.1 }}>
            Predict NFT Floor Prices<br />
            <span style={{ background: 'linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
              With AI Precision
            </span>
          </h1>
          <p className="mt-4 text-sm max-w-xl mx-auto" style={{ color: 'var(--text-muted)', fontFamily: 'Space Mono', lineHeight: 1.8 }}>
            Dual-tower neural network combining 90-day on-chain time-series with CLIP visual embeddings.
            Deployable on OpenGradient for verifiable on-chain inference.
          </p>
        </div>

        {/* Search */}
        <div className="mb-10 animate-slide-up" style={{ animationDelay: '0.1s' }}>
          <SearchBar onSearch={handleSearch} loading={customLoading} />
          {error && (
            <div className="text-center mt-2 text-xs" style={{ color: '#fc5c7d', fontFamily: 'Space Mono' }}>
              Error: {error}
            </div>
          )}
        </div>

        {/* Custom result */}
        {customResult && (
          <div className="mb-8 animate-slide-up">
            <div className="text-xs mb-3" style={{ color: 'var(--text-muted)', fontFamily: 'Space Mono' }}>CUSTOM QUERY</div>
            <div className="grid gap-4" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))' }}>
              <div
                className="cursor-pointer transition-transform hover:scale-[1.01]"
                onClick={() => setDetailResult(customResult)}
              >
                <ForecastCard
                  result={customResult}
                  loading={customLoading}
                  collection={customResult.collection_name}
                  onRefresh={() => {}}
                />
              </div>
            </div>
          </div>
        )}

        {/* Collection grid */}
        <div className="mb-4 flex items-center justify-between">
          <div className="text-xs" style={{ color: 'var(--text-muted)', fontFamily: 'Space Mono' }}>
            TOP COLLECTIONS · 7-DAY FORECAST
          </div>
          <div className="text-xs" style={{ color: 'var(--text-muted)', fontFamily: 'Space Mono' }}>
            {Object.keys(results).filter(k => k.endsWith('-7')).length}/{COLLECTIONS.length} loaded
          </div>
        </div>

        <div className="grid gap-4" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))' }}>
          {COLLECTIONS.map((col, i) => {
            const key = `${col.slug}-7`
            const res = results[key]
            const isLoading = loading[key]
            return (
              <div
                key={col.slug}
                className="cursor-pointer transition-transform hover:scale-[1.01]"
                style={{ animationDelay: `${i * 0.08}s` }}
                onClick={() => res && setDetailResult(res)}
              >
                <ForecastCard
                  result={res}
                  loading={isLoading}
                  collection={col.slug}
                  onRefresh={() => handleCollectionForecast(col.slug)}
                />
              </div>
            )
          })}
        </div>

        {/* Model architecture info */}
        <div className="mt-16 card p-8 animate-slide-up" style={{ animationDelay: '0.3s' }}>
          <h2 className="mb-6" style={{ fontFamily: 'Syne', fontWeight: 800, fontSize: 22 }}>Model Architecture</h2>
          <div className="grid gap-6" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))' }}>
            {[
              { icon: '⏱', title: 'Time-Series Tower', desc: 'PatchTST encoder — 4-layer, 128-dim transformer on 90-day patches of floor price, volume, sales + 7 derived features' },
              { icon: '👁', title: 'Vision Tower', desc: 'Frozen CLIP ViT-B/32 collection embeddings (512-dim) capturing visual trait DNA across 50 sample tokens per collection' },
              { icon: '⚡', title: 'Gated Fusion', desc: 'Cross-attention gate learns to weight visual vs. time-series signals dynamically per collection and market regime' },
              { icon: '🎯', title: 'Dual Output', desc: 'Direction head (BCE) + magnitude head (Huber loss). Predicts both which way and how much floor will move' },
              { icon: '🔐', title: 'TEE-Ready', desc: 'ONNX export (opset 17) + INT8 quantization for OpenGradient verifiable inference via TEE or ZKML modes' },
              { icon: '📊', title: '~2.5M Params', desc: 'Trains in <1 hour on a consumer GPU. Pure CPU inference in <100ms. Deployable on Railway free tier' },
            ].map(item => (
              <div key={item.title} className="card-inner p-4">
                <div className="text-2xl mb-2">{item.icon}</div>
                <div className="font-bold text-sm mb-1" style={{ fontFamily: 'Syne' }}>{item.title}</div>
                <div className="text-xs leading-relaxed" style={{ color: 'var(--text-muted)', fontFamily: 'Space Mono' }}>{item.desc}</div>
              </div>
            ))}
          </div>
        </div>

        {/* API snippet */}
        <div className="mt-8 card p-8">
          <h2 className="mb-4" style={{ fontFamily: 'Syne', fontWeight: 800, fontSize: 22 }}>API Usage</h2>
          <pre className="rounded-xl p-5 text-xs overflow-x-auto" style={{ background: 'var(--bg-3)', border: '1px solid var(--border)', color: '#5cf0a0', fontFamily: 'Space Mono', lineHeight: 1.7 }}>
{`# POST /api/forecast
curl -X POST https://your-app.railway.app/api/forecast \\
  -H "Content-Type: application/json" \\
  -d '{
    "collection": "boredapeyachtclub",
    "horizon": 7,
    "include_chart": true
  }'

# Response
{
  "direction": "DOWN",
  "confidence": 0.712,
  "predicted_pct_change": -4.32,
  "predicted_floor_eth": 38.21,
  "current_floor_eth": 39.94,
  "chart_data": [...]
}`}
          </pre>
        </div>

        <footer className="mt-12 py-6 text-center text-xs" style={{ color: 'var(--text-muted)', fontFamily: 'Space Mono', borderTop: '1px solid var(--border)' }}>
          NFT Floor Forecaster · Multimodal AI · Built for OpenGradient Hub ·{' '}
          <span style={{ color: '#fc5c7d' }}>Not financial advice</span>
        </footer>
      </main>

      {/* Detail modal */}
      {detailResult && <DetailPanel result={detailResult} onClose={() => setDetailResult(null)} />}

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>
    </div>
  )
}
