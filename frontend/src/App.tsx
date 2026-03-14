import { useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react'
import * as XLSX from 'xlsx'

type Page = 'landing' | 'dashboard'

type SignalLight = 'Red' | 'Yellow' | 'Green'

type EnvironmentNode = {
  phase: 0 | 1
  q1: number
  q2: number
  wait1: number
  wait2: number
  northSouthLight?: SignalLight
  eastWestLight?: SignalLight
}

type EnvironmentTick = {
  time: number
  intersections: Record<string, EnvironmentNode>
}

type RawDirectionData = {
  light?: string
  queue?: number
  wait?: number
}

type RawNestedNode = {
  North_South?: RawDirectionData
  East_West?: RawDirectionData
}

type DerivedNode = {
  id: string
  phase: 0 | 1
  q1: number
  q2: number
  totalQueue: number
  wait1: number
  wait2: number
  avgWait: number
}

type IncidentEvent = {
  time: string
  level: 'info' | 'warning' | 'critical'
  message: string
}

type InsightData = {
  title: string
  unit: string
  labels: string[]
  values: number[]
  description: string
}

type InsightSelection =
  | { kind: 'totalQueue' }
  | { kind: 'avgWait' }
  | { kind: 'greenSignals' }
  | { kind: 'peakNode' }
  | { kind: 'nodeQueue'; nodeId: string }

const FALLBACK_TICK: EnvironmentTick = {
  time: 0,
  intersections: {},
}

const MAX_HISTORY_FRAMES = 2000

function isEnvironmentTick(value: unknown): value is EnvironmentTick {
  if (!value || typeof value !== 'object') {
    return false
  }

  const possibleTick = value as Partial<EnvironmentTick>
  if (typeof possibleTick.time !== 'number') {
    return false
  }

  if (!possibleTick.intersections || typeof possibleTick.intersections !== 'object') {
    return false
  }

  return Object.values(possibleTick.intersections).every((node) => {
    const typedNode = node as Partial<EnvironmentNode>
    return (
      (typedNode.phase === 0 || typedNode.phase === 1) &&
      typeof typedNode.q1 === 'number' &&
      typeof typedNode.q2 === 'number' &&
      typeof typedNode.wait1 === 'number' &&
      typeof typedNode.wait2 === 'number'
    )
  })
}

function parsePhase(lightValue: unknown): 0 | 1 {
  if (typeof lightValue !== 'string') {
    return 0
  }

  return lightValue.toLowerCase() === 'green' ? 1 : 0
}

function parseSignalLight(lightValue: unknown): SignalLight {
  if (typeof lightValue !== 'string') {
    return 'Red'
  }

  const normalized = lightValue.trim().toLowerCase()
  if (normalized === 'green') {
    return 'Green'
  }

  if (normalized === 'yellow' || normalized === 'amber' || normalized === 'y') {
    return 'Yellow'
  }

  return 'Red'
}

function normalizeBridgeTick(value: unknown): EnvironmentTick | null {
  if (isEnvironmentTick(value)) {
    const normalizedIntersections: Record<string, EnvironmentNode> = {}

    for (const [nodeId, node] of Object.entries(value.intersections)) {
      normalizedIntersections[nodeId] = {
        ...node,
        northSouthLight: node.phase === 0 ? 'Green' : 'Red',
        eastWestLight: node.phase === 1 ? 'Green' : 'Red',
      }
    }

    return {
      time: value.time,
      intersections: normalizedIntersections,
    }
  }

  if (!value || typeof value !== 'object') {
    return null
  }

  const candidate = value as {
    time?: unknown
    intersections?: Record<string, RawNestedNode>
  }

  if (typeof candidate.time !== 'number') {
    return null
  }

  if (!candidate.intersections || typeof candidate.intersections !== 'object') {
    return null
  }

  const normalizedIntersections: Record<string, EnvironmentNode> = {}

  for (const [nodeId, nodeValue] of Object.entries(candidate.intersections)) {
    if (!nodeValue || typeof nodeValue !== 'object') {
      return null
    }

    const northSouth = nodeValue.North_South ?? {}
    const eastWest = nodeValue.East_West ?? {}

    const q1 = typeof northSouth.queue === 'number' ? northSouth.queue : 0
    const q2 = typeof eastWest.queue === 'number' ? eastWest.queue : 0
    const wait1 = typeof northSouth.wait === 'number' ? northSouth.wait : 0
    const wait2 = typeof eastWest.wait === 'number' ? eastWest.wait : 0
    const northSouthLight = parseSignalLight(northSouth.light)
    const eastWestLight = parseSignalLight(eastWest.light)

    let phase: 0 | 1 = parsePhase(northSouth.light)
    if (eastWestLight === 'Green') {
      phase = 1
    } else if (northSouthLight === 'Green') {
      phase = 0
    }

    normalizedIntersections[nodeId] = {
      phase,
      q1,
      q2,
      wait1,
      wait2,
      northSouthLight,
      eastWestLight,
    }
  }

  return {
    time: candidate.time,
    intersections: normalizedIntersections,
  }
}

function App() {
  const [page, setPage] = useState<Page>('landing')
  const [environmentHistory, setEnvironmentHistory] = useState<EnvironmentTick[]>([
    FALLBACK_TICK,
  ])

  useEffect(() => {
    let cancelled = false

    const pollBridge = async () => {
      try {
        const response = await fetch('/data_bridge.json', { cache: 'no-store' })
        if (!response.ok) {
          return
        }

        const payload: unknown = await response.json()
        const normalizedPayload = normalizeBridgeTick(payload)

        if (!normalizedPayload || cancelled) {
          return
        }

        setEnvironmentHistory((previousHistory) => {
          const latest = previousHistory[previousHistory.length - 1]
          if (!latest || normalizedPayload.time > latest.time) {
            const withoutFallback =
              previousHistory.length === 1 && previousHistory[0].time === 0
                ? []
                : previousHistory

            const nextHistory = [...withoutFallback, normalizedPayload]
            return nextHistory.slice(-MAX_HISTORY_FRAMES)
          }

          if (normalizedPayload.time === latest.time) {
            return [...previousHistory.slice(0, -1), normalizedPayload]
          }

          return previousHistory
        })
      } catch {
        return
      }
    }

    pollBridge()
    const timer = window.setInterval(pollBridge, 1000)

    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [])

  return page === 'landing' ? (
    <LandingPage
      onEnterDashboard={() => setPage('dashboard')}
      environmentHistory={environmentHistory}
    />
  ) : (
    <DashboardPage
      onBackToLanding={() => setPage('landing')}
      environmentHistory={environmentHistory}
    />
  )
}

type LandingPageProps = {
  onEnterDashboard: () => void
  environmentHistory: EnvironmentTick[]
}

function LandingPage({ onEnterDashboard, environmentHistory }: LandingPageProps) {
  const latest = environmentHistory[environmentHistory.length - 1]
  const nodeCount = Object.keys(latest.intersections).length

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100">
      <div className="fixed inset-0 -z-10 bg-[radial-gradient(circle_at_8%_10%,rgba(56,189,248,0.14),transparent_34%),radial-gradient(circle_at_88%_6%,rgba(129,140,248,0.12),transparent_30%),radial-gradient(circle_at_50%_100%,rgba(45,212,191,0.08),transparent_42%)]" />

      <section className="mx-auto flex min-h-screen w-full max-w-6xl flex-col items-center justify-center px-6 py-10 text-center sm:px-8">
        <p className="text-xs font-semibold uppercase tracking-[0.28em] text-sky-200/90">
          City Wide Traffic Monitor
        </p>

        <h1 className="mt-4 max-w-4xl text-5xl font-bold leading-tight text-white sm:text-6xl lg:text-7xl">
          6IX STREETS
        </h1>

        <p className="mt-4 max-w-3xl text-sm text-slate-300 sm:text-base">
          Real-time traffic insights that help reduce congestion, shorten waits,
          and keep city movement smooth.
        </p>

        <div className="mt-8 flex flex-wrap justify-center gap-3">
          <button
            type="button"
            onClick={onEnterDashboard}
            className="rounded-xl bg-sky-300 px-6 py-3 text-sm font-semibold text-slate-950 shadow-lg shadow-sky-400/10 transition hover:bg-sky-200"
          >
            View Live Dashboard
          </button>
        </div>

        <div className="mt-12 grid w-full max-w-5xl gap-4 md:grid-cols-3">
          <LandingStat
            title="Current Update"
            value={`t=${latest.time}`}
            detail="Latest system snapshot"
          />
          <LandingStat
            title="Intersections Monitored"
            value={String(nodeCount)}
            detail="Live city network nodes"
          />
          <LandingStat
            title="Traffic Visibility"
            value="24/7"
            detail="Clear, live congestion overview"
          />
        </div>
      </section>
    </main>
  )
}

type DashboardPageProps = {
  onBackToLanding: () => void
  environmentHistory: EnvironmentTick[]
}

function DashboardPage({ onBackToLanding, environmentHistory }: DashboardPageProps) {
  const latest = environmentHistory[environmentHistory.length - 1]
  const [cityFrameIndex, setCityFrameIndex] = useState(0)
  const [cityZoom, setCityZoom] = useState(0.5)
  const [cityPan, setCityPan] = useState({ x: 0, y: 0 })
  const [isCityDragging, setIsCityDragging] = useState(false)
  const [isCityFullscreen, setIsCityFullscreen] = useState(false)
  const previousCityFrameMaxRef = useRef(0)
  const cityPanelRef = useRef<HTMLDivElement | null>(null)
  const cityPanStartRef = useRef<{
    pointerX: number
    pointerY: number
    originX: number
    originY: number
  } | null>(null)

  useEffect(() => {
    const handleFullscreenChange = () => {
      const isFullscreen =
        document.fullscreenElement !== null &&
        document.fullscreenElement === cityPanelRef.current
      setIsCityFullscreen(isFullscreen)
    }

    document.addEventListener('fullscreenchange', handleFullscreenChange)
    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange)
    }
  }, [])

  useEffect(() => {
    const nextMax = Math.max(environmentHistory.length - 1, 0)

    setCityFrameIndex((currentIndex) => {
      const previousMax = previousCityFrameMaxRef.current
      const wasAtLatestFrame = currentIndex >= previousMax

      if (wasAtLatestFrame) {
        return nextMax
      }

      return Math.min(currentIndex, nextMax)
    })

    previousCityFrameMaxRef.current = nextMax
  }, [environmentHistory])

  const nodes = useMemo(() => toDerivedNodes(latest), [latest])
  const tickLabels = environmentHistory.map((tick) => `T${tick.time}`)
  const totalQueue = nodes.reduce((sum, node) => sum + node.totalQueue, 0)
  const averageNodeWait =
    Math.round((nodes.reduce((sum, node) => sum + node.avgWait, 0) / nodes.length) * 10) /
    10
  const greenNodes = nodes.filter((node) => node.phase === 1).length
  const redNodes = nodes.length - greenNodes

  const totalQueueSeries = environmentHistory.map((tick) =>
    toDerivedNodes(tick).reduce((sum, node) => sum + node.totalQueue, 0),
  )
  const avgWaitSeries = environmentHistory.map((tick) => {
    const tickNodes = toDerivedNodes(tick)
    const value = tickNodes.reduce((sum, node) => sum + node.avgWait, 0) / tickNodes.length
    return Math.round(value * 10) / 10
  })
  const greenNodeSeries = environmentHistory.map(
    (tick) => toDerivedNodes(tick).filter((node) => node.phase === 1).length,
  )
  const maxQueueSeries = environmentHistory.map((tick) => {
    const tickNodes = toDerivedNodes(tick)
    return Math.max(...tickNodes.map((node) => node.totalQueue))
  })

  const maxQueueNode = nodes.reduce((maxNode, node) =>
    node.totalQueue > maxNode.totalQueue ? node : maxNode,
  )

  const generatedEvents = buildEventsFromNodes(latest.time, nodes)
  const [insightSelection, setInsightSelection] = useState<InsightSelection>({
    kind: 'totalQueue',
  })

  const selectedInsight = useMemo<InsightData>(() => {
    if (insightSelection.kind === 'totalQueue') {
      return {
        title: 'Cars Waiting',
        unit: 'cars',
        labels: tickLabels,
        values: totalQueueSeries,
        description: 'Total queued vehicles across the network by environment tick.',
      }
    }

    if (insightSelection.kind === 'avgWait') {
      return {
        title: 'Average Wait Time',
        unit: 'sec',
        labels: tickLabels,
        values: avgWaitSeries,
        description: 'Average wait across all intersections by environment tick.',
      }
    }

    if (insightSelection.kind === 'greenSignals') {
      return {
        title: 'Green Signals',
        unit: 'lights',
        labels: tickLabels,
        values: greenNodeSeries,
        description: 'Count of intersections on green at each tick.',
      }
    }

    if (insightSelection.kind === 'peakNode') {
      return {
        title: 'Peak Node Queue',
        unit: 'cars',
        labels: tickLabels,
        values: maxQueueSeries,
        description: 'Highest single-node queue observed at each environment tick.',
      }
    }

    const nodeId = insightSelection.nodeId
    const queueSeries = environmentHistory.map((tick) => {
      const tickNode = tick.intersections[nodeId]
      if (!tickNode) {
        return 0
      }
      return tickNode.q1 + tickNode.q2
    })

    return {
      title: `${nodeId.replace('node_', 'Node ')} Queue`,
      unit: 'cars',
      labels: tickLabels,
      values: queueSeries,
      description: `Queue trend for ${nodeId.replace('node_', 'Node ')} across recorded ticks.`,
    }
  }, [
    insightSelection,
    tickLabels,
    totalQueueSeries,
    avgWaitSeries,
    greenNodeSeries,
    maxQueueSeries,
    environmentHistory,
  ])

  const cityFrame = environmentHistory[cityFrameIndex] ?? latest
  const cityLayout = useMemo(() => buildCityLayout(cityFrame), [cityFrame])
  const cityFrameMax = Math.max(environmentHistory.length - 1, 0)

  const toggleCityFullscreen = async () => {
    try {
      if (document.fullscreenElement === cityPanelRef.current) {
        await document.exitFullscreen()
        return
      }

      await cityPanelRef.current?.requestFullscreen()
    } catch {
      return
    }
  }

  const handleCityPanStart = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) {
      return
    }

    cityPanStartRef.current = {
      pointerX: event.clientX,
      pointerY: event.clientY,
      originX: cityPan.x,
      originY: cityPan.y,
    }
    setIsCityDragging(true)
    event.preventDefault()
    event.currentTarget.setPointerCapture(event.pointerId)
  }

  const handleCityPanMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!cityPanStartRef.current) {
      return
    }

    event.preventDefault()
    const deltaX = event.clientX - cityPanStartRef.current.pointerX
    const deltaY = event.clientY - cityPanStartRef.current.pointerY

    setCityPan({
      x: cityPanStartRef.current.originX + deltaX,
      y: cityPanStartRef.current.originY + deltaY,
    })
  }

  const handleCityPanEnd = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId)
    }

    cityPanStartRef.current = null
    setIsCityDragging(false)
  }

  const exportInsightToExcel = (insight: InsightData) => {
    const rows = insight.labels.map((label, index) => ({
      Tick: label,
      Value: insight.values[index],
      Metric: insight.title,
      Unit: insight.unit,
    }))

    const workbook = XLSX.utils.book_new()
    const sheet = XLSX.utils.json_to_sheet(rows)
    XLSX.utils.book_append_sheet(workbook, sheet, 'Insight Data')

    const fileName = `6ix-streets-${insight.title
      .toLowerCase()
      .replace(/\s+/g, '-')}.xlsx`
    XLSX.writeFile(workbook, fileName)
  }

  const exportInsightToPowerBI = (insight: InsightData) => {
    const rows = insight.labels.map((label, index) => ({
      tick: label,
      metric: insight.title,
      value: insight.values[index],
      unit: insight.unit,
    }))

    const worksheet = XLSX.utils.json_to_sheet(rows)
    const csv = XLSX.utils.sheet_to_csv(worksheet)
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)

    const link = document.createElement('a')
    link.href = url
    link.download = `6ix-streets-${insight.title
      .toLowerCase()
      .replace(/\s+/g, '-')}-powerbi.csv`
    link.click()

    URL.revokeObjectURL(url)
  }

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100">
      <div className="fixed inset-0 -z-10 bg-[radial-gradient(circle_at_15%_0%,rgba(56,189,248,0.14),transparent_36%),radial-gradient(circle_at_80%_0%,rgba(129,140,248,0.12),transparent_35%),radial-gradient(circle_at_50%_100%,rgba(96,165,250,0.08),transparent_45%)]" />

      <div className="mx-auto w-full max-w-[1450px] px-4 py-5 sm:px-6 lg:px-8 lg:py-8">
        <header className="rounded-3xl border border-white/10 bg-white/[0.03] p-5 shadow-lg shadow-black/20 sm:p-6">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.22em] text-sky-200/90">
                6ix Streets • Live Operations
              </p>
              <h2 className="mt-2 text-2xl font-semibold text-white sm:text-3xl">
                City Traffic Overview
              </h2>
              <p className="mt-2 text-sm text-slate-300 sm:text-base">
                A simplified view of congestion, wait times, and signal status
                across the network.
              </p>
            </div>

            <button
              type="button"
              onClick={onBackToLanding}
              className="rounded-xl border border-white/20 bg-white/5 px-4 py-2 text-sm font-semibold text-slate-100 transition hover:border-sky-200/40 hover:bg-white/10"
            >
              Back to Landing
            </button>
          </div>
        </header>

        <section className="mt-6 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <MetricTile
            label="Cars Waiting"
            value={String(totalQueue)}
            detail="Vehicles currently queued"
            gradient="from-indigo-300 to-violet-300"
            onClick={() => setInsightSelection({ kind: 'totalQueue' })}
          />
          <MetricTile
            label="Average Wait Time"
            value={`${averageNodeWait}s`}
            detail="Typical delay per intersection"
            gradient="from-sky-300 to-cyan-300"
            onClick={() => setInsightSelection({ kind: 'avgWait' })}
          />
          <MetricTile
            label="Signals Open / Stopped"
            value={`${greenNodes}/${redNodes}`}
            detail="Current light status mix"
            gradient="from-teal-300 to-emerald-300"
            onClick={() => setInsightSelection({ kind: 'greenSignals' })}
          />
          <MetricTile
            label="Most Congested Spot"
            value={maxQueueNode.id.replace('node_', 'Node ')}
            detail={`${maxQueueNode.totalQueue} cars waiting`}
            gradient="from-amber-200 to-orange-300"
            onClick={() => setInsightSelection({ kind: 'peakNode' })}
          />
        </section>

        <section className="mt-6 grid gap-6 xl:grid-cols-[1.6fr_1fr]">
          <article className="rounded-3xl border border-white/10 bg-white/[0.03] p-5 shadow-lg shadow-black/20 sm:p-6">
            <h3 className="text-lg font-semibold text-white sm:text-xl">Selected Insight</h3>
            <p className="text-sm text-slate-400">
              Pick any top metric tile or intersection card to update this chart.
            </p>

            <div className="mt-5 rounded-2xl border border-white/10 bg-slate-950/55 p-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <p className="text-sm text-slate-400">{selectedInsight.description}</p>
                  <p className="mt-1 text-xs text-slate-500">
                    Samples: {selectedInsight.values.length}
                  </p>
                </div>
              </div>

              <div className="mt-4">
                <InsightChart insight={selectedInsight} />
              </div>

              <div className="mt-4 flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={() => exportInsightToExcel(selectedInsight)}
                  className="rounded-xl border border-white/20 bg-white/5 px-4 py-2 text-sm font-semibold text-slate-100 transition hover:bg-white/10"
                >
                  Export to Excel (.xlsx)
                </button>
                <button
                  type="button"
                  onClick={() => exportInsightToPowerBI(selectedInsight)}
                  className="rounded-xl border border-sky-300/40 bg-sky-400/10 px-4 py-2 text-sm font-semibold text-sky-100 transition hover:bg-sky-400/20"
                >
                  Export for Power BI (.csv)
                </button>
              </div>
            </div>
          </article>

          <article className="rounded-3xl border border-white/10 bg-white/[0.03] p-5 shadow-lg shadow-black/20 sm:p-6">
            <h3 className="text-lg font-semibold text-white sm:text-xl">Live Alerts</h3>
            <p className="text-sm text-slate-400">
              Important traffic updates generated from current conditions.
            </p>
            <ul className="mt-4 space-y-3">
              {generatedEvents.map((event) => (
                <li
                  key={`${event.time}-${event.message}`}
                  className="rounded-2xl border border-white/10 bg-slate-950/60 p-3"
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-xs text-slate-400">Update {event.time}</span>
                    <LevelPill level={event.level} />
                  </div>
                  <p className="mt-2 text-sm text-slate-200">{event.message}</p>
                </li>
              ))}
            </ul>
          </article>
        </section>

        <section className="mt-6 rounded-3xl border border-white/10 bg-white/[0.03] p-5 shadow-lg shadow-black/20 sm:p-6">
          <h3 className="text-lg font-semibold text-white sm:text-xl">Intersection Snapshot</h3>
          <p className="text-sm text-slate-400">
            Simple per-location status for quick decisions.
          </p>
          <div className="mt-4 overflow-x-auto pb-2">
            <div className="flex min-w-max gap-3 pr-2">
              {nodes.map((node) => (
                <div key={node.id} className="w-72 shrink-0">
                  <IntersectionCard
                    node={node}
                    onClick={() => setInsightSelection({ kind: 'nodeQueue', nodeId: node.id })}
                  />
                </div>
              ))}
            </div>
          </div>
        </section>

        <section
          ref={cityPanelRef}
          className={`mt-6 rounded-3xl border border-white/10 bg-white/[0.03] p-3.5 shadow-lg shadow-black/20 sm:p-4 ${
            isCityFullscreen ? 'h-full overflow-auto bg-slate-950' : ''
          }`}
        >
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h3 className="text-lg font-semibold text-white sm:text-xl">
                Top-Down City Simulation
              </h3>
              <p className="text-sm text-slate-400">
                Frame-by-frame map of all branches from live simulation data.
              </p>
            </div>

            <div className="flex items-center gap-2">
              <span className="rounded-lg border border-slate-600/60 bg-slate-900/70 px-3 py-2 text-xs text-slate-300">
                Frame {cityFrameIndex + 1}/{cityFrameMax + 1} • t={cityFrame.time}
              </span>
              <button
                type="button"
                onClick={toggleCityFullscreen}
                className="rounded-lg border border-white/20 bg-white/5 px-3 py-2 text-xs font-semibold text-slate-100 transition hover:bg-white/10"
              >
                {isCityFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
              </button>
            </div>
          </div>

          <div className="mt-2.5 rounded-2xl border border-white/10 bg-slate-950/55 p-2">
            <div className="sticky top-0 z-10 mb-2 flex flex-wrap items-center gap-2 rounded-lg border border-white/5 bg-slate-950/65 p-1.5 backdrop-blur-sm">
              <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                Zoom
              </span>
              <button
                type="button"
                onClick={() => setCityZoom((zoom) => Math.max(0.5, Number((zoom - 0.2).toFixed(1))))}
                className="rounded-md border border-white/20 bg-white/5 px-2.5 py-1 text-xs font-semibold text-slate-100 transition hover:bg-white/10"
              >
                -
              </button>
              <input
                type="range"
                min={0.5}
                max={2.6}
                step={0.1}
                value={cityZoom}
                onChange={(event) => setCityZoom(Number(event.target.value))}
                className="w-36 accent-sky-300"
              />
              <button
                type="button"
                onClick={() => setCityZoom((zoom) => Math.min(2.6, Number((zoom + 0.2).toFixed(1))))}
                className="rounded-md border border-white/20 bg-white/5 px-2.5 py-1 text-xs font-semibold text-slate-100 transition hover:bg-white/10"
              >
                +
              </button>
              <span className="rounded-md border border-slate-600/60 bg-slate-900/70 px-2 py-1 text-[11px] text-slate-300">
                {Math.round(cityZoom * 100)}%
              </span>
              <button
                type="button"
                onClick={() => {
                  setCityZoom(0.5)
                  setCityPan({ x: 0, y: 0 })
                }}
                className="rounded-md border border-white/20 bg-white/5 px-2.5 py-1 text-xs font-semibold text-slate-100 transition hover:bg-white/10"
              >
                Reset View
              </button>
              <div className="ml-auto flex items-center gap-3 rounded-md border border-slate-700/70 bg-slate-900/70 px-3 py-1 text-[11px] text-slate-300">
                <span className="inline-flex items-center gap-1.5">
                  <span className="h-2.5 w-2.5 rounded-full bg-emerald-400" /> Green
                </span>
                <span className="inline-flex items-center gap-1.5">
                  <span className="h-2.5 w-2.5 rounded-full bg-yellow-300" /> Yellow
                </span>
                <span className="inline-flex items-center gap-1.5">
                  <span className="h-2.5 w-2.5 rounded-full bg-rose-400" /> Red
                </span>
              </div>
            </div>

            <div
              className={`select-none overflow-hidden rounded-xl border border-white/10 bg-slate-900/70 p-2 ${
                isCityDragging ? 'cursor-grabbing' : 'cursor-grab'
              }`}
              style={{ touchAction: 'none' }}
              onPointerDown={handleCityPanStart}
              onPointerMove={handleCityPanMove}
              onPointerUp={handleCityPanEnd}
              onPointerCancel={handleCityPanEnd}
              onPointerLeave={handleCityPanEnd}
            >
              <svg
                viewBox="0 0 100 100"
                className={`h-full w-full rounded-lg select-none ${isCityFullscreen ? 'h-[36vh]' : 'h-[11rem]'}`}
                style={{
                  transform: `translate(${cityPan.x}px, ${cityPan.y}px) scale(${cityZoom})`,
                  transformOrigin: 'center center',
                  userSelect: 'none',
                }}
              >
              {cityLayout.roads.map((road) => (
                <line
                  key={road.id}
                  x1={road.x1}
                  y1={road.y1}
                  x2={road.x2}
                  y2={road.y2}
                  className="stroke-slate-600"
                  strokeWidth="2.2"
                />
              ))}

              {cityLayout.nodes.map((entry) => {
                const northSouthLight =
                  entry.node.northSouthLight ??
                  (entry.node.phase === 0 ? 'Green' : 'Red')
                const eastWestLight =
                  entry.node.eastWestLight ??
                  (entry.node.phase === 1 ? 'Green' : 'Red')
                const northSouthGlow = roadGlowColor(entry.node.wait1)
                const eastWestGlow = roadGlowColor(entry.node.wait2)

                return (
                  <g key={entry.id}>
                    <line
                      x1={entry.x}
                      y1={entry.y - 6}
                      x2={entry.x}
                      y2={entry.y + 6}
                      stroke={northSouthGlow}
                      strokeWidth="3.6"
                      strokeLinecap="round"
                    />
                    <line
                      x1={entry.x - 6}
                      y1={entry.y}
                      x2={entry.x + 6}
                      y2={entry.y}
                      stroke={eastWestGlow}
                      strokeWidth="3.6"
                      strokeLinecap="round"
                    />

                    <line
                      x1={entry.x}
                      y1={entry.y - 6}
                      x2={entry.x}
                      y2={entry.y + 6}
                      stroke={signalLightColor(northSouthLight)}
                      strokeWidth="2.8"
                      strokeLinecap="round"
                    />
                    <line
                      x1={entry.x - 6}
                      y1={entry.y}
                      x2={entry.x + 6}
                      y2={entry.y}
                      stroke={signalLightColor(eastWestLight)}
                      strokeWidth="2.8"
                      strokeLinecap="round"
                    />

                    <circle
                      cx={entry.x}
                      cy={entry.y}
                      r="1.4"
                      fill="rgba(15, 23, 42, 0.95)"
                      className="stroke-white/35"
                      strokeWidth="0.4"
                    />
                  </g>
                )
              })}
              </svg>
            </div>

            <div className="sticky bottom-0 z-10 mt-3 flex flex-wrap items-center gap-2 rounded-lg border border-white/10 bg-slate-950/80 p-1.5 backdrop-blur-sm">
              <button
                type="button"
                onClick={() => setCityFrameIndex(0)}
                disabled={cityFrameIndex <= 0}
                className="rounded-lg border border-white/20 bg-white/5 px-3 py-2 text-xs font-semibold text-slate-100 transition hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-40"
                aria-label="Jump to first frame"
              >
                {'<<'}
              </button>
              <button
                type="button"
                onClick={() => setCityFrameIndex((index) => Math.max(0, index - 1))}
                disabled={cityFrameIndex <= 0}
                className="rounded-lg border border-white/20 bg-white/5 px-3 py-2 text-xs font-semibold text-slate-100 transition hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-40"
                aria-label="Previous frame"
              >
                {'<'}
              </button>

              <input
                type="range"
                min={0}
                max={cityFrameMax}
                value={Math.min(cityFrameIndex, cityFrameMax)}
                onChange={(event) => {
                  setCityFrameIndex(Number(event.target.value))
                }}
                className="min-w-[220px] flex-1 accent-sky-300"
              />

              <button
                type="button"
                onClick={() => setCityFrameIndex((index) => Math.min(cityFrameMax, index + 1))}
                disabled={cityFrameIndex >= cityFrameMax}
                className="rounded-lg border border-white/20 bg-white/5 px-3 py-2 text-xs font-semibold text-slate-100 transition hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-40"
                aria-label="Next frame"
              >
                {'>'}
              </button>
              <button
                type="button"
                onClick={() => setCityFrameIndex(cityFrameMax)}
                disabled={cityFrameIndex >= cityFrameMax}
                className="rounded-lg border border-white/20 bg-white/5 px-3 py-2 text-xs font-semibold text-slate-100 transition hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-40"
                aria-label="Jump to last frame"
              >
                {'>>'}
              </button>
            </div>
          </div>
        </section>
      </div>
    </main>
  )
}

type CityLayoutNode = {
  id: string
  node: EnvironmentNode
  x: number
  y: number
  row: number
  col: number
}

type CityLayoutRoad = {
  id: string
  x1: number
  y1: number
  x2: number
  y2: number
}

type CityLayout = {
  nodes: CityLayoutNode[]
  roads: CityLayoutRoad[]
}

function buildCityLayout(frame: EnvironmentTick): CityLayout {
  const ids = Object.keys(frame.intersections).sort()
  if (ids.length === 0) {
    return { nodes: [], roads: [] }
  }

  const columns = Math.ceil(Math.sqrt(ids.length))
  const rows = Math.ceil(ids.length / columns)
  const minX = 14
  const maxX = 86
  const minY = 16
  const maxY = 84

  const xStep = columns > 1 ? (maxX - minX) / (columns - 1) : 0
  const yStep = rows > 1 ? (maxY - minY) / (rows - 1) : 0

  const nodes: CityLayoutNode[] = ids.map((id, index) => {
    const col = index % columns
    const row = Math.floor(index / columns)
    return {
      id,
      node: frame.intersections[id],
      col,
      row,
      x: minX + col * xStep,
      y: minY + row * yStep,
    }
  })

  const nodeByGrid = new Map(nodes.map((node) => [`${node.row}:${node.col}`, node]))
  const roads: CityLayoutRoad[] = []

  nodes.forEach((node) => {
    const right = nodeByGrid.get(`${node.row}:${node.col + 1}`)
    if (right) {
      roads.push({
        id: `${node.id}->${right.id}`,
        x1: node.x,
        y1: node.y,
        x2: right.x,
        y2: right.y,
      })
    }

    const down = nodeByGrid.get(`${node.row + 1}:${node.col}`)
    if (down) {
      roads.push({
        id: `${node.id}->${down.id}`,
        x1: node.x,
        y1: node.y,
        x2: down.x,
        y2: down.y,
      })
    }
  })

  return { nodes, roads }
}

function signalLightColor(light: SignalLight): string {
  if (light === 'Green') {
    return '#34d399'
  }

  if (light === 'Yellow') {
    return '#ffea00'
  }

  return '#fb7185'
}

function roadGlowColor(waitSeconds: number): string {
  if (waitSeconds > 180) {
    return 'rgba(248, 113, 113, 0.55)'
  }

  if (waitSeconds > 60) {
    return 'rgba(250, 204, 21, 0.5)'
  }

  return 'rgba(255, 255, 255, 0.28)'
}

function toDerivedNodes(tick: EnvironmentTick): DerivedNode[] {
  return Object.entries(tick.intersections)
    .sort(([a], [b]) => {
      const aMatch = a.match(/\d+/)
      const bMatch = b.match(/\d+/)

      if (aMatch && bMatch) {
        return Number(aMatch[0]) - Number(bMatch[0])
      }

      return a.localeCompare(b)
    })
    .map(([id, node]) => {
      const totalQueue = node.q1 + node.q2
      const avgWait = (node.wait1 + node.wait2) / 2

      return {
        id,
        phase: node.phase,
        q1: node.q1,
        q2: node.q2,
        totalQueue,
        wait1: node.wait1,
        wait2: node.wait2,
        avgWait,
      }
    })
}

function buildEventsFromNodes(time: number, nodes: DerivedNode[]): IncidentEvent[] {
  const events: IncidentEvent[] = []

  nodes.forEach((node) => {
    if (node.totalQueue >= 6) {
      events.push({
        time: String(time),
        level: 'warning',
        message: `${node.id.replace('node_', 'Node ')} is building congestion (${node.totalQueue} cars waiting).`,
      })
    }

    if (node.avgWait >= 8) {
      events.push({
        time: String(time),
        level: 'critical',
        message: `${node.id.replace('node_', 'Node ')} has elevated delay (${node.avgWait.toFixed(1)}s average wait).`,
      })
    }

    if (node.phase === 1 && node.totalQueue <= 3) {
      events.push({
        time: String(time),
        level: 'info',
        message: `${node.id.replace('node_', 'Node ')} is flowing smoothly right now.`,
      })
    }
  })

  if (events.length === 0) {
    events.push({
      time: String(time),
      level: 'info',
      message: 'All nodes stable for current tick.',
    })
  }

  return events.slice(0, 5)
}

type LandingStatProps = {
  title: string
  value: string
  detail: string
}

function LandingStat({ title, value, detail }: LandingStatProps) {
  return (
    <article className="rounded-2xl border border-white/10 bg-white/[0.03] p-4 text-center shadow-lg shadow-black/20">
      <p className="text-xs uppercase tracking-wide text-slate-400">{title}</p>
      <p className="mt-2 text-2xl font-semibold text-white">{value}</p>
      <p className="mt-1 text-xs text-slate-300">{detail}</p>
    </article>
  )
}

type MetricTileProps = {
  label: string
  value: string
  detail: string
  gradient: string
  onClick?: () => void
}

function MetricTile({ label, value, detail, gradient, onClick }: MetricTileProps) {
  return (
    <article
      onClick={onClick}
      className="cursor-pointer rounded-2xl border border-white/10 bg-white/[0.03] p-4 shadow-md shadow-black/20 transition hover:border-sky-200/30 hover:bg-white/[0.04]"
    >
      <div className={`h-1 rounded-full bg-gradient-to-r ${gradient}`} />
      <p className="mt-3 text-xs uppercase tracking-wide text-slate-400">{label}</p>
      <p className="mt-2 text-2xl font-semibold text-white">{value}</p>
      <p className="mt-1 text-xs text-slate-300">{detail}</p>
    </article>
  )
}

type InsightChartProps = {
  insight: InsightData
}

function InsightChart({ insight }: InsightChartProps) {
  const sampledInsight = downsampleInsight(insight, 120)

  const maxValue = Math.max(...sampledInsight.values, 1)
  const minValue = Math.min(...sampledInsight.values)
  const range = Math.max(maxValue - minValue, 1)

  const points = sampledInsight.values
    .map((value, index) => {
      const x = (index / Math.max(sampledInsight.values.length - 1, 1)) * 100
      const normalized = (value - minValue) / range
      const y = 100 - normalized * 100
      return `${x},${y}`
    })
    .join(' ')

  return (
    <div className="rounded-2xl border border-white/10 bg-slate-950/55 p-4">
      <p className="text-sm font-medium text-slate-200">{insight.title}</p>
      <div className="mt-3 rounded-xl border border-white/10 bg-slate-900/60 p-3">
        <svg viewBox="0 0 100 100" className="h-52 w-full">
          <polyline
            fill="none"
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
            points={points}
            className="stroke-sky-300"
          />
          {sampledInsight.values.length <= 45
            ? sampledInsight.values.map((value, index) => {
                const x =
                  (index / Math.max(sampledInsight.values.length - 1, 1)) * 100
                const normalized = (value - minValue) / range
                const y = 100 - normalized * 100
                return (
                  <circle
                    key={`${insight.title}-pt-${index}`}
                    cx={x}
                    cy={y}
                    r="1.8"
                    className="fill-sky-200"
                  />
                )
              })
            : null}
        </svg>
      </div>
      <p className="mt-2 text-xs text-slate-400">
        Range: {minValue}–{maxValue} {insight.unit} • Showing {sampledInsight.values.length} of {insight.values.length}
      </p>
    </div>
  )
}

function downsampleInsight(insight: InsightData, maxPoints: number): InsightData {
  if (insight.values.length <= maxPoints || maxPoints <= 0) {
    return insight
  }

  const lastIndex = insight.values.length - 1
  const stride = lastIndex / (maxPoints - 1)
  const labels: string[] = []
  const values: number[] = []

  for (let index = 0; index < maxPoints; index += 1) {
    const sourceIndex = Math.round(index * stride)
    const clampedIndex = Math.min(sourceIndex, lastIndex)
    labels.push(insight.labels[clampedIndex])
    values.push(insight.values[clampedIndex])
  }

  return {
    ...insight,
    labels,
    values,
  }
}

type LevelPillProps = {
  level: IncidentEvent['level']
}

function LevelPill({ level }: LevelPillProps) {
  const styleByLevel: Record<IncidentEvent['level'], string> = {
    info: 'border-sky-400/30 bg-sky-400/10 text-sky-200',
    warning: 'border-amber-300/30 bg-amber-300/10 text-amber-200',
    critical: 'border-rose-300/30 bg-rose-300/10 text-rose-200',
  }

  return (
    <span
      className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${styleByLevel[level]}`}
    >
      {level}
    </span>
  )
}

type IntersectionCardProps = {
  node: DerivedNode
  onClick?: () => void
}

function IntersectionCard({ node, onClick }: IntersectionCardProps) {
  const nodeName = node.id.replace('node_', 'Node ')

  return (
    <article
      onClick={onClick}
      className="cursor-pointer rounded-2xl border border-white/10 bg-slate-950/55 p-3 transition hover:border-sky-200/30 hover:bg-slate-900/75"
    >
      <div className="flex items-center justify-between gap-2">
        <p className="text-sm font-semibold text-slate-100">{nodeName}</p>
        <span
          className={`rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${
            node.phase === 1
              ? 'bg-emerald-300/20 text-emerald-200'
              : 'bg-rose-300/20 text-rose-200'
          }`}
        >
          {node.phase === 1 ? 'Green' : 'Red'}
        </span>
      </div>

      <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
        <NodeStat label="Cars Waiting" value={String(node.totalQueue)} />
        <NodeStat label="Average Delay" value={`${node.avgWait.toFixed(1)}s`} />
        <NodeStat label="Lane A Queue" value={String(node.q1)} />
        <NodeStat label="Lane B Queue" value={String(node.q2)} />
      </div>
    </article>
  )
}

type NodeStatProps = {
  label: string
  value: string
}

function NodeStat({ label, value }: NodeStatProps) {
  return (
    <div className="rounded-lg border border-white/10 bg-slate-900/65 px-2 py-2">
      <p className="text-[10px] uppercase tracking-wide text-slate-500">{label}</p>
      <p className="mt-1 text-xs text-slate-100">{value}</p>
    </div>
  )
}

export default App
