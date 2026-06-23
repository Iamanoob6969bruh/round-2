function titleCase(s) {
  return s.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}

export default function PerformanceStage({ perf }) {
  return (
    <>
      <h3 className="section-header">⏱️ Performance</h3>
      <div className="perf-row">
        {Object.entries(perf.stages_ms).map(([stage, ms]) => (
          <span className="perf-pill" key={stage}>{titleCase(stage)} <span className="perf-value">{ms.toFixed(0)}ms</span></span>
        ))}
        <span className="perf-pill total"><strong>Total</strong> <span className="perf-value">{perf.total_ms.toFixed(0)}ms</span></span>
      </div>
      <p className="caption">Throughput: <strong>{perf.fps.toFixed(1)} FPS</strong> · Resolution: {perf.width}×{perf.height}px</p>
    </>
  )
}
