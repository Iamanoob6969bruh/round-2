import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts'
import { MetricCard, Expander } from '../Common.jsx'

const PIE_COLORS = ['#4f46e5', '#0891b2', '#dc2626', '#d97706', '#059669']

export default function MetricsStage({ metrics, perClass, perf }) {
  const pieData = Object.entries(perf.stages_ms).map(([name, value]) => ({ name, value }))

  return (
    <>
      <h3 className="section-header">📈 Detection Metrics</h3>
      <div className="metric-grid" style={{ gridTemplateColumns: 'repeat(4, 1fr)' }}>
        <MetricCard icon="🎯" value={`${(metrics.avg_detection_conf * 100).toFixed(0)}%`} label="Avg Detection Conf." />
        <MetricCard icon="✅" value={`${metrics.high_conf}/${metrics.total_detections}`} label="High Conf. Detections" />
        <MetricCard icon="⚠️" value={`${(metrics.avg_violation_conf * 100).toFixed(0)}%`} label="Avg Violation Conf." />
        <MetricCard icon="🔢" value={`${metrics.plates_found}/${metrics.violation_count}`} label="Plates Recognized" />
      </div>

      <Expander title="Per-Class Detection Breakdown">
        {perClass.length > 0 ? (
          <div className="table-wrap" style={{ maxHeight: 'none' }}>
            <table>
              <thead><tr><th>Class</th><th>Count</th><th>Avg Conf</th><th>Min</th><th>Max</th></tr></thead>
              <tbody>
                {perClass.map((c) => <tr key={c.cls}><td>{c.cls}</td><td>{c.count}</td><td>{c.avg}</td><td>{c.min}</td><td>{c.max}</td></tr>)}
              </tbody>
            </table>
          </div>
        ) : <p className="caption">No vehicle classes detected.</p>}
      </Expander>

      <Expander title="Computational Efficiency">
        <table className="md-table">
          <tbody>
            <tr><td>Total Latency</td><td>{perf.total_ms.toFixed(0)} ms</td></tr>
            <tr><td>Throughput</td><td>{perf.fps.toFixed(1)} FPS</td></tr>
            <tr><td>Resolution</td><td>{perf.width} × {perf.height} px</td></tr>
            <tr><td>Pixels Processed</td><td>{perf.pixels.toLocaleString()}</td></tr>
            <tr><td>Processing Speed</td><td>{perf.px_per_ms} px/ms</td></tr>
          </tbody>
        </table>
        {pieData.length > 0 && (
          <div style={{ height: 240, marginTop: '1rem' }}>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={60} outerRadius={90} paddingAngle={2} label={(e) => e.name} labelLine={false}>
                  {pieData.map((_, i) => <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />)}
                </Pie>
                <Tooltip contentStyle={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: 8, fontSize: '0.8rem' }} formatter={(v) => `${v} ms`} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}
      </Expander>
    </>
  )
}
