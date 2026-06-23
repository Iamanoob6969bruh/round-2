import { MetricCard } from '../Common.jsx'

export default function DetectionStage({ data }) {
  const typesStr = Object.entries(data.types).map(([k, v]) => `${k}: ${v}`).join(', ') || '—'
  return (
    <>
      <h3 className="section-header">🚗 Vehicle Detection</h3>
      <div className="metric-grid" style={{ gridTemplateColumns: 'repeat(3, 1fr)' }}>
        <MetricCard icon="🚗" value={data.vehicles} label="Vehicles" />
        <MetricCard icon="🧑" value={data.persons} label="Persons" />
        <MetricCard icon="🏷️" value={typesStr} label="Classification" />
      </div>
    </>
  )
}
