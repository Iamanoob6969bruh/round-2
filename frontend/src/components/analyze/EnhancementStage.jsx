import { MetricCard, Expander, JsonView } from '../Common.jsx'
import { CalibrationBadge } from './Innovation.jsx'

export default function EnhancementStage({ data, calibration }) {
  return (
    <>
      <h3 className="section-header">⚙️ Feedback-Loop Enhancement</h3>
      <CalibrationBadge calibration={calibration} />
      <div className="metric-grid" style={{ gridTemplateColumns: 'repeat(3, 1fr)' }}>
        <MetricCard icon="🔄" value={data.iterations} label="Iterations" />
        <MetricCard icon={data.converged ? '✅' : '⚠️'} value={data.converged ? 'Yes' : 'Partial'} label="Converged" />
        <MetricCard icon="📐" value={data.resolution} label="Resolution" />
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginTop: '1rem' }}>
        <div><div className="caption">Initial Metrics</div><JsonView data={data.initial} /></div>
        <div><div className="caption">Final Metrics</div><JsonView data={data.final} /></div>
      </div>
      {data.history?.length > 0 && (
        <div style={{ marginTop: '0.75rem' }}>
          <Expander title="View feedback loop iterations">
            {data.history.map((h, idx) => (
              <div className="loop-step" key={idx}>
                <div className="step-num">{idx + 1}</div>
                <span className="step-action">{h.action}</span>
                <span className="step-metrics">B:{h.metrics.brightness} C:{h.metrics.contrast} S:{h.metrics.sharpness}</span>
              </div>
            ))}
          </Expander>
        </div>
      )}
    </>
  )
}
