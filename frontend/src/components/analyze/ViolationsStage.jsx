import { SeverityBadge, Expander, JsonView } from '../Common.jsx'
import { EdsPanel, ExplanationPanel, TriageSummary, RiskChip } from './Innovation.jsx'
import { BASE } from '../../api.js'

export default function ViolationsStage({ violations, savedCount, triage, methodology }) {
  return (
    <>
      <h3 className="section-header">⚠️ Violations & Plate Recognition</h3>

      {violations.length > 0 ? (
        <>
          <TriageSummary triage={triage} methodology={methodology} />

          <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', margin: '1rem 0 0.75rem' }}>
            {violations.length} violation{violations.length > 1 ? 's' : ''} detected — ordered by risk
          </p>

          {violations.map((v, i) => (
            <div key={v.index}>
              <div className="violation-card">
                <div className="violation-head">
                  <span className="violation-title">{v.title}</span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
                    <RiskChip risk={v.risk} />
                    <SeverityBadge severity={v.severity} />
                    <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                      {(v.confidence * 100).toFixed(0)}%{v.plate_text && <> · <strong>{v.plate_text}</strong></>}
                    </span>
                    {v.violation_id && (
                      <a href={`${BASE}/evidence/pdf/${v.violation_id}`} target="_blank" rel="noopener noreferrer"
                         style={{ fontSize: '0.75rem', background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: '6px', padding: '2px 8px', textDecoration: 'none', color: 'var(--text-primary)', whiteSpace: 'nowrap' }}>
                        📄 PDF
                      </a>
                    )}
                  </div>
                </div>
              </div>
              <EdsPanel v={v} />
              <ExplanationPanel explanation={v.explanation} />
              <Expander title={`Evidence & registration`}>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                  <div>
                    <div className="caption">Flagged Vehicle</div>
                    {v.context_crop ? <img src={v.context_crop} alt="vehicle" style={{ width: '100%', borderRadius: '8px' }} /> : <span className="caption">N/A</span>}
                  </div>
                  <div>
                    <div className="caption">Number Plate</div>
                    {v.plate_crop ? <img src={v.plate_crop} alt="plate" style={{ width: '100%', borderRadius: '8px' }} /> : <div className="success-inline" style={{ background: '#f0fdfa', borderColor: '#99f6e4', color: '#0891b2' }}>Plate not detected</div>}
                  </div>
                </div>
                {v.plate_text && (
                  <>
                    <div className="success-inline"><strong>Plate:</strong> {v.plate_text} · OCR confidence: {(v.plate_confidence * 100).toFixed(0)}%</div>
                    <JsonView data={v.registration} />
                  </>
                )}
              </Expander>
            </div>
          ))}
        </>
      ) : (
        <div className="success-card">
          <strong style={{ color: '#059669' }}>✅ No violations detected</strong>
          <p style={{ color: 'var(--text-secondary)', margin: '0.3rem 0 0', fontSize: '0.875rem' }}>All vehicles appear compliant.</p>
        </div>
      )}

      {savedCount > 0 && (
        <p className="caption" style={{ marginTop: '0.5rem' }}>💾 {savedCount} violation(s) saved to database.</p>
      )}
    </>
  )
}
