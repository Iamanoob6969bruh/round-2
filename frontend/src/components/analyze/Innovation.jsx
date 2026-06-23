const ROUTE_META = {
  'auto-issue':   { cls: 'route-auto-issue',   label: 'AUTO-ISSUE' },
  'human-review': { cls: 'route-human-review', label: 'HUMAN REVIEW' },
  'discard':      { cls: 'route-discard',      label: 'DISCARD' },
}

function edsColor(score) {
  if (score >= 75) return '#059669'
  if (score >= 45) return '#d97706'
  return '#94a3b8'
}

export function EdsGauge({ score }) {
  const R = 42, C = 2 * Math.PI * R
  const pct = Math.max(0, Math.min(score, 100)) / 100
  return (
    <div className="eds-gauge">
      <svg width="100" height="100" viewBox="0 0 100 100">
        <circle cx="50" cy="50" r={R} fill="none" stroke="#e2e8f0" strokeWidth="8" />
        <circle cx="50" cy="50" r={R} fill="none" stroke={edsColor(score)} strokeWidth="8"
          strokeLinecap="round" strokeDasharray={C} strokeDashoffset={C * (1 - pct)}
          style={{ transition: 'stroke-dashoffset 0.6s ease', transform: 'rotate(-90deg)', transformOrigin: '50% 50%' }} />
      </svg>
      <div className="eds-num">
        <span className="eds-val" style={{ color: edsColor(score) }}>{score.toFixed(0)}</span>
        <span className="eds-cap">EDS</span>
      </div>
    </div>
  )
}

export function RoutingBadge({ routing }) {
  const m = ROUTE_META[routing] || ROUTE_META['discard']
  return <span className={`routing-badge ${m.cls}`}><span className="dot" /> {m.label}</span>
}

export function FactorBars({ factors }) {
  return (
    <div>
      {Object.entries(factors).map(([label, val]) => (
        <div className="factor-row" key={label}>
          <span className="factor-label">{label}</span>
          <span className="factor-track"><span className="factor-fill" style={{ width: `${val}%` }} /></span>
          <span className="factor-val">{val.toFixed(0)}</span>
        </div>
      ))}
    </div>
  )
}

export function EdsPanel({ v }) {
  return (
    <div className="eds-card">
      <EdsGauge score={v.eds} />
      <div className="eds-body">
        <RoutingBadge routing={v.routing} />
        <div className="eds-rationale">{v.eds_rationale}</div>
        <div style={{ marginTop: '0.6rem' }}><FactorBars factors={v.eds_factors} /></div>
      </div>
    </div>
  )
}

export function ExplanationPanel({ explanation }) {
  if (!explanation) return null
  return (
    <div className="explain-panel">
      <div className="xai-verdict"><span>🧠</span><span>{explanation.verdict}</span></div>
      <ul>{explanation.reasoning.map((r, i) => <li key={i}>{r}</li>)}</ul>
      <div className="explain-meta">
        <span>Method: <b>{explanation.method}</b></span>
        <span>Region: <code>{explanation.evidence_bbox}</code></span>
      </div>
    </div>
  )
}

export function TriageSummary({ triage, methodology }) {
  if (!triage || triage.total === 0) return null
  const c = triage.counts
  return (
    <div>
      <div className="triage-grid">
        <div className="triage-card t-auto"><div className="tc-count">{c['auto-issue']}</div><div className="tc-label">Auto-Issue</div></div>
        <div className="triage-card t-review"><div className="tc-count">{c['human-review']}</div><div className="tc-label">Human Review</div></div>
        <div className="triage-card t-discard"><div className="tc-count">{c['discard']}</div><div className="tc-label">Discarded</div></div>
      </div>
      <div className="effort-banner">Only <b>{triage.human_effort_pct}%</b> of cases require human attention — the rest are auto-processed.</div>
      {methodology && <p className="caption" style={{ marginTop: '0.25rem' }}>EDS: {methodology.model.toLowerCase()} {methodology.calibration_note}</p>}
    </div>
  )
}

export function CalibrationBadge({ calibration }) {
  if (!calibration) return null
  return (
    <div>
      <div className="calib-badge"><span>🛰️</span> {calibration.headline}</div>
      {calibration.notes?.length > 0 && <ul className="calib-notes">{calibration.notes.map((n, i) => <li key={i}>• {n}</li>)}</ul>}
    </div>
  )
}

export function RiskChip({ risk }) {
  return <span className="risk-chip">⚠ {risk.toFixed(0)}</span>
}
