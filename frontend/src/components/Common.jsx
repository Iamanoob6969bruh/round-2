import { useState } from 'react'

export function MetricCard({ icon, value, label }) {
  return (
    <div className="metric-card">
      <div className="metric-icon">{icon}</div>
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
    </div>
  )
}

export function SeverityBadge({ severity }) {
  const cls = { high: 'severity-high', medium: 'severity-medium', low: 'severity-low' }[severity] || 'severity-medium'
  return <span className={`severity-badge ${cls}`}>{severity}</span>
}

export function Expander({ title, children, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="expander">
      <div className="expander-head" onClick={() => setOpen(!open)}>
        <span>{title}</span>
        <span className={`expander-arrow ${open ? 'open' : ''}`}>▶</span>
      </div>
      {open && <div className="expander-body">{children}</div>}
    </div>
  )
}

export function JsonView({ data }) {
  return <pre className="json-view">{JSON.stringify(data, null, 2)}</pre>
}

export function Spinner({ text }) {
  return (
    <div className="spinner-wrap">
      <div className="spinner" />
      <div className="spinner-text">{text}</div>
    </div>
  )
}

export function Tabs({ tabs, active, onChange }) {
  return (
    <div className="tabs">
      {tabs.map((t) => (
        <button
          key={t.id}
          className={`tab ${active === t.id ? 'active' : ''}`}
          onClick={() => onChange(t.id)}
        >
          {t.label}
        </button>
      ))}
    </div>
  )
}
