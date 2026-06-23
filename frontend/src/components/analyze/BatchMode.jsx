import { useState, useRef } from 'react'
import { analyzeBatch } from '../../api.js'
import { Spinner, MetricCard } from '../Common.jsx'

const ROUTE_CLS = { 'auto-issue': 'route-auto-issue', 'human-review': 'route-human-review', 'discard': 'route-discard' }

export default function BatchMode() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const fileRef = useRef(null)

  async function handleFiles(fileList) {
    const files = Array.from(fileList || [])
    if (!files.length) return
    setLoading(true); setError(null); setResult(null)
    try { setResult(await analyzeBatch(files)) }
    catch (e) { setError(e.message) }
    finally { setLoading(false) }
  }

  return (
    <div>
      <div className="upload-zone" onClick={() => fileRef.current?.click()}>
        <div className="upload-icon">🗂️</div>
        <div className="upload-text">Select multiple traffic images for batch triage</div>
        <div className="upload-hint">EDS auto-routes every case</div>
        <input ref={fileRef} type="file" multiple accept=".jpg,.jpeg,.png,.bmp" style={{ display: 'none' }} onChange={(e) => handleFiles(e.target.files)} />
      </div>

      {loading && <Spinner text="Processing batch..." />}
      {error && <div className="error-box"><strong>Error:</strong><pre>{error}</pre></div>}

      {result && !loading && (
        <div>
          <div className="metric-grid" style={{ gridTemplateColumns: 'repeat(4, 1fr)' }}>
            <MetricCard icon="🖼️" value={result.images_processed} label="Images Processed" />
            <MetricCard icon="🚨" value={result.total_violations} label="Violations Found" />
            <MetricCard icon="✓" value={`${result.auto_issue_pct}%`} label="Auto-Issued" />
            <MetricCard icon="⚖️" value={`${result.human_effort_pct}%`} label="Need Review" />
          </div>

          <div className="effort-banner">
            Auto-cleared <b>{result.routing_counts['auto-issue']}</b> cases, discarded <b>{result.routing_counts['discard']}</b> weak-evidence, leaving <b>{result.routing_counts['human-review']}</b> for human review.
          </div>

          <h3 className="section-header">Batch Results</h3>
          <div className="table-wrap" style={{ maxHeight: 'none' }}>
            <table>
              <thead><tr><th>Preview</th><th>File</th><th>Violations</th><th>Routing</th><th>Risk</th></tr></thead>
              <tbody>
                {result.items.map((it, i) => (
                  <tr key={i}>
                    <td>{it.thumbnail ? <img className="batch-thumb" src={it.thumbnail} alt="" /> : '—'}</td>
                    <td>{it.filename}</td>
                    <td>
                      <div className="batch-row-violations">
                        {it.violations.length === 0 ? <span style={{ color: 'var(--text-muted)' }}>none</span>
                          : it.violations.map((v, j) => <span className="batch-mini" key={j}>{v.title} · EDS {v.eds.toFixed(0)}</span>)}
                      </div>
                    </td>
                    <td>
                      {it.violations.length === 0 ? '—' :
                        [...new Set(it.violations.map(v => v.routing))].map(r =>
                          <span key={r} className={`routing-badge ${ROUTE_CLS[r]}`} style={{ fontSize: '0.65rem', padding: '0.15rem 0.5rem', marginRight: 4 }}><span className="dot" />{r}</span>)}
                    </td>
                    <td><strong>{it.top_risk.toFixed(0)}</strong></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
