import { useState, useEffect } from 'react'
import { getEvaluation } from '../api.js'
import { MetricCard, Spinner } from '../components/Common.jsx'

export default function Evaluation() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)

  function refresh() {
    setLoading(true)
    getEvaluation().then(setData).catch(() => setData(null)).finally(() => setLoading(false))
  }

  useEffect(() => { refresh() }, [])

  if (loading) return <div><div className="page-header"><h1>Evaluation</h1></div><Spinner text="Loading metrics..." /></div>

  const op = data?.operational || {}
  const cs = data?.confidence_stats || {}
  const bm = data?.benchmark
  const routing = op.routing || { 'auto-issue': 0, 'human-review': 0, 'discard': 0 }

  return (
    <div>
      <div className="page-header">
        <h1>Performance Evaluation</h1>
        <p>Accuracy benchmarks on labelled data + live operational metrics.</p>
      </div>

      <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: '1rem' }}>
        <button onClick={refresh} className="routing-badge route-auto-issue" style={{ cursor: 'pointer', border: 'none' }}>
          <span className="dot" /> Refresh
        </button>
      </div>

      {/* ═══ BENCHMARK (from labelled evaluation) ═══ */}
      {bm && <BenchmarkSection bm={bm} />}

      {/* ═══ OPERATIONAL (live session) ═══ */}
      {op.images_analysed > 0 && (
        <>
          <h3 className="section-header">Live Operational Metrics (this session)</h3>
          <div className="metric-grid" style={{ gridTemplateColumns: 'repeat(4, 1fr)' }}>
            <MetricCard icon="🖼️" value={op.images_analysed} label="Images Analysed" />
            <MetricCard icon="⏱️" value={`${op.avg_latency_ms}ms`} label="Avg Latency" />
            <MetricCard icon="🚀" value={op.throughput_fps} label="Throughput (FPS)" />
            <MetricCard icon="⚖️" value={`${op.human_effort_pct}%`} label="Need Human Review" />
          </div>

          <h3 className="section-header">Detection & Confidence</h3>
          <div className="metric-grid" style={{ gridTemplateColumns: 'repeat(4, 1fr)' }}>
            <MetricCard icon="🔍" value={`${(cs.detection_mean * 100).toFixed(0)}%`} label="Avg Detection Conf." />
            <MetricCard icon="✅" value={`${cs.detection_high_pct}%`} label="High-Conf (≥70%)" />
            <MetricCard icon="⚠️" value={`${(cs.violation_mean * 100).toFixed(0)}%`} label="Avg Violation Conf." />
            <MetricCard icon="📋" value={cs.avg_detections_per_image} label="Avg Objects / Image" />
          </div>

          <h3 className="section-header">EDS Routing</h3>
          <div className="triage-grid">
            <div className="triage-card t-auto"><div className="tc-count">{routing['auto-issue']}</div><div className="tc-label">Auto-Issued (EDS ≥75)</div></div>
            <div className="triage-card t-review"><div className="tc-count">{routing['human-review']}</div><div className="tc-label">Human Review (45–75)</div></div>
            <div className="triage-card t-discard"><div className="tc-count">{routing['discard']}</div><div className="tc-label">Discarded (&lt;45)</div></div>
          </div>
        </>
      )}

      {!op.images_analysed && !bm && (
        <div className="card" style={{ textAlign: 'center', padding: '2.5rem' }}>
          <p style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>📊</p>
          <p style={{ color: 'var(--text-secondary)' }}>No runs yet. Analyze some images first, or run the benchmark harness.</p>
        </div>
      )}
    </div>
  )
}

function BenchmarkSection({ bm }) {
  const det = bm.detection || {}
  const viol = bm.violations || {}
  const plate = bm.plate_recognition || {}
  const mAP = bm.mAP || {}
  const perf = bm.performance || {}
  const perClass = bm.detection_per_class || {}

  return (
    <>
      <h3 className="section-header">Accuracy Benchmark (Labelled Validation Set)</h3>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.82rem', marginBottom: '1rem' }}>
        Evaluated on {bm.images_evaluated} labelled images with human-verified ground truth.
      </p>

      {/* Top-level accuracy cards */}
      <div className="metric-grid" style={{ gridTemplateColumns: 'repeat(4, 1fr)' }}>
        <MetricCard icon="🎯" value={`${(det.f1 * 100).toFixed(1)}%`} label="Detection F1" />
        <MetricCard icon="📐" value={`${(mAP.mAP * 100).toFixed(1)}%`} label="mAP@0.5" />
        <MetricCard icon="⚠️" value={`${(viol.f1 * 100).toFixed(1)}%`} label="Violation F1" />
        <MetricCard icon="🔢" value={`${(plate.exact_match_rate * 100).toFixed(1)}%`} label="Plate Exact Match" />
      </div>

      {/* Detection detail table */}
      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <h4 style={{ fontSize: '0.9rem', fontWeight: 600, marginBottom: '0.75rem' }}>Object Detection (IoU ≥ 0.5)</h4>
        <div className="table-wrap" style={{ border: 'none', boxShadow: 'none' }}>
          <table>
            <thead><tr><th>Metric</th><th>Score</th></tr></thead>
            <tbody>
              <tr><td>Precision</td><td><b>{(det.precision * 100).toFixed(1)}%</b></td></tr>
              <tr><td>Recall</td><td><b>{(det.recall * 100).toFixed(1)}%</b></td></tr>
              <tr><td>F1-score</td><td><b>{(det.f1 * 100).toFixed(1)}%</b></td></tr>
              <tr><td>mAP@0.5</td><td><b>{(mAP.mAP * 100).toFixed(1)}%</b></td></tr>
              <tr><td>True Positives</td><td>{det.tp}</td></tr>
              <tr><td>False Negatives</td><td>{det.fn}</td></tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Per-class breakdown */}
      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <h4 style={{ fontSize: '0.9rem', fontWeight: 600, marginBottom: '0.75rem' }}>Per-Class Detection</h4>
        <div className="table-wrap" style={{ border: 'none', boxShadow: 'none' }}>
          <table>
            <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>TP</th><th>FN</th></tr></thead>
            <tbody>
              {Object.entries(perClass).map(([cls, m]) => (
                <tr key={cls}>
                  <td style={{ fontWeight: 500 }}>{cls}</td>
                  <td>{(m.precision * 100).toFixed(0)}%</td>
                  <td>{(m.recall * 100).toFixed(0)}%</td>
                  <td><b>{(m.f1 * 100).toFixed(1)}%</b></td>
                  <td>{m.tp}</td>
                  <td>{m.fn}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Violation classification */}
      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <h4 style={{ fontSize: '0.9rem', fontWeight: 600, marginBottom: '0.75rem' }}>Violation Classification (IoU ≥ 0.3)</h4>
        <div className="table-wrap" style={{ border: 'none', boxShadow: 'none' }}>
          <table>
            <thead><tr><th>Metric</th><th>Score</th></tr></thead>
            <tbody>
              <tr><td>Precision</td><td><b>{(viol.precision * 100).toFixed(1)}%</b></td></tr>
              <tr><td>Recall</td><td><b>{(viol.recall * 100).toFixed(1)}%</b></td></tr>
              <tr><td>F1-score</td><td><b>{(viol.f1 * 100).toFixed(1)}%</b></td></tr>
              <tr><td>True Positives</td><td>{viol.tp}</td></tr>
              <tr><td>False Positives</td><td>{viol.fp}</td></tr>
              <tr><td>False Negatives</td><td>{viol.fn}</td></tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Plate recognition */}
      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <h4 style={{ fontSize: '0.9rem', fontWeight: 600, marginBottom: '0.75rem' }}>License Plate Recognition</h4>
        <div className="table-wrap" style={{ border: 'none', boxShadow: 'none' }}>
          <table>
            <thead><tr><th>Metric</th><th>Score</th></tr></thead>
            <tbody>
              <tr><td>Exact Match Rate</td><td><b>{(plate.exact_match_rate * 100).toFixed(1)}%</b></td></tr>
              <tr><td>Partial Match Rate</td><td><b>{(plate.partial_match_rate * 100).toFixed(1)}%</b></td></tr>
              <tr><td>Exact Matches</td><td>{plate.exact_matches} / {plate.total_ground_truth}</td></tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Throughput */}
      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <h4 style={{ fontSize: '0.9rem', fontWeight: 600, marginBottom: '0.75rem' }}>Computational Efficiency & Scalability</h4>
        <div className="table-wrap" style={{ border: 'none', boxShadow: 'none' }}>
          <table>
            <thead><tr><th>Metric</th><th>CPU (measured)</th><th>GPU (projected)</th></tr></thead>
            <tbody>
              <tr><td>Latency per image</td><td>266 ms</td><td>~33 ms</td></tr>
              <tr><td>Throughput</td><td>3.76 FPS</td><td>~30 FPS</td></tr>
              <tr><td>4-GPU cluster</td><td>—</td><td>~120 FPS (432K images/hr)</td></tr>
            </tbody>
          </table>
        </div>
        <ul className="calib-notes" style={{ marginTop: '0.5rem' }}>
          <li>• Measured on typical CCTV resolution (379×523). Stateless design scales linearly with workers.</li>
          <li>• A city fleet (2,000 cameras × 1 frame/min = 120K images/hr) is handled by a 4-GPU deployment.</li>
          <li>• Batch endpoint (<code>/api/analyze_batch</code>) processes bulk uploads with triage ordering.</li>
        </ul>
      </div>

      {/* Per-violation-type breakdown */}
      {bm.violations_per_type && Object.keys(bm.violations_per_type).length > 0 && (
        <div className="card" style={{ marginBottom: '1.5rem' }}>
          <h4 style={{ fontSize: '0.9rem', fontWeight: 600, marginBottom: '0.75rem' }}>Per Violation Type</h4>
          <div className="table-wrap" style={{ border: 'none', boxShadow: 'none' }}>
            <table>
              <thead><tr><th>Type</th><th>Precision</th><th>Recall</th><th>F1</th><th>TP</th><th>FP</th><th>FN</th></tr></thead>
              <tbody>
                {Object.entries(bm.violations_per_type).map(([t, m]) => (
                  <tr key={t}>
                    <td style={{ fontWeight: 500 }}>{t.replace(/_/g, ' ')}</td>
                    <td>{(m.precision * 100).toFixed(0)}%</td>
                    <td>{(m.recall * 100).toFixed(0)}%</td>
                    <td><b>{(m.f1 * 100).toFixed(1)}%</b></td>
                    <td>{m.tp}</td>
                    <td>{m.fp}</td>
                    <td>{m.fn}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Confusion Matrix */}
      {bm.confusion_matrix && <ConfusionMatrix cm={bm.confusion_matrix} />}

      {/* Robustness Benchmark */}
      {bm.robustness && <RobustnessSection rob={bm.robustness} />}
    </>
  )
}

function RobustnessSection({ rob }) {
  const conditions = rob.conditions || {}
  return (
    <div className="card" style={{ marginBottom: '1.5rem' }}>
      <h4 style={{ fontSize: '0.9rem', fontWeight: 600, marginBottom: '0.75rem' }}>Robustness Benchmark (Degraded Conditions)</h4>
      <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '0.75rem' }}>
        Detection performance on synthetically degraded images — with vs. without the feedback-loop enhancer.
      </p>
      <div className="table-wrap" style={{ border: 'none', boxShadow: 'none' }}>
        <table>
          <thead>
            <tr><th>Condition</th><th>Without Enhancer</th><th>With Enhancer</th><th>Improvement</th></tr>
          </thead>
          <tbody>
            {Object.entries(conditions).map(([cond, d]) => {
              const imp = d.detection_improvement_pct
              const color = imp > 0 ? '#059669' : imp < -5 ? '#dc2626' : '#6b7280'
              return (
                <tr key={cond}>
                  <td style={{ fontWeight: 500, textTransform: 'capitalize' }}>{cond.replace(/_/g, ' ')}</td>
                  <td>{d.without_enhancer.avg_detections} det (conf {(d.without_enhancer.avg_confidence * 100).toFixed(0)}%)</td>
                  <td>{d.with_enhancer.avg_detections} det (conf {(d.with_enhancer.avg_confidence * 100).toFixed(0)}%)</td>
                  <td style={{ color, fontWeight: 600 }}>{imp > 0 ? '+' : ''}{imp}%</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <ul className="calib-notes" style={{ marginTop: '0.5rem' }}>
        <li>• Low-light: gamma 2.5 darkening. Motion blur: 15px directional kernel. Rain: diagonal streak overlay.</li>
        <li>• The enhancer is most effective on illumination problems; motion blur requires deblurring (future work).</li>
      </ul>
    </div>
  )
}

function ConfusionMatrix({ cm }) {
  const labels = (cm.labels || []).map(l => l === '_unmatched' ? 'False Alarm / Missed' : l.replace(/_/g, ' '))
  const matrix = cm.matrix || []
  const maxVal = Math.max(1, ...matrix.flat())
  return (
    <div className="card" style={{ marginBottom: '1.5rem' }}>
      <h4 style={{ fontSize: '0.9rem', fontWeight: 600, marginBottom: '0.75rem' }}>Confusion Matrix (Violation Classification)</h4>
      <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '0.75rem' }}>
        Rows = actual violation type, Columns = predicted type. Diagonal = correct classifications.
      </p>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ fontSize: '0.72rem', textAlign: 'center' }}>
          <thead>
            <tr>
              <th style={{ textAlign: 'left', minWidth: '110px' }}>Actual ↓ / Pred →</th>
              {labels.map((l, i) => <th key={i} style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)', maxWidth: '28px', padding: '4px 2px' }}>{l}</th>)}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, ri) => (
              <tr key={ri}>
                <td style={{ textAlign: 'left', fontWeight: 500, whiteSpace: 'nowrap' }}>{labels[ri]}</td>
                {row.map((val, ci) => {
                  const isDiag = ri === ci
                  const intensity = val / maxVal
                  const bg = val === 0 ? 'transparent' : isDiag
                    ? `rgba(5,150,105,${0.15 + intensity * 0.6})`
                    : `rgba(220,38,38,${0.1 + intensity * 0.5})`
                  return <td key={ci} style={{ background: bg, fontWeight: isDiag && val > 0 ? 700 : 400, padding: '4px 6px' }}>{val || ''}</td>
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
