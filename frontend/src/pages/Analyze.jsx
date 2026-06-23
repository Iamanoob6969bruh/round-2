import { useState, useRef, useEffect } from 'react'
import { analyzeImage, getDemoImages, analyzeDemo } from '../api.js'
import { Spinner, Tabs } from '../components/Common.jsx'
import EnhancementStage from '../components/analyze/EnhancementStage.jsx'
import DetectionStage from '../components/analyze/DetectionStage.jsx'
import ViolationsStage from '../components/analyze/ViolationsStage.jsx'
import PerformanceStage from '../components/analyze/PerformanceStage.jsx'
import MetricsStage from '../components/analyze/MetricsStage.jsx'
import BatchMode from '../components/analyze/BatchMode.jsx'
import ShinyText from '../components/ShinyText.jsx'

const RESULT_TABS = [
  { id: 'result', label: 'Result' },
  { id: 'pipeline', label: 'Pipeline' },
  { id: 'metrics', label: 'Performance & Metrics' },
]

export default function Analyze() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [dragging, setDragging] = useState(false)
  const [mode, setMode] = useState('single')
  const [tab, setTab] = useState('result')
  const [demos, setDemos] = useState([])
  const fileRef = useRef(null)

  useEffect(() => { getDemoImages().then(d => setDemos(d?.images || [])).catch(() => setDemos([])) }, [])

  async function handleFile(file) {
    if (!file) return
    setLoading(true); setError(null); setResult(null); setTab('result')
    try { setResult(await analyzeImage(file)) }
    catch (e) { setError(e.message) }
    finally { setLoading(false) }
  }

  async function handleDemo(filename) {
    setLoading(true); setError(null); setResult(null); setTab('result')
    try { setResult(await analyzeDemo(filename)) }
    catch (e) { setError(e.message) }
    finally { setLoading(false) }
  }

  function onDrop(e) {
    e.preventDefault(); setDragging(false)
    const file = e.dataTransfer.files?.[0]
    if (file) handleFile(file)
  }

  return (
    <div>
      <div className="page-header">
        <h1>Traffic Violation Analysis</h1>
        <p>Upload a traffic image for automated detection, violation identification, and plate recognition.</p>
      </div>

      <div className="mode-toggle">
        <button className={mode === 'single' ? 'active' : ''} onClick={() => setMode('single')}>Single Image</button>
        <button className={mode === 'batch' ? 'active' : ''} onClick={() => setMode('batch')}>Batch Triage</button>
      </div>

      {mode === 'batch' ? <BatchMode /> : (
        <>
          {!result && !loading && (
            <>
              <div
                className={`upload-zone ${dragging ? 'dragging' : ''}`}
                onClick={() => fileRef.current?.click()}
                onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
                onDragLeave={() => setDragging(false)}
                onDrop={onDrop}
              >
                <div className="upload-icon">📤</div>
                <div className="upload-text">Drag & drop a traffic image, or click to browse</div>
                <div className="upload-hint">Supports JPG, JPEG, PNG, BMP</div>
              </div>

              {demos.length > 0 && (
                <div style={{ marginTop: '2.5rem' }}>
                  <div className="mb-4">
                    <ShinyText
                      text="Or try a curated demo image"
                      speed={1.7}
                      delay={0.2}
                      color="#94a3b8"
                      shineColor="#ffffff"
                      spread={35}
                      direction="left"
                      yoyo
                      pauseOnHover={false}
                      disabled={false}
                      className="text-lg md:text-xl font-bold uppercase tracking-wider block"
                    />
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '1.25rem' }}>
                    {demos.map(d => (
                      <button
                        key={d.file}
                        onClick={() => handleDemo(d.file)}
                        style={{
                          display: 'flex',
                          flexDirection: 'column',
                          overflow: 'hidden',
                          border: '1px solid var(--border)',
                          borderRadius: '12px',
                          background: 'var(--bg-card)',
                          cursor: 'pointer',
                          textAlign: 'left',
                          transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                          position: 'relative',
                          padding: 0,
                          backdropFilter: 'var(--backdrop-blur)',
                          WebkitBackdropFilter: 'var(--backdrop-blur)',
                        }}
                        onMouseEnter={e => {
                          e.currentTarget.style.borderColor = 'var(--accent)';
                          e.currentTarget.style.transform = 'translateY(-4px)';
                          e.currentTarget.style.boxShadow = '0 10px 20px -5px rgba(6, 182, 212, 0.15)';
                          const img = e.currentTarget.querySelector('img');
                          if (img) img.style.transform = 'scale(1.08)';
                        }}
                        onMouseLeave={e => {
                          e.currentTarget.style.borderColor = 'var(--border)';
                          e.currentTarget.style.transform = 'translateY(0)';
                          e.currentTarget.style.boxShadow = 'none';
                          const img = e.currentTarget.querySelector('img');
                          if (img) img.style.transform = 'scale(1)';
                        }}
                      >
                        <div style={{ width: '100%', height: '120px', overflow: 'hidden', borderBottom: '1px solid var(--border)', position: 'relative' }}>
                          <img
                            src={`/demo/${d.file}`}
                            alt={d.title}
                            style={{
                              width: '100%',
                              height: '100%',
                              objectFit: 'cover',
                              transition: 'transform 0.4s ease',
                            }}
                          />
                          <div style={{
                            position: 'absolute',
                            top: '8px',
                            right: '8px',
                            background: 'rgba(0, 0, 0, 0.65)',
                            padding: '2px 8px',
                            borderRadius: '20px',
                            fontSize: '0.65rem',
                            fontWeight: '600',
                            letterSpacing: '0.05em',
                            textTransform: 'uppercase',
                            color: 'var(--accent)',
                            backdropFilter: 'blur(4px)',
                            border: '1px solid rgba(6, 182, 212, 0.3)'
                          }}>
                            Demo
                          </div>
                        </div>
                        <div style={{ padding: '0.85rem' }}>
                          <div style={{ fontWeight: 600, fontSize: '0.9rem', color: 'var(--text)', letterSpacing: '-0.01em' }}>{d.title}</div>
                          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px', lineHeight: '1.3' }}>{d.description}</div>
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
          <input ref={fileRef} type="file" accept=".jpg,.jpeg,.png,.bmp" style={{ display: 'none' }} onChange={(e) => handleFile(e.target.files?.[0])} />

          {loading && <Spinner text="Running detection pipeline..." />}
          {error && <div className="error-box"><strong>Error:</strong><pre>{error}</pre></div>}

          {result && !loading && (
            <div className="fade-in">
              <ResultBanner result={result} onReset={() => { setResult(null); setError(null) }} />

              <Tabs tabs={RESULT_TABS} active={tab} onChange={setTab} />

              {tab === 'result' && (
                <>
                  <div className="img-grid">
                    <div className="img-container">
                      <div className="img-label">📷 Original</div>
                      <img src={result.original} alt="original" />
                    </div>
                    <div className="img-container">
                      <div className="img-label">✨ Annotated</div>
                      <img src={result.annotated} alt="annotated" />
                    </div>
                  </div>
                  <ViolationsStage violations={result.violations} savedCount={result.saved_count} triage={result.triage} methodology={result.eds_methodology} />
                </>
              )}

              {tab === 'pipeline' && (
                <>
                  <EnhancementStage data={result.enhancement} calibration={result.calibration} />
                  <DetectionStage data={result.detection} />
                </>
              )}

              {tab === 'metrics' && (
                <>
                  <PerformanceStage perf={result.performance} />
                  <MetricsStage metrics={result.metrics} perClass={result.detection.per_class} perf={result.performance} />
                </>
              )}
            </div>
          )}
        </>
      )}
    </div>
  )
}

function ResultBanner({ result, onReset }) {
  const count = result.violations.length
  const flagged = count > 0
  const c = result.triage?.counts || {}
  return (
    <div className={`result-banner ${flagged ? 'flagged' : 'clean'}`}>
      <div className="result-icon">{flagged ? '🚨' : '✅'}</div>
      <div className="result-text">
        <h2>{flagged ? `${count} violation${count > 1 ? 's' : ''} detected` : 'No violations detected'}</h2>
        <p>
          {flagged
            ? `${result.detection.vehicles} vehicles, ${result.detection.persons} persons analysed · processed in ${result.performance.total_ms.toFixed(0)}ms`
            : `All ${result.detection.vehicles} vehicles appear compliant · processed in ${result.performance.total_ms.toFixed(0)}ms`}
        </p>
      </div>
      {flagged && result.triage?.total > 0 && (
        <div className="result-chips">
          <div className="result-chip rc-auto"><span className="rc-num">{c['auto-issue'] ?? 0}</span><span className="rc-label">Auto</span></div>
          <div className="result-chip rc-review"><span className="rc-num">{c['human-review'] ?? 0}</span><span className="rc-label">Review</span></div>
          <div className="result-chip rc-discard"><span className="rc-num">{c['discard'] ?? 0}</span><span className="rc-label">Discard</span></div>
        </div>
      )}
      <button className="btn-ghost" onClick={onReset}>↻ New image</button>
    </div>
  )
}
