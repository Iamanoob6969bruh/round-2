const STAGES = [
  { icon: '⚙️', title: 'Feedback-Loop Enhancement', desc: 'Iteratively corrects brightness, contrast, exposure, and blur until metrics converge.' },
  { icon: '🚗', title: 'Vehicle Detection', desc: 'YOLOv8-based detection of cars, buses, trucks, two-wheelers, and auto-rickshaws.' },
  { icon: '⚠️', title: 'Violation Detection', desc: 'Triple riding, no helmet, stop-line, red-light, wrong-side, seatbelt, and parking violations.' },
  { icon: '🔢', title: 'Plate Recognition', desc: 'Locates plates on flagged vehicles, performs OCR with iterative enhancement.' },
  { icon: '📦', title: 'Evidence & Analytics', desc: 'Generates annotated evidence, metadata packages, and searchable records.' },
]

const VIOLATIONS = [
  ['Helmet non-compliance', 'Active', 'YOLO helmet model + head-shape geometry'],
  ['Triple riding', 'Active', 'Person↔vehicle spatial-association rules'],
  ['Stop-line violation', 'Active', 'Zebra-crossing detection + line-crossing geometry'],
  ['Red-light violation', 'Active', 'HSV signal analysis + stop-line crossing'],
  ['Wrong-side driving', 'Active', 'Pack-isolation lateral-outlier analysis'],
  ['Seatbelt non-compliance', 'Active', 'Torso diagonal-strap Hough-line detection'],
  ['Illegal parking', 'Active', 'No-parking zones + crossing-obstruction geometry'],
]

const FEATURES = [
  ['🛡️', 'Evidence Defensibility Score', 'Every violation scored 0–100 on evidence quality, then auto-routed to issue/review/discard.'],
  ['🧠', 'Explainable AI', 'Plain-English, measurement-grounded justifications for every decision.'],
  ['⚖️', 'Risk-based Triage', 'Cases ordered by danger — only a fraction need human attention.'],
  ['🛰️', 'Self-calibrating', 'Photometric normalization from image statistics — zero per-site tuning.'],
]

export default function About() {
  return (
    <div>
      <div className="page-header">
        <h1>About TrafficVioLens</h1>
        <p>End-to-end automated traffic violation detection and evidence generation.</p>
      </div>

      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <h3 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '1rem' }}>Pipeline Architecture</h3>
        <div style={{ overflowX: 'auto', padding: '0.5rem 0 1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', minWidth: '700px', fontSize: '0.72rem' }}>
            {STAGES.map((s, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center' }}>
                <div style={{ background: 'var(--bg-secondary)', border: '1.5px solid var(--border)', borderRadius: '8px', padding: '0.5rem 0.7rem', textAlign: 'center', minWidth: '90px' }}>
                  <div style={{ fontSize: '1.2rem' }}>{s.icon}</div>
                  <div style={{ fontWeight: 600, marginTop: '2px' }}>{s.title}</div>
                </div>
                {i < STAGES.length - 1 && <div style={{ color: 'var(--text-muted)', fontSize: '1.1rem', margin: '0 2px' }}>→</div>}
              </div>
            ))}
          </div>
        </div>
        {STAGES.map((s, i) => (
          <div className="about-stage" key={i}>
            <div className="as-icon">{s.icon}</div>
            <div><strong>Stage {i + 1} — {s.title}</strong><p>{s.desc}</p></div>
          </div>
        ))}
      </div>

      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <h3 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '1rem' }}>Violation Coverage</h3>
        <div className="table-wrap" style={{ border: 'none', boxShadow: 'none' }}>
          <table>
            <thead><tr><th>Violation</th><th>Status</th><th>Method</th></tr></thead>
            <tbody>
              {VIOLATIONS.map(([name, status, method]) => (
                <tr key={name}>
                  <td style={{ fontWeight: 500 }}>{name}</td>
                  <td><span className="severity-badge" style={{ background: '#f0fdf4', color: '#059669' }}>{status}</span></td>
                  <td>{method}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <h3 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '1rem' }}>What Makes This Different</h3>
        {FEATURES.map(([icon, title, desc]) => (
          <div className="about-stage" key={title}>
            <div className="as-icon">{icon}</div>
            <div><strong>{title}</strong><p>{desc}</p></div>
          </div>
        ))}
      </div>

      <div className="card" style={{ textAlign: 'center' }}>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
          Built for <strong>Flipkart Grid 6.0</strong> — Theme 3: Robotics & Computer Vision
        </p>
      </div>
    </div>
  )
}
