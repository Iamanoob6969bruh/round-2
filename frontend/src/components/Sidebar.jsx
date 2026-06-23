import { NavLink } from 'react-router-dom'

const NAV = [
  { to: '/', icon: '🔍', label: 'Analyze' },
  { to: '/analytics', icon: '📊', label: 'Analytics' },
  { to: '/evaluation', icon: '📐', label: 'Evaluation' },
  { to: '/about', icon: 'ℹ️', label: 'About' },
]

export default function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <h1>🚦 TrafficVioLens</h1>
        <p>AI Vision System</p>
      </div>
      <nav className="nav-links">
        {NAV.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
          >
            <span className="nav-icon">{item.icon}</span>
            <span className="nav-label">{item.label}</span>
          </NavLink>
        ))}
      </nav>
      <div className="sidebar-footer">Flipkart Grid 6.0 · Theme 3</div>
    </aside>
  )
}
