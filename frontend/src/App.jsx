import { Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar.jsx'
import Analyze from './pages/Analyze.jsx'
import Analytics from './pages/Analytics.jsx'
import Evaluation from './pages/Evaluation.jsx'
import About from './pages/About.jsx'
import DarkVeil from './components/DarkVeil.jsx'

export default function App() {
  return (
    <>
      <div style={{ position: 'fixed', top: 0, left: 0, width: '100%', height: '100%', zIndex: -1 }}>
        <DarkVeil
          hueShift={0}
          noiseIntensity={0}
          scanlineIntensity={0}
          speed={0.7}
          scanlineFrequency={0}
          warpAmount={0}
        />
      </div>
      <div className="app-shell" style={{ position: 'relative', zIndex: 1 }}>
        <Navbar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Analyze />} />
            <Route path="/analytics" element={<Analytics />} />
            <Route path="/evaluation" element={<Evaluation />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </main>
      </div>
    </>
  )
}
