// API client for the TrafficVioLens backend.
export const BASE = (import.meta.env.VITE_API_URL || '') + '/api'

export async function analyzeImage(file) {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${BASE}/analyze`, { method: 'POST', body: form })
  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try {
      const j = await res.json()
      detail = j.detail || detail
    } catch (_) {}
    throw new Error(detail)
  }
  return res.json()
}

export async function getAnalytics() {
  const res = await fetch(`${BASE}/analytics`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export async function analyzeBatch(files) {
  const form = new FormData()
  for (const f of files) form.append('files', f)
  const res = await fetch(`${BASE}/analyze_batch`, { method: 'POST', body: form })
  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try { const j = await res.json(); detail = j.detail || detail } catch (_) {}
    throw new Error(detail)
  }
  return res.json()
}

export async function searchPlate(plate) {
  const res = await fetch(`${BASE}/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ plate }),
  })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export async function getEvaluation() {
  const res = await fetch(`${BASE}/evaluation`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

const DEMO_IMAGES = [
  {
    file: '1_helmet.png',
    title: 'Helmet Violation',
    description: 'Two-wheeler riders without helmets'
  },
  {
    file: '2_triple_riding.png',
    title: 'Triple Riding',
    description: 'Three persons on a single two-wheeler'
  },
  {
    file: '3_stopline_parking.png',
    title: 'Stop-line & Parking',
    description: 'Vehicles crossing zebra crossing + illegal parking'
  },
  {
    file: '4_red_light.jpg',
    title: 'Red-light Violation',
    description: 'Vehicles running a red signal'
  },
  {
    file: '5_wrong_side.jpg',
    title: 'Wrong-side Driving',
    description: 'Vehicle isolated on the opposing lane'
  }
]

export async function getDemoImages() {
  return { images: DEMO_IMAGES }
}

export async function analyzeDemo(filename) {
  try {
    const response = await fetch(`/demo/${filename}`)
    if (!response.ok) throw new Error(`Failed to fetch local demo image /demo/${filename}`)
    const blob = await response.blob()
    const file = new File([blob], filename, { type: blob.type })
    return await analyzeImage(file)
  } catch (err) {
    console.error("Local demo fetch/upload failed, falling back to backend analysis", err)
    const res = await fetch(`${BASE}/demo/analyze/${filename}`, { method: 'POST' })
    if (!res.ok) {
      let detail = `HTTP ${res.status}`
      try { const j = await res.json(); detail = j.detail || detail } catch (_) {}
      throw new Error(detail)
    }
    return res.json()
  }
}
