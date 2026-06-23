import { useState, useEffect } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, PieChart, Pie, Cell, ResponsiveContainer } from 'recharts'
import { getAnalytics, searchPlate } from '../api.js'
import { MetricCard, Spinner } from '../components/Common.jsx'

const COLORS = ['#6366f1', '#f43f5e', '#f59e0b', '#10b981', '#06b6d4']

export default function Analytics() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [query, setQuery] = useState('')
  const [searchResults, setSearchResults] = useState(null)

  useEffect(() => { getAnalytics().then(setData).catch(() => setData(null)).finally(() => setLoading(false)) }, [])

  useEffect(() => {
    if (!query) { setSearchResults(null); return }
    const t = setTimeout(async () => {
      try { setSearchResults((await searchPlate(query)).results) }
      catch (_) { setSearchResults([]) }
    }, 350)
    return () => clearTimeout(t)
  }, [query])

  if (loading) return (
    <div className="flex flex-col items-center justify-center min-h-[50vh]">
      <div className="page-header text-center"><h1>Analytics</h1></div>
      <Spinner text="Loading historical insights..." />
    </div>
  )

  const summary = data?.summary || {}
  const total = summary.total_violations || 0

  if (total === 0) return (
    <div className="max-w-4xl mx-auto">
      <div className="page-header mb-8">
        <h1 className="text-3xl font-extrabold tracking-tight text-white mb-2">Analytics</h1>
        <p className="text-slate-400">Historical insights and searchable records.</p>
      </div>
      <div className="card text-center py-16 flex flex-col items-center justify-center">
        <p className="text-5xl mb-4">📭</p>
        <p className="text-slate-300 font-semibold">No violations recorded yet.</p>
        <p className="text-slate-500 text-sm mt-1">Process some video or image files in the Analyze tab to generate charts.</p>
      </div>
    </div>
  )

  const barData = Object.entries(summary.violation_types || {}).map(([name, value]) => ({ 
    name: name.replace(/_/g, ' '), 
    value 
  }))

  return (
    <div className="max-w-7xl mx-auto space-y-8">
      {/* Page Header */}
      <div className="page-header">
        <h1 className="text-3xl font-extrabold tracking-tight text-white mb-2">System Analytics</h1>
        <p className="text-slate-400 text-sm">Historical insights, trends, and searchable violation records database.</p>
      </div>

      {/* Main Bento Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        
        {/* Metric 1 */}
        <div className="card flex items-center justify-between p-6 group hover:border-indigo-500/30 transition-all duration-300">
          <div>
            <p className="text-xs font-bold text-slate-400 uppercase tracking-wider">Total Violations</p>
            <p className="text-4xl font-extrabold text-white mt-2 font-mono">{total}</p>
          </div>
          <span className="text-4xl p-3 bg-indigo-500/10 rounded-2xl group-hover:scale-110 transition-transform duration-300">🚨</span>
        </div>

        {/* Metric 2 */}
        <div className="card flex items-center justify-between p-6 group hover:border-emerald-500/30 transition-all duration-300">
          <div>
            <p className="text-xs font-bold text-slate-400 uppercase tracking-wider">Avg Confidence</p>
            <p className="text-4xl font-extrabold text-emerald-400 mt-2 font-mono">{(summary.avg_confidence * 100).toFixed(0)}%</p>
          </div>
          <span className="text-4xl p-3 bg-emerald-500/10 rounded-2xl group-hover:scale-110 transition-transform duration-300">🎯</span>
        </div>

        {/* Metric 3 */}
        <div className="card flex items-center justify-between p-6 group hover:border-cyan-500/30 transition-all duration-300">
          <div>
            <p className="text-xs font-bold text-slate-400 uppercase tracking-wider">Plates Identified</p>
            <p className="text-4xl font-extrabold text-cyan-400 mt-2 font-mono">{summary.plates_identified}</p>
          </div>
          <span className="text-4xl p-3 bg-cyan-500/10 rounded-2xl group-hover:scale-110 transition-transform duration-300">🔢</span>
        </div>

        {/* Bar Chart Tile (Double width on large screens) */}
        {barData.length > 0 && (
          <>
            <div className="card lg:col-span-2 p-6 flex flex-col">
              <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-6">Violations by Category</h3>
              <div className="h-[300px] w-full mt-auto">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={barData} margin={{ top: 10, right: 10, bottom: 5, left: -10 }}>
                    <defs>
                      <linearGradient id="barGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#818cf8" stopOpacity={0.8}/>
                        <stop offset="95%" stopColor="#4f46e5" stopOpacity={0.15}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                    <XAxis dataKey="name" stroke="#64748b" tick={{ fontSize: 10, fill: '#94a3b8' }} />
                    <YAxis stroke="#64748b" tick={{ fontSize: 10, fill: '#94a3b8' }} allowDecimals={false} />
                    <Tooltip 
                      contentStyle={{ 
                        background: 'rgba(15, 23, 42, 0.95)', 
                        border: '1px solid rgba(255, 255, 255, 0.1)', 
                        borderRadius: 12, 
                        fontSize: '0.85rem',
                        color: '#f9fafb'
                      }} 
                    />
                    <Bar dataKey="value" radius={[6, 6, 0, 0]} fill="url(#barGradient)" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Pie Chart Tile */}
            <div className="card p-6 flex flex-col items-center justify-between relative">
              <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider w-full text-left mb-4">Distribution</h3>
              <div className="h-[240px] w-full relative flex items-center justify-center">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie data={barData} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={65} outerRadius={90} paddingAngle={3} labelLine={false}>
                      {barData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} stroke="rgba(10,15,30,0.5)" strokeWidth={2} />)}
                    </Pie>
                    <Tooltip 
                      contentStyle={{ 
                        background: 'rgba(15, 23, 42, 0.95)', 
                        border: '1px solid rgba(255, 255, 255, 0.1)', 
                        borderRadius: 12, 
                        fontSize: '0.85rem',
                        color: '#f9fafb'
                      }} 
                    />
                  </PieChart>
                </ResponsiveContainer>
                {/* Center Badge */}
                <div className="absolute flex flex-col items-center justify-center text-center pointer-events-none">
                  <span className="text-xs text-slate-400 uppercase font-bold tracking-wider">Total</span>
                  <span className="text-3xl font-extrabold text-white mt-1">{total}</span>
                </div>
              </div>
            </div>
          </>
        )}

        {/* Search Tile (Full Width) */}
        <div className="card lg:col-span-3 p-6 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider">Real-time Plate Lookup</h3>
            <span className="text-xs text-indigo-400 font-semibold font-mono">Query Database</span>
          </div>
          <div className="relative">
            <input 
              className="text-input pl-10 pr-4 py-3 bg-slate-950/60 border-white/5 text-slate-200" 
              placeholder="Enter license plate number... (e.g. KA18EJ8800)" 
              value={query} 
              onChange={(e) => setQuery(e.target.value)} 
            />
            <span className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-500 text-lg">🔍</span>
          </div>
          {query && searchResults !== null && (
            <div className="pt-2 animate-in fade-in slide-in-from-top-2 duration-200">
              {searchResults.length > 0
                ? <RecordsTable columns={Object.keys(searchResults[0])} records={searchResults} />
                : <p className="text-xs text-slate-500 italic py-2">No matching records found.</p>}
            </div>
          )}
        </div>

        {/* Historical Database Table (Full Width) */}
        <div className="card lg:col-span-3 p-6 space-y-4">
          <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider">Violation Records History</h3>
          <RecordsTable columns={data.columns} records={data.records} />
        </div>

      </div>
    </div>
  )
}

function RecordsTable({ columns, records }) {
  if (!records?.length) return <p className="text-xs text-slate-500 italic py-4">No records in database.</p>
  const cols = columns?.length ? columns : Object.keys(records[0])
  return (
    <div className="table-wrap rounded-xl border border-white/5 overflow-hidden">
      <table>
        <thead>
          <tr className="bg-slate-900/50">
            {cols.map((c) => (
              <th key={c} className="text-slate-300 font-bold uppercase tracking-wider py-3.5 px-4 border-b border-white/5">
                {c.replace(/_/g, ' ')}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {records.map((row, i) => (
            <tr key={i} className="hover:bg-white/[0.02] transition-colors border-b border-white/[0.02] last:border-0">
              {cols.map((c) => {
                const val = String(row[c] ?? '')
                const isConfidence = c.includes('confidence') || c.includes('score')
                const displayVal = isConfidence && !isNaN(val) ? `${(parseFloat(val) * 100).toFixed(0)}%` : val
                return (
                  <td key={c} className="py-3 px-4 font-medium text-slate-300 font-sans">
                    {displayVal}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
