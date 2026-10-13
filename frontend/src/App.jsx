import { useState , useEffect } from 'react'
import './App.css'
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts'

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

function App() {
  const [transaction, setTransaction] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [totalTransactions, setTotalTransactions] = useState(0)
  const [fraudCount, setFraudCount] = useState(0)
  const [history, setHistory] = useState([])
  const legitimateCount = totalTransactions - fraudCount

const chartData = [
  { name: 'Legitimate', value: legitimateCount },
  { name: 'Fraud', value: fraudCount }
]

const COLORS = ['#22c55e', '#ef4444']


  async function loadPredictions() {
  try {
   const response = await fetch(`${API_URL}/predictions`)

  if (!response.ok) {
  throw new Error('Failed to load predictions')
}

    const data = await response.json()

    setHistory(data)
    setTotalTransactions(data.length)
    setFraudCount(
      data.filter(item => item.prediction === 'Fraud').length
    )

  } catch (error) {
    console.error('Error loading predictions:', error)
  }
}

useEffect(() => {
  loadPredictions()
}, [])



async function handleAnalyze() {
  setLoading(true)
  setError('')
  setResult(null)

  try {
    const transactionData = JSON.parse(transaction)

    const response = await fetch(`${API_URL}/predict`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify(transactionData)
})
    if (!response.ok) {
      throw new Error('Unable to analyze transaction. Check your input data.')
    }

    const data = await response.json()

setResult(data)

await loadPredictions()

  } catch (err) {
    setError(err.message)
  } finally {
    setLoading(false)
  }
}


  return (
  <div className="dashboard">
    <header>
      <h1>FraudLens</h1>
      <p>AI-Powered Fraud Detection Dashboard</p>
    </header>

    <main>
      <h2>Dashboard Overview</h2>

      <div className="stats">

        <div className="card">
          <h3>Transactions Analyzed</h3>
          <p>{totalTransactions}</p>
        </div>

        <div className="card">
          <h3>Fraud Detected</h3>
          <p>{fraudCount}</p>
        </div>

        <div className="card">
          <h3>Fraud Detection Rate</h3>
          <p>
            {totalTransactions > 0
              ? ((fraudCount / totalTransactions) * 100).toFixed(1)
              : 0}%
          </p>
        </div>

      </div>
      <section className="analytics">
  <h2>Fraud Analytics</h2>

  <div style={{ width: '100%', height: 300 }}>
    {totalTransactions > 0 ? (
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={chartData}
            dataKey="value"
            nameKey="name"
            cx="50%"
            cy="50%"
            outerRadius={100}
          >
            {chartData.map((entry, index) => (
              <Cell
                key={entry.name}
                fill={COLORS[index]}
              />
            ))}
          </Pie>

          <Tooltip />
          <Legend />
        </PieChart>
      </ResponsiveContainer>
    ) : (
      <p>Analyze a transaction to see fraud analytics.</p>
    )}
  </div>
</section>

        <section className="analysis">
          <h2>Transaction Analysis</h2>
          <p>Paste a transaction as JSON to analyze it using
             our XGBoost fraud detection model.
          </p>
          <textarea
             placeholder='{"Time": 0, "V1": -1.35, ..., "Amount": 149.62}'
             value={transaction}
             onChange={(event) => setTransaction(event.target.value)}
             rows={8}
          />



          <button onClick={handleAnalyze} disabled={loading}>
            {loading ? 'Analyzing...' : 'Analyze Transaction'}
          </button>
          
          {error && (
           <p style={{ color: '#f87171' }}>
          {error}
           </p>
  )}

          {result && (
            <div className="prediction-result">
              <h3>Prediction Result</h3>

              <p>
                Prediction: <strong>{result.prediction}</strong>
              </p>

              <p>
                Fraud Probability:
                <strong> {(result.fraud_probability * 100).toFixed(2)}%</strong>
              </p>
            </div>
          )}

        </section>
        <section className="history">
  <h2>Recent Predictions</h2>

  {history.length === 0 ? (
    <p>No transactions analyzed yet.</p>
  ) : (
    <table>
      <thead>
        <tr>
          <th>Amount</th>
          <th>Prediction</th>
          <th>Fraud Probability</th>
        </tr>
      </thead>

      <tbody>
        {history.map(item => (
          <tr key={item.id}>
            <td>{item.amount.toFixed(2)}</td>
            <td>{item.prediction}</td>
            <td>{(item.fraud_probability * 100).toFixed(2)}%</td>
          </tr>
        ))}
      </tbody>
    </table>
  )}
</section>

      </main>
    </div>
  )
}

export default App