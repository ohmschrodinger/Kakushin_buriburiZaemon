import { createMockAnalysis } from '../mocks/analysisResponse'

const USE_MOCK = import.meta.env.VITE_USE_MOCK !== 'false'
const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
const SERVER_BASE = import.meta.env.VITE_SERVER_BASE_URL || 'http://localhost:5000'

export async function analyzeProfile(profile) {
  if (USE_MOCK) {
    await new Promise((resolve) => setTimeout(resolve, 900))
    return createMockAnalysis(profile)
  }

  const response = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(profile),
  })
  if (!response.ok) throw new Error(`Analysis failed (${response.status})`)
  return response.json()
}

export async function getVoiceToken() {
  const response = await fetch(`${SERVER_BASE}/api/voice/token`)
  const body = await response.json().catch(() => ({}))
  if (!response.ok || !body.signed_url) throw new Error(body.error || 'Voice service unavailable')
  return body.signed_url
}

export const isMockMode = USE_MOCK
