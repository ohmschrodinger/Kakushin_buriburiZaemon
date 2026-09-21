import { useMemo, useState } from 'react'
import { AlertTriangle, ArrowRight, BarChart3, Check, ChevronLeft, CircleDollarSign, Headphones, Landmark, LoaderCircle, Mic, ShieldCheck, Sparkles, Target, TrendingUp, Wallet, X } from 'lucide-react'
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { analyzeProfile, getVoiceToken, isMockMode } from './services/api'
import { money, percent, titleCase } from './utils/format'

const initialProfile = {
  userId: `demo-${Date.now()}`,
  monthlyIncome: '', monthlyExpenses: '', monthlyEmi: '', existingSavings: '', existingDebt: '',
  dependents: 0, employmentType: 'salaried', age: '', gender: '', state: '', isRural: false,
  casteCategory: '', landHoldingAcres: '', businessType: '', hasBankAccount: true, hasInsurance: false,
  hasHealthInsurance: false, riskAppetite: 'moderate', preferredLang: 'en', suspiciousInput: '', userMessage: '',
}

function App() {
  const [screen, setScreen] = useState('welcome')
  const [profile, setProfile] = useState(initialProfile)
  const [analysis, setAnalysis] = useState(null)
  const [error, setError] = useState('')

  function update(name, value) { setProfile((current) => ({ ...current, [name]: value })) }
  async function submitProfile(event) {
    event.preventDefault()
    setError('')
    if (!profile.monthlyIncome || Number(profile.monthlyIncome) <= 0 || profile.monthlyExpenses === '') {
      setError('Please add your monthly income and expenses to continue.')
      return
    }
    setScreen('loading')
    try {
      const result = await analyzeProfile({
        ...profile,
        monthlyIncome: Number(profile.monthlyIncome), monthlyExpenses: Number(profile.monthlyExpenses),
        monthlyEmi: Number(profile.monthlyEmi || 0), existingSavings: Number(profile.existingSavings || 0),
        existingDebt: Number(profile.existingDebt || 0), dependents: Number(profile.dependents || 0),
        age: profile.age ? Number(profile.age) : null, landHoldingAcres: profile.landHoldingAcres ? Number(profile.landHoldingAcres) : null,
        gender: profile.gender || null, state: profile.state || null, casteCategory: profile.casteCategory || null,
        businessType: profile.businessType || null, suspiciousInput: profile.suspiciousInput || null, userMessage: profile.userMessage || null,
      })
      setAnalysis(result)
      setScreen('dashboard')
    } catch (cause) {
      setError(cause.message || 'We could not complete the analysis.')
      setScreen('onboarding')
    }
  }

  return <div className="app-shell">
    <header className="topbar">
      <button className="brand" onClick={() => setScreen('welcome')} aria-label="Go to ArthSaathi home"><span className="brand-mark">A</span><span>Arth<span>Saathi</span></span></button>
      {analysis && <button className="header-voice" onClick={() => setScreen('voice')}><Mic size={16} /> Talk to Saathi</button>}
    </header>
    {screen === 'welcome' && <Welcome onStart={() => setScreen('onboarding')} />}
    {screen === 'onboarding' && <Onboarding profile={profile} update={update} onBack={() => setScreen('welcome')} onSubmit={submitProfile} error={error} />}
    {screen === 'loading' && <Loading />}
    {screen === 'dashboard' && <Dashboard analysis={analysis} profile={profile} onVoice={() => setScreen('voice')} onNew={() => { setProfile(initialProfile); setAnalysis(null); setScreen('onboarding') }} />}
    {screen === 'voice' && <Voice onBack={() => setScreen(analysis ? 'dashboard' : 'welcome')} />}
  </div>
}

function Welcome({ onStart }) {
  return <main className="welcome page-enter">
    <div className="welcome-copy">
      <div className="eyebrow"><span className="eyebrow-dot" /> Your money, made clearer</div>
      <h1>A calmer way to make sense of your <em>money.</em></h1>
      <p className="hero-text">ArthSaathi brings your financial health, risk signals, government schemes, goals, and future scenarios into one grounded view.</p>
      <button className="primary-button hero-button" onClick={onStart}>Get started <ArrowRight size={18} /></button>
      <div className="trust-line"><ShieldCheck size={16} /> Built around your numbers, not promises</div>
    </div>
    <div className="welcome-orbit" aria-hidden="true">
      <div className="orbit-ring ring-one" /><div className="orbit-ring ring-two" /><div className="orbit-core"><CircleDollarSign size={36} /><span>Clarity<br /><b>starts here</b></span></div>
      <div className="orbit-chip chip-one"><TrendingUp size={15} /> Future scenarios</div><div className="orbit-chip chip-two"><ShieldCheck size={15} /> Safer decisions</div><div className="orbit-chip chip-three"><Target size={15} /> Your goals</div>
    </div>
    <div className="welcome-foot"><span>Financial health</span><span>Risk awareness</span><span>Actionable next steps</span></div>
  </main>
}

function Onboarding({ profile, update, onBack, onSubmit, error }) {
  const [step, setStep] = useState(1)
  const next = () => setStep((value) => Math.min(value + 1, 3))
  return <main className="onboarding page-enter">
    <div className="form-intro"><button className="back-button" onClick={onBack}><ChevronLeft size={18} /> Back</button><div className="eyebrow">Your starting point</div><h1>Let’s put your money<br /><em>in context.</em></h1><p>A few honest numbers are enough to begin. You can leave optional details blank.</p></div>
    <div className="form-panel">
      <div className="stepper"><span className={step >= 1 ? 'active' : ''}>01 <b>Snapshot</b></span><i /><span className={step >= 2 ? 'active' : ''}>02 <b>Context</b></span><i /><span className={step >= 3 ? 'active' : ''}>03 <b>Focus</b></span></div>
      <form onSubmit={onSubmit}>
        {step === 1 && <section className="form-step"><div className="section-kicker">The essentials</div><h2>What does your month look like?</h2><div className="field-grid"><Field label="Monthly income" name="monthlyIncome" value={profile.monthlyIncome} update={update} required prefix="₹" /><Field label="Monthly expenses" name="monthlyExpenses" value={profile.monthlyExpenses} update={update} required prefix="₹" /><Field label="Monthly EMI / debt payments" name="monthlyEmi" value={profile.monthlyEmi} update={update} prefix="₹" /><Field label="Current savings" name="existingSavings" value={profile.existingSavings} update={update} prefix="₹" /><Field label="Existing debt" name="existingDebt" value={profile.existingDebt} update={update} prefix="₹" /><Field label="People financially dependent on you" name="dependents" value={profile.dependents} update={update} type="number" /></div><div className="helper"><Sparkles size={15} /> Your data powers the analysis. It is not used to make guarantees.</div></section>}
        {step === 2 && <section className="form-step"><div className="section-kicker">Useful context</div><h2>What else shapes your decisions?</h2><div className="field-grid"><Field label="Age" name="age" value={profile.age} update={update} type="number" /><SelectField label="Employment type" name="employmentType" value={profile.employmentType} update={update} options={[['salaried','Salaried'],['self_employed','Self-employed'],['gig','Gig worker'],['farmer','Farmer'],['daily_wage','Daily wage'],['unemployed','Unemployed']]} /><SelectField label="Risk appetite" name="riskAppetite" value={profile.riskAppetite} update={update} options={[['conservative','Conservative'],['moderate','Moderate'],['aggressive','Aggressive']]} /><SelectField label="Preferred language" name="preferredLang" value={profile.preferredLang} update={update} options={[['en','English'],['hi','Hindi / Hinglish'],['mr','Marathi'],['kn','Kannada']]} /><Field label="State" name="state" value={profile.state} update={update} type="text" /><Field label="Business / work type" name="businessType" value={profile.businessType} update={update} type="text" /></div><div className="toggle-row"><Toggle label="I have a bank account" value={profile.hasBankAccount} onChange={(value) => update('hasBankAccount', value)} /><Toggle label="I already have insurance" value={profile.hasInsurance} onChange={(value) => update('hasInsurance', value)} /></div></section>}
        {step === 3 && <section className="form-step"><div className="section-kicker">Your focus</div><h2>What would you like help with?</h2><label className="full-field"><span>Your message or financial goal <small>Optional</small></span><textarea value={profile.userMessage} onChange={(event) => update('userMessage', event.target.value)} placeholder="For example: I want to build an emergency fund or buy a house in five years." /></label><label className="full-field"><span>Something you want checked for scams <small>Optional</small></span><textarea value={profile.suspiciousInput} onChange={(event) => update('suspiciousInput', event.target.value)} placeholder="Paste a message, offer, or leave blank." /></label><div className="review-strip"><Check size={18} /><span><b>Ready when you are.</b> We’ll return calculations, context, and modeled scenarios together.</span></div></section>}
        {error && <div className="error-box"><AlertTriangle size={17} /> {error}</div>}
        <div className="form-actions">{step > 1 && <button type="button" className="secondary-button" onClick={() => setStep((value) => value - 1)}>Back</button>}{step < 3 ? <button type="button" className="primary-button" onClick={next}>Continue <ArrowRight size={17} /></button> : <button type="submit" className="primary-button">Analyze my picture <ArrowRight size={17} /></button>}</div>
      </form>
    </div>
  </main>
}

function Field({ label, name, value, update, type = 'number', required = false, prefix }) { return <label className="field"><span>{label} {required && <small>Required</small>}</span><div className="input-wrap">{prefix && <b>{prefix}</b>}<input type={type} min="0" value={value} required={required} onChange={(event) => update(name, event.target.value)} /></div></label> }
function SelectField({ label, name, value, update, options }) { return <label className="field"><span>{label}</span><select value={value} onChange={(event) => update(name, event.target.value)}>{options.map(([key, text]) => <option key={key} value={key}>{text}</option>)}</select></label> }
function Toggle({ label, value, onChange }) { return <button type="button" className={`toggle ${value ? 'on' : ''}`} onClick={() => onChange(!value)}><span className="toggle-knob" /><span>{label}</span></button> }
function Loading() { return <main className="loading page-enter"><div className="loader-mark"><LoaderCircle size={44} /></div><div className="eyebrow">Working through your picture</div><h1>Finding the useful<br /><em>signal.</em></h1><p>Calculating your financial snapshot, checking relevant context, and shaping your next view.</p><div className="loading-track"><span /></div><small>This usually takes less than a minute.</small></main> }

function Dashboard({ analysis, profile, onVoice, onNew }) {
  const [activeView, setActiveView] = useState('overview')
  return <main className="dashboard page-enter"><aside className="sidebar"><div className="side-label">YOUR SAATHI</div><button className={activeView === 'overview' ? 'side-link active' : 'side-link'} onClick={() => setActiveView('overview')}><BarChart3 size={17} /> Overview</button><button className={activeView === 'voice' ? 'side-link active' : 'side-link'} onClick={onVoice}><Mic size={17} /> Voice assistant</button><div className="sidebar-bottom"><div className="mode-pill"><span className="live-dot" /> {isMockMode ? 'Demo mode' : 'Live analysis'}</div><button className="new-analysis" onClick={onNew}>Start new analysis <ArrowRight size={15} /></button></div></aside><div className="dashboard-main"><div className="dashboard-heading"><div><div className="eyebrow">Your financial picture</div><h1>Good morning, <em>Saathi.</em></h1><p>Here’s the signal in your numbers, without the noise.</p></div><button className="voice-pill" onClick={onVoice}><Mic size={17} /> Ask Saathi</button></div><div className="dashboard-grid"><Overview data={analysis} profile={profile} /><Risk data={analysis} /><Simulation data={analysis} /><Scams data={analysis} /><Schemes data={analysis} /><Goals data={analysis} /><Narrative data={analysis} /><Actions data={analysis} /></div></div></main>
}
function Overview({ data, profile }) { const m = data.financialMetrics || {}; const income = m.monthly_income ?? profile?.monthlyIncome; const expenses = m.monthly_expenses ?? profile?.monthlyExpenses; return <section className="dash-section overview-section span-2"><SectionHeading icon={Wallet} eyebrow="Financial overview" title="The month at a glance" /><div className="metric-grid"><Metric label="Monthly income" value={money(income)} /><Metric label="Monthly expenses" value={money(expenses)} /><Metric label="Monthly surplus" value={money(m.monthly_surplus)} accent /><Metric label="Emergency runway" value={m.emergency_months != null ? `${Number(m.emergency_months).toFixed(1)} mo` : 'Not available'} /><Metric label="Debt-to-income" value={percent(m.dti_ratio)} /><Metric label="Savings" value={money(m.savings)} /></div></section> }
function Metric({ label, value, accent }) { return <div className={`metric ${accent ? 'accent' : ''}`}><span>{label}</span><strong>{value || 'Not available'}</strong></div> }
function Risk({ data }) { return <section className="dash-section risk-section"><SectionHeading icon={ShieldCheck} eyebrow="Financial health" title="Risk, in context" /><div className="risk-score"><div className="score-ring" style={{ '--score': `${data.riskScore || 0}%` }}><strong>{data.riskScore ?? '—'}</strong><span>/100</span></div><div><span className="status-tag moderate">{titleCase(data.riskCategory || 'unknown')} risk</span><p>Built from debt, surplus, emergency runway, income stability, and savings.</p></div></div><div className="factor-list">{Object.entries(data.riskBreakdown || {}).slice(0, 4).map(([key, value]) => <div key={key}><span>{titleCase(key)}</span><b>{value}/100</b><i><em style={{ width: `${Math.min(100, value)}%` }} /></i></div>)}</div></section> }
function Simulation({ data }) { const paths = data.simulationPaths || {}; const chartData = useMemo(() => Array.from({ length: 36 }, (_, index) => { const row = { month: index + 1 }; Object.entries(paths).forEach(([key, path]) => { row[key] = path.monthly_data?.[index]?.median ?? null }); return row }), [paths]); return <section className="dash-section span-2 simulation-section"><SectionHeading icon={TrendingUp} eyebrow="Future simulation" title="Three ways the next 36 months could unfold" /><div className="simulation-note"><span className="info-dot">i</span> These are modeled financial scenarios, not predictions or guaranteed outcomes.</div>{Object.keys(paths).length ? <div className="chart-wrap"><ResponsiveContainer width="100%" height={250}><AreaChart data={chartData}><defs><linearGradient id="modFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#ef574d" stopOpacity={0.28} /><stop offset="100%" stopColor="#ef574d" stopOpacity={0} /></linearGradient></defs><CartesianGrid stroke="#2b3031" vertical={false} /><XAxis dataKey="month" stroke="#727979" tickLine={false} axisLine={false} /><YAxis stroke="#727979" tickLine={false} axisLine={false} tickFormatter={(value) => `₹${Math.round(value / 1000)}k`} /><Tooltip contentStyle={{ background: '#191c1d', border: '1px solid #363d3d', borderRadius: 8 }} formatter={(value) => money(value)} labelFormatter={(label) => `Month ${label}`} /><Area type="monotone" dataKey="status_quo" stroke="#777f7e" fill="transparent" strokeWidth={2} name="Status quo" /><Area type="monotone" dataKey="moderate" stroke="#ef574d" fill="url(#modFill)" strokeWidth={2.5} name="Moderate" /><Area type="monotone" dataKey="optimal" stroke="#d0a66b" fill="transparent" strokeWidth={2} name="Optimal" /></AreaChart></ResponsiveContainer></div> : <Empty text="Simulation data is not available yet." />}</section> }
function Scams({ data }) { const scams = data.scamFlags || []; return <section className="dash-section"><SectionHeading icon={AlertTriangle} eyebrow="Scam detection" title="Stay one step ahead" />{scams.length ? scams.map((item, index) => <div className="alert-card" key={index}><AlertTriangle size={18} /><div><b>{item.scam_type || item.pattern || 'Potential match'}</b><p>{item.warning || item.evidence || 'Review the retrieved evidence carefully.'}</p><small>Confidence {percent(item.confidence)}</small></div></div>) : <Empty icon={ShieldCheck} text="No relevant scam match was returned." subtle="That is not a guarantee of safety." />}</section> }
function Schemes({ data }) { const schemes = data.eligibleSchemes || []; return <section className="dash-section"><SectionHeading icon={Landmark} eyebrow="Government schemes" title="Worth a closer look" />{schemes.length ? schemes.map((item, index) => <div className="scheme-row" key={index}><div className="scheme-icon"><Landmark size={16} /></div><div><b>{item.name}</b><p>{item.gap || item.how_to_apply || 'Supporting information available.'}</p></div><span className={`status-tag ${item.eligible ? 'good' : 'pending'}`}>{item.eligibility_status || (item.eligible ? 'Eligible' : 'Potential')}</span></div>) : <Empty text="No scheme matches available." />}</section> }
function Goals({ data }) { return <section className="dash-section"><SectionHeading icon={Target} eyebrow="Financial goals" title="What matters to you" />{data.goals?.length ? data.goals.map((goal, index) => <div className="goal-row" key={index}><div className="goal-number">0{index + 1}</div><div><b>{goal.title}</b><p>{goal.target_amount != null ? money(goal.target_amount) : 'Target not set'} {goal.horizon_months != null ? `· ${goal.horizon_months} months` : ''}</p></div><span className="priority">P{goal.priority ?? '—'}</span></div>) : <Empty text="No explicit goals were found." />}</section> }
function Narrative({ data }) { return <section className="dash-section span-2 narrative"><SectionHeading icon={Sparkles} eyebrow="Explainability" title="The story in the numbers" />{data.decisionCards?.length ? data.decisionCards.map((card, index) => <div className="narrative-row" key={index}><span>0{index + 1}</span><div><b>{card.action}</b><p>{card.why}</p></div></div>) : <Empty text="Explanation is not available." />}</section> }
function Actions({ data }) { return <section className="dash-section span-2 actions-section"><div className="actions-heading"><SectionHeading icon={Check} eyebrow="Coach plan" title="Your next few moves" /><span className="action-count">{data.actionPlan?.length || 0} actions</span></div>{data.actionPlan?.length ? data.actionPlan.map((item, index) => <div className="action-row" key={index}><span className="action-check">{index + 1}</span><div><b>{item.task}</b><p>{item.deadline || 'When it feels right'}{item.scheme ? ` · ${item.scheme}` : ''}</p></div></div>) : <Empty text="Your action plan is not available." />}</section> }
function SectionHeading({ icon: Icon, eyebrow, title }) { return <div className="section-heading"><div className="section-icon"><Icon size={17} /></div><div><span>{eyebrow}</span><h2>{title}</h2></div></div> }
function Empty({ text, subtle, icon: Icon = CircleDollarSign }) { return <div className="empty"><Icon size={20} /><span>{text}<small>{subtle}</small></span></div> }
function Voice({ onBack }) { const [status, setStatus] = useState('ready'); const [message, setMessage] = useState(''); const [socket, setSocket] = useState(null); async function start() { setStatus('connecting'); setMessage(''); try { const signedUrl = await getVoiceToken(); const connection = new WebSocket(signedUrl); connection.onopen = () => { setSocket(connection); setStatus('listening') }; connection.onclose = () => { setSocket(null); setStatus('disconnected') }; connection.onerror = () => { setStatus('error'); setMessage('The voice service could not be reached.') } } catch (error) { setStatus('error'); setMessage(error.message) } } function stop() { socket?.close(); setSocket(null); setStatus('disconnected') } return <main className="voice-page page-enter"><button className="back-button" onClick={onBack}><ChevronLeft size={18} /> Back to your picture</button><div className="voice-card"><div className={`voice-orb ${status}`}><Mic size={34} /></div><div className="eyebrow">Voice assistant</div><h1>Talk it through<br /><em>with Saathi.</em></h1><p>Ask about your financial picture in a natural conversation. Your browser receives a temporary connection from the backend.</p><div className={`voice-status ${status}`}><span /> {status === 'ready' ? 'Ready to connect' : status === 'connecting' ? 'Connecting securely…' : status === 'listening' ? 'Listening' : status === 'disconnected' ? 'Disconnected' : 'Connection error'}</div>{message && <div className="error-box"><AlertTriangle size={16} /> {message}</div>}{status === 'listening' ? <button className="secondary-button voice-stop" onClick={stop}><X size={17} /> End conversation</button> : <button className="primary-button" onClick={start} disabled={status === 'connecting'}>{status === 'connecting' ? <LoaderCircle className="spin" size={17} /> : <Headphones size={17} />} Start conversation</button>}<small className="voice-note">Voice availability depends on the configured ElevenLabs endpoint.</small></div></main> }

export default App
