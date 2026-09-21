function monthlyScenario(start, growth, drift) {
  return Array.from({ length: 36 }, (_, index) => {
    const month = index + 1
    const median = Math.round(start + growth * month + drift * Math.sin(month / 3))
    return { month, p10: Math.round(median * 0.86), median, p90: Math.round(median * 1.14) }
  })
}

export function createMockAnalysis(profile) {
  const surplus = Number(profile.monthlyIncome) - Number(profile.monthlyExpenses) - Number(profile.monthlyEmi || 0)
  const scenarios = {
    status_quo: { projected: 485000, start: 120000, growth: 10150 },
    moderate: { projected: 612000, start: 120000, growth: 13700 },
    optimal: { projected: 790000, start: 120000, growth: 18600 },
  }
  const simulationPaths = Object.fromEntries(Object.entries(scenarios).map(([name, item]) => [name, {
    projected_value: item.projected,
    p10_value: Math.round(item.projected * 0.84),
    p90_value: Math.round(item.projected * 1.16),
    success_prob: name === 'status_quo' ? 0.28 : name === 'moderate' ? 0.52 : 0.71,
    horizon_months: 36,
    monthly_data: monthlyScenario(item.start, item.growth, 4200),
  }]))

  return {
    userId: profile.userId,
    latencyMs: 940,
    credibilityScore: 100,
    credibilityFlags: [],
    financialMetrics: {
      monthly_surplus: surplus,
      dti_ratio: Number(profile.monthlyEmi || 0) / Number(profile.monthlyIncome),
      emergency_months: Number(profile.existingSavings || 0) / Number(profile.monthlyExpenses),
      annual_income: Number(profile.monthlyIncome) * 12,
      income_volatility: profile.employmentType !== 'salaried',
      total_debt: Number(profile.existingDebt || 0),
      savings: Number(profile.existingSavings || 0),
    },
    riskScore: 68.7,
    riskCategory: 'moderate',
    riskBreakdown: { debt_score: 60, emergency_score: 66.7, income_score: 100, surplus_score: 40, investment_score: 80 },
    scamFlags: [],
    eligibleSchemes: [
      { name: 'PM Suraksha Bima Yojana (PMSBY)', eligible: true, eligibility_status: 'eligible', annual_benefit: 200000, gap: null, how_to_apply: 'Enroll through your bank.' },
      { name: 'Atal Pension Yojana (APY)', eligible: false, eligibility_status: 'potentially_eligible', annual_benefit: 60000, gap: 'Confirm tax status and linked mobile number.', how_to_apply: null },
    ],
    goals: [{ title: 'Build emergency fund', target_amount: Number(profile.monthlyExpenses) * 6, horizon_months: 12, priority: 1 }],
    simulationPaths,
    decisionCards: [
      { action: 'Build your buffer', why: 'Your emergency runway is the clearest first foundation.', if_ignored: 'Unexpected costs may force expensive borrowing.', priority: 1 },
      { action: 'Keep debt visible', why: 'Your DTI shows how much monthly income is committed to EMI.', if_ignored: 'New borrowing could reduce flexibility.', priority: 2 },
    ],
    actionPlan: [
      { task: 'Set aside a fixed amount toward your emergency fund.', daily_amount: Math.max(0, Math.round(surplus / 30)), deadline: 'This month', scheme: null },
      { task: 'Review the PMSBY enrollment steps with your bank.', daily_amount: null, deadline: 'This week', scheme: 'PM Suraksha Bima Yojana (PMSBY)' },
    ],
    finalResponse: 'Your profile shows a moderate financial risk level. The modeled scenarios give you a range to plan around, not a promise of what will happen.',
  }
}
