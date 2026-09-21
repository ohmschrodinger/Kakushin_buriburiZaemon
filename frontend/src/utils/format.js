export function money(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 'Not available'
  return new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 }).format(Number(value))
}

export function percent(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 'Not available'
  return `${(Number(value) * 100).toFixed(digits)}%`
}

export function titleCase(value = '') {
  return value.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase())
}
