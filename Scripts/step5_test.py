"""Step 5 test: pull exact XBRL figures for JPM FY2025.

Compares against what text search found in Step 4.
"""
from filingsense.xbrl.client import fetch_company_facts
from filingsense.xbrl.facts import get_fact

CIK = "19617" # JPMorgan Chase
FY = 2025

data = fetch_company_facts(CIK)
print(f"company: {data.get('entityName')}, CIK {data.get('cik')}")

TAGS = [
    ("Total assets", "Assets"),
    ("Total provision for credit losses", "ProvisionForLoanLeaseAndOtherLosses"),
    ("Net income", "NetIncomeLoss"),
    ("Total deposits", "Deposits"),
]

for label, tag in TAGS:
    fact = get_fact(data, tag, fy=FY)
    if fact is None:
        print(f"{label:35s} NOT FOUND in XBRL")
        continue
    print(f"{label:35s} ${fact['val']:>18,}   (filed {fact['filed']}, {fact['form']} FY{fact['fy']})")

print("\nStep 4 text search said provision was $14,212 (millions) — XBRL is in")
print("whole dollars, so look for 14,212,000,000. They should agree.")
