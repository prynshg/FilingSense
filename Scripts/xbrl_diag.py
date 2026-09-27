# save as scripts/xbrl_diag.py
from filingsense.xbrl.client import fetch_company_facts

data = fetch_company_facts("19617")
points = data["facts"]["us-gaap"]["ProvisionForLoanLeaseAndOtherLosses"]["units"]["USD"]
cands = [p for p in points if p.get("form") == "10-K" and p.get("fy") == 2025 and p.get("fp") == "FY"]
print(f"{len(cands)} candidates matched fy=2025:")
for p in sorted(cands, key=lambda p: p["filed"]):
    extra = {k: v for k, v in p.items()
             if k not in ("start", "end", "val", "accn", "fy", "fp", "form", "filed", "frame")}
    print(f"  val={p['val']:>15,}  end={p.get('end')}  frame={p.get('frame')}  filed={p['filed']}  extra={extra}")
