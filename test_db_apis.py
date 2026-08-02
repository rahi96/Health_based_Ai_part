"""Quick test of all APIs against local Docker server - checks DB sourcing."""
import json

import httpx

BASE = "http://localhost:8002"
USER_ID = 2

ENDPOINTS = [
    ("GET", "/health", None),
    ("GET", f"/api/v1/cycle-engine/engine/summary?user_id={USER_ID}", None),
    ("GET", f"/api/v1/cycle-engine/bbt/ui?user_id={USER_ID}", None),
    ("GET", f"/api/v1/cycle-engine/opk/ui?user_id={USER_ID}", None),
    ("GET", f"/api/v1/cycle-engine/calendar/month?user_id={USER_ID}&month=2026-08", None),
    ("GET", f"/api/v1/cycle-engine/reconciliation/current?user_id={USER_ID}", None),
    ("GET", f"/api/v1/cycle-engine/ttc/surge-banner?user_id={USER_ID}", None),
    ("GET", f"/api/v1/cycle-engine/awareness/current-phase?user_id={USER_ID}", None),
    ("GET", f"/api/cycle-awareness?user_id={USER_ID}", None),
    ("GET", f"/api/health-trends?user_id={USER_ID}&period=7d", None),
    ("GET", f"/api/numera-insight?user_id={USER_ID}", None),
    ("GET", f"/api/smart-analysis?user_id={USER_ID}", None),
    ("GET", "/api/daily-scripture", None),
    ("POST", "/api/chat/response", {"user_id": USER_ID, "message": "How is my cycle going?"}),
]


def main():
    results = []
    with httpx.Client(timeout=120) as client:
        for method, path, body in ENDPOINTS:
            url = BASE + path
            try:
                if method == "GET":
                    r = client.get(url)
                else:
                    r = client.post(url, json=body)
                data = r.json()
                text = json.dumps(data, ensure_ascii=False)
                from_db = "mysql" in text.lower()
                summary = text[:300]
                results.append((method, path, r.status_code, from_db, summary))
            except Exception as exc:
                results.append((method, path, "ERR", False, str(exc)[:200]))

    print("=" * 100)
    for method, path, status, from_db, summary in results:
        db_flag = "DB:mysql" if from_db else "-"
        print(f"\n[{status}] {method} {path}  ({db_flag})")
        print(f"    {summary}")
    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
