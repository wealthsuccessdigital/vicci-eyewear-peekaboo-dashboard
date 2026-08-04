"""Fetches GA4 and Search Console data for the AEO report.

Returns {"available": False} if the service account, GA4 property ID, or
GSC site URL aren't configured, so the report still builds without this
section rather than failing the whole pipeline.
"""

import json
import os
import re
from collections import defaultdict
from datetime import date, timedelta


def _months_ago(n):
    """First-of-month date n calendar months before today."""
    d = date.today()
    total = d.year * 12 + (d.month - 1) - n
    y, m = divmod(total, 12)
    return date(y, m + 1, 1)


def _pct_change(cur, prev):
    if prev in (None, 0) or cur is None:
        return None
    return round((cur - prev) / prev * 100, 1)


def _compute_deltas(monthly, field):
    """MoM/YoY % change using the last COMPLETE month (excludes an
    in-progress current month so a partial month is never compared
    against a full one)."""
    current_ym = date.today().strftime("%Y-%m")
    complete = [m for m in monthly if m["month"] != current_ym]
    if len(complete) < 2:
        return {"current": complete[-1][field] if complete else None, "mom": None, "yoy": None}
    last = complete[-1]
    mom_prev = complete[-2]
    yoy = None
    if len(complete) >= 13:
        yoy = _pct_change(last[field], complete[-13][field])
    return {
        "current": last[field],
        "mom": _pct_change(last[field], mom_prev[field]),
        "yoy": yoy,
    }


def fetch_google_analytics(sa_info, property_id, days=30, months=24):
    from google.oauth2 import service_account
    from google.analytics.data_v1beta import BetaAnalyticsDataClient
    from google.analytics.data_v1beta.types import (
        RunReportRequest, DateRange, Dimension, Metric,
    )

    creds = service_account.Credentials.from_service_account_info(
        sa_info, scopes=["https://www.googleapis.com/auth/analytics.readonly"]
    )
    # REST transport avoids a gRPC/local-cert-store mismatch seen on some
    # Windows dev machines; harmless on Linux CI runners too.
    client = BetaAnalyticsDataClient(credentials=creds, transport="rest")

    channel_resp = client.run_report(RunReportRequest(
        property=f"properties/{property_id}",
        dimensions=[Dimension(name="sessionDefaultChannelGroup")],
        metrics=[Metric(name="sessions"), Metric(name="totalUsers")],
        date_ranges=[DateRange(start_date=f"{days}daysAgo", end_date="today")],
    ))
    channels = sorted(
        (
            {
                "channel": row.dimension_values[0].value,
                "sessions": int(row.metric_values[0].value),
                "users": int(row.metric_values[1].value),
            }
            for row in channel_resp.rows
        ),
        key=lambda c: c["sessions"],
        reverse=True,
    )

    totals_resp = client.run_report(RunReportRequest(
        property=f"properties/{property_id}",
        metrics=[
            Metric(name="sessions"), Metric(name="totalUsers"),
            Metric(name="engagementRate"), Metric(name="conversions"),
        ],
        date_ranges=[DateRange(start_date=f"{days}daysAgo", end_date="today")],
    ))
    row = totals_resp.rows[0] if totals_resp.rows else None
    totals = {
        "sessions": int(row.metric_values[0].value) if row else 0,
        "users": int(row.metric_values[1].value) if row else 0,
        "engagementRate": round(float(row.metric_values[2].value) * 100, 1) if row else 0,
        "conversions": int(float(row.metric_values[3].value)) if row else 0,
    }

    # Monthly historical series for the Trends section.
    monthly_resp = client.run_report(RunReportRequest(
        property=f"properties/{property_id}",
        dimensions=[Dimension(name="yearMonth"), Dimension(name="sessionDefaultChannelGroup")],
        metrics=[Metric(name="sessions"), Metric(name="totalUsers")],
        date_ranges=[DateRange(start_date=_months_ago(months).isoformat(), end_date="today")],
    ))
    buckets = defaultdict(lambda: {"sessions": 0, "users": 0, "aiAssistantSessions": 0})
    for r in monthly_resp.rows:
        ym_raw = r.dimension_values[0].value  # "202401"
        channel = r.dimension_values[1].value
        sessions = int(r.metric_values[0].value)
        users = int(r.metric_values[1].value)
        ym = f"{ym_raw[:4]}-{ym_raw[4:]}"
        buckets[ym]["sessions"] += sessions
        buckets[ym]["users"] += users
        if re.search(r"ai assistant", channel, re.I):
            buckets[ym]["aiAssistantSessions"] += sessions
    monthly = [{"month": ym, **v} for ym, v in sorted(buckets.items())]

    return {
        "totals": totals,
        "channels": channels,
        "monthly": monthly,
        "deltas": {
            "sessions": _compute_deltas(monthly, "sessions"),
            "aiAssistantSessions": _compute_deltas(monthly, "aiAssistantSessions"),
        },
    }


def fetch_search_console(sa_info, site_url, days=30, months=24):
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    creds = service_account.Credentials.from_service_account_info(
        sa_info, scopes=["https://www.googleapis.com/auth/webmasters.readonly"]
    )
    service = build("searchconsole", "v1", credentials=creds)

    end = date.today()
    start = end - timedelta(days=days)

    totals_resp = service.searchanalytics().query(
        siteUrl=site_url,
        body={"startDate": str(start), "endDate": str(end), "dimensions": []},
    ).execute()
    trow = (totals_resp.get("rows") or [{}])[0]
    totals = {
        "clicks": trow.get("clicks", 0),
        "impressions": trow.get("impressions", 0),
        "ctr": round(trow.get("ctr", 0) * 100, 2),
        "position": round(trow.get("position", 0), 1),
    }

    queries_resp = service.searchanalytics().query(
        siteUrl=site_url,
        body={
            "startDate": str(start), "endDate": str(end),
            "dimensions": ["query"], "rowLimit": 20,
        },
    ).execute()
    top_queries = [
        {
            "query": r["keys"][0],
            "clicks": r["clicks"],
            "impressions": r["impressions"],
            "ctr": round(r["ctr"] * 100, 2),
            "position": round(r["position"], 1),
        }
        for r in queries_resp.get("rows", [])
    ]

    # Monthly historical series for the Trends section. GSC has no
    # month-granularity dimension, so pull daily rows and bucket them.
    monthly_start = _months_ago(months)
    daily_resp = service.searchanalytics().query(
        siteUrl=site_url,
        body={
            "startDate": monthly_start.isoformat(), "endDate": str(end),
            "dimensions": ["date"], "rowLimit": 25000,
        },
    ).execute()
    buckets = defaultdict(lambda: {"clicks": 0, "impressions": 0, "posWeighted": 0.0})
    for r in daily_resp.get("rows", []):
        ym = r["keys"][0][:7]
        buckets[ym]["clicks"] += r["clicks"]
        buckets[ym]["impressions"] += r["impressions"]
        buckets[ym]["posWeighted"] += r["position"] * r["impressions"]
    monthly = []
    for ym, v in sorted(buckets.items()):
        impr = v["impressions"]
        monthly.append({
            "month": ym,
            "clicks": v["clicks"],
            "impressions": impr,
            "ctr": round(v["clicks"] / impr * 100, 2) if impr else 0,
            "position": round(v["posWeighted"] / impr, 1) if impr else 0,
        })

    return {
        "totals": totals,
        "topQueries": top_queries,
        "monthly": monthly,
        "deltas": {
            "clicks": _compute_deltas(monthly, "clicks"),
            "impressions": _compute_deltas(monthly, "impressions"),
        },
    }


def fetch_google_data(cfg, days=30, months=24):
    sa_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    ga4_property_id = cfg.get("ga4_property_id")
    gsc_site_url = cfg.get("gsc_site_url")

    if not sa_json or not ga4_property_id or not gsc_site_url:
        return {"available": False}

    try:
        sa_info = json.loads(sa_json)
        ga4 = fetch_google_analytics(sa_info, ga4_property_id, days, months)
        gsc = fetch_search_console(sa_info, gsc_site_url, days, months)
        return {"available": True, "days": days, "ga4": ga4, "gsc": gsc}
    except Exception as e:
        print(f"  Warning: Google Analytics/Search Console fetch failed: {e}")
        return {"available": False, "error": str(e)}
