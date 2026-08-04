"""Fetches GA4 and Search Console data for the AEO report.

Returns {"available": False} if the service account, GA4 property ID, or
GSC site URL aren't configured, so the report still builds without this
section rather than failing the whole pipeline.
"""

import json
import os
from datetime import date, timedelta


def fetch_google_analytics(sa_info, property_id, days=30):
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

    return {"totals": totals, "channels": channels}


def fetch_search_console(sa_info, site_url, days=30):
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

    return {"totals": totals, "topQueries": top_queries}


def fetch_google_data(cfg, days=30):
    sa_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    ga4_property_id = cfg.get("ga4_property_id")
    gsc_site_url = cfg.get("gsc_site_url")

    if not sa_json or not ga4_property_id or not gsc_site_url:
        return {"available": False}

    try:
        sa_info = json.loads(sa_json)
        ga4 = fetch_google_analytics(sa_info, ga4_property_id, days)
        gsc = fetch_search_console(sa_info, gsc_site_url, days)
        return {"available": True, "days": days, "ga4": ga4, "gsc": gsc}
    except Exception as e:
        print(f"  Warning: Google Analytics/Search Console fetch failed: {e}")
        return {"available": False, "error": str(e)}
