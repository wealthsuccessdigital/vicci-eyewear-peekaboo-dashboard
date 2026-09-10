"""Fetches recent blog/content activity from the site's XML sitemap(s).

Returns {"available": False} if no sitemap URL is configured. Otherwise
returns the most-recently-modified content pieces plus per-month activity
buckets, so the report can show what has been published and updated.

Publish vs. update: a sitemap only carries <lastmod>, which moves both on
first publication and on any later edit. To tell them apart we persist a
snapshot (content_cache.json) and, from the second run onward, flag a URL
as "new" the first run we ever see it, and "updated" when its lastmod
advances. On the very first run there is no baseline, so everything is
reported by lastmod alone with no new/updated badge.
"""

import json
import os
import re
import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone

import requests

_NS = {
    "sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
    "image": "http://www.google.com/schemas/sitemap-image/1.1",
}

_NEW_BADGE_DAYS = 120      # how long a genuinely-new URL keeps its "New" badge
_UPDATED_BADGE_DAYS = 45   # how long an edited URL keeps its "Updated" badge


def _today():
    return date.today().isoformat()


def _days_between(a, b):
    """Whole days from ISO date string a to ISO date string b."""
    return (date.fromisoformat(b) - date.fromisoformat(a)).days


def _parse_lastmod(s):
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.strip())
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def _title_from_slug(loc):
    slug = loc.rstrip("/").rsplit("/", 1)[-1]
    slug = re.sub(r"[-_]+", " ", slug)
    return (slug[:1].upper() + slug[1:]) if slug else loc


def _blog_from_loc(loc):
    m = re.search(r"/blogs/([^/]+)/", loc)
    return m.group(1) if m else None


def load_content_cache(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"  Warning: could not read content cache at {path}: {e}")
    return {}


def save_content_cache(path, cache):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=0)
    except Exception as e:
        print(f"  Warning: could not write content cache to {path}: {e}")


def _fetch_sitemap(url):
    resp = requests.get(url, timeout=30, headers={"User-Agent": "vicci-aeo-report/1.0"})
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    items = []
    for u in root.findall("sm:url", _NS):
        loc_el = u.find("sm:loc", _NS)
        if loc_el is None or not loc_el.text:
            continue
        lm_el = u.find("sm:lastmod", _NS)
        img_el = u.find("image:image/image:loc", _NS)
        img_title_el = u.find("image:image/image:title", _NS)
        items.append({
            "loc": loc_el.text.strip(),
            "lastmod": lm_el.text.strip() if lm_el is not None and lm_el.text else None,
            "image": img_el.text.strip() if img_el is not None and img_el.text else None,
            "image_title": img_title_el.text.strip() if img_title_el is not None and img_title_el.text else None,
        })
    return items


def fetch_sitemap_content(cfg, cache_path, recent_limit=40, months=12):
    urls = cfg.get("content_sitemap_urls")
    if isinstance(urls, str):
        urls = [urls]
    if not urls and cfg.get("content_sitemap_url"):
        urls = [cfg["content_sitemap_url"]]
    if not urls:
        return {"available": False}

    cache = load_content_cache(cache_path)
    meta = cache.setdefault("_meta", {})
    first_run = "initialized" not in meta
    today = _today()
    if first_run:
        meta["initialized"] = today
    initialized = meta["initialized"]

    raw = []
    try:
        for u in urls:
            raw.extend(_fetch_sitemap(u))
    except Exception as e:
        print(f"  Warning: sitemap fetch failed: {e}")
        return {"available": False, "error": str(e)}

    # Keep only article-level blog URLs (drop the /blogs/<blog> index itself).
    articles = [it for it in raw if re.search(r"/blogs/[^/]+/.+", it["loc"])]

    now = datetime.now(timezone.utc)
    entries = []
    for it in articles:
        loc = it["loc"]
        lm_dt = _parse_lastmod(it["lastmod"])
        lm_date = lm_dt.date().isoformat() if lm_dt else None

        prev = cache.get(loc)
        if prev is None:
            prev = {"first_seen": today, "lastmod": it["lastmod"]}
            cache[loc] = prev
        else:
            if it["lastmod"] and it["lastmod"] > (prev.get("lastmod") or ""):
                prev["last_updated"] = today
            prev["lastmod"] = it["lastmod"] or prev.get("lastmod")

        first_seen = prev.get("first_seen")
        # A first_seen strictly after the day tracking began = a genuinely new URL.
        known_publish = first_seen if (first_seen and first_seen > initialized) else None
        last_updated = prev.get("last_updated")

        status = None
        if known_publish and _days_between(known_publish, today) <= _NEW_BADGE_DAYS:
            status = "new"
        elif last_updated and _days_between(last_updated, today) <= _UPDATED_BADGE_DAYS:
            status = "updated"

        entries.append({
            "url": loc,
            "title": it["image_title"] or _title_from_slug(loc),
            "image": it["image"],
            "blog": _blog_from_loc(loc),
            "lastmod": it["lastmod"],
            "lastmodDate": lm_date,
            "publishedDate": known_publish,
            "status": status,
            "daysAgo": (now - lm_dt).days if lm_dt else None,
        })

    save_content_cache(cache_path, cache)

    entries.sort(key=lambda e: e["lastmod"] or "", reverse=True)

    def _within(days, e):
        return e["daysAgo"] is not None and e["daysAgo"] <= days

    monthly = {}
    for e in entries:
        if e["lastmodDate"]:
            ym = e["lastmodDate"][:7]
            monthly[ym] = monthly.get(ym, 0) + 1
    ym_sorted = sorted(monthly.keys())[-months:]
    monthly_out = [{"month": ym, "modified": monthly[ym]} for ym in ym_sorted]

    newly_published = [
        e for e in entries
        if e["publishedDate"] and _days_between(e["publishedDate"], today) <= 90
    ]

    return {
        "available": True,
        "generatedAt": today,
        "initializedAt": initialized,
        "firstRun": first_run,
        "sitemaps": urls,
        "totals": {
            "total": len(entries),
            "updated30d": sum(1 for e in entries if _within(30, e)),
            "updated90d": sum(1 for e in entries if _within(90, e)),
            "newPublished90d": len(newly_published),
            "lastChange": entries[0]["lastmodDate"] if entries else None,
        },
        "monthly": monthly_out,
        "recent": entries[:recent_limit],
    }
