"""Fetches recent content activity from the site's XML sitemap(s).

Returns {"available": False} if no sitemap URL is configured. Otherwise
returns the most-recently-modified content pieces (blog posts, product
pages, collection pages, static pages) plus per-month activity buckets,
so the report can show what has been published and updated.

Publish vs. update: a sitemap only carries <lastmod>, which moves both on
first publication and on any later edit. To tell them apart we persist a
snapshot (content_cache.json) and, from the second run onward, flag a URL
as "new" the first run we ever see it, and "updated" when its lastmod
advances. On the very first run there is no baseline, so everything is
reported by lastmod alone with no new/updated badge.

Note on product/collection churn: Shopify bumps a product's or
collection's lastmod on many non-editorial events (inventory, price, tag
changes), so "changed this month" is a much noisier signal for those
types than for blog posts and pages. The report groups by type so that
noise stays visually separable.
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

_TYPES = ("blog", "product", "collection", "page")
_PER_TYPE_RECENT = 40      # rows kept per type for the report table


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


def _classify(loc):
    """Map a URL to a content type, or None to skip it (homepage, blog index)."""
    if re.search(r"/blogs/[^/]+/.+", loc):
        return "blog"
    if re.search(r"/products/[^/]+", loc):
        return "product"
    if re.search(r"/collections/[^/]+", loc):
        return "collection"
    if re.search(r"/pages/[^/]+", loc):
        return "page"
    return None


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


def _type_totals(rows, today):
    def _within(days, e):
        return e["daysAgo"] is not None and e["daysAgo"] <= days
    newly = [e for e in rows if e["publishedDate"] and _days_between(e["publishedDate"], today) <= 90]
    return {
        "total": len(rows),
        "updated30d": sum(1 for e in rows if _within(30, e)),
        "updated90d": sum(1 for e in rows if _within(90, e)),
        "newPublished90d": len(newly),
        "lastChange": rows[0]["lastmodDate"] if rows else None,
    }


def fetch_sitemap_content(cfg, cache_path, months=12):
    urls = cfg.get("content_sitemap_urls")
    if isinstance(urls, str):
        urls = [urls]
    if not urls and cfg.get("content_sitemap_url"):
        urls = [cfg["content_sitemap_url"]]
    if not urls:
        return {"available": False}

    cache = load_content_cache(cache_path)
    meta = cache.setdefault("_meta", {})
    today = _today()
    first_run = "initialized" not in meta
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

    # Each content type gets its own "tracking started" date. Adding a new
    # sitemap later must not flag its whole back catalogue as newly published,
    # so a type's first run is a silent baseline: no new/updated badges.
    type_init = meta.setdefault("type_initialized", {})
    if initialized and "blog" not in type_init:
        type_init["blog"] = initialized  # migrate the original blog-only baseline
    present_types = {t for t in (_classify(it["loc"]) for it in raw) if t}
    type_first_run = {t: (t not in type_init) for t in present_types}
    for t in present_types:
        type_init.setdefault(t, today)

    now = datetime.now(timezone.utc)
    entries = []
    for it in raw:
        loc = it["loc"]
        ctype = _classify(loc)
        if ctype is None:
            continue

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
        t_init = type_init.get(ctype, today)
        # A first_seen strictly after this type's tracking began = a genuinely new URL.
        known_publish = None
        if not type_first_run.get(ctype) and first_seen and first_seen > t_init:
            known_publish = first_seen
        last_updated = prev.get("last_updated")

        status = None
        if known_publish and _days_between(known_publish, today) <= _NEW_BADGE_DAYS:
            status = "new"
        # "Updated" only for types where a lastmod bump means an editorial edit.
        # Shopify rewrites product/collection lastmod on every inventory or
        # price change, so an "Updated" badge there would just be noise.
        elif (ctype in ("blog", "page") and not type_first_run.get(ctype)
              and last_updated and _days_between(last_updated, today) <= _UPDATED_BADGE_DAYS):
            status = "updated"

        entries.append({
            "url": loc,
            "type": ctype,
            "title": it["image_title"] or _title_from_slug(loc),
            "image": it["image"],
            "lastmod": it["lastmod"],
            "lastmodDate": lm_date,
            "publishedDate": known_publish,
            "status": status,
            "daysAgo": (now - lm_dt).days if lm_dt else None,
        })

    save_content_cache(cache_path, cache)

    entries.sort(key=lambda e: e["lastmod"] or "", reverse=True)
    by_type = {t: [e for e in entries if e["type"] == t] for t in _TYPES}

    # Per-month activity, split by type, over the trailing `months`.
    monthly = {}
    for e in entries:
        if not e["lastmodDate"]:
            continue
        ym = e["lastmodDate"][:7]
        row = monthly.setdefault(ym, {t: 0 for t in _TYPES})
        row[e["type"]] += 1
    ym_sorted = sorted(monthly.keys())[-months:]
    monthly_out = [dict(month=ym, **monthly[ym]) for ym in ym_sorted]

    # Table payload: the most recent rows of every type, so a type filter in
    # the report always has enough to show even when one type dominates the
    # global recency order.
    recent = sorted(
        [e for t in _TYPES for e in by_type[t][:_PER_TYPE_RECENT]],
        key=lambda e: e["lastmod"] or "", reverse=True,
    )

    totals = _type_totals(entries, today)
    totals["byType"] = {}
    for t in _TYPES:
        tt = _type_totals(by_type[t], today)
        tt["initializedAt"] = type_init.get(t)
        tt["firstRun"] = bool(type_first_run.get(t))
        totals["byType"][t] = tt

    # Overall "first run" is true only while every tracked type is still a
    # baseline (nothing can be flagged new/updated yet).
    overall_first_run = all(type_first_run.get(t) for t in present_types) if present_types else first_run
    earliest_init = min((type_init[t] for t in present_types if t in type_init), default=initialized)

    return {
        "available": True,
        "generatedAt": today,
        "initializedAt": earliest_init,
        "firstRun": overall_first_run,
        "sitemaps": urls,
        "totals": totals,
        "monthly": monthly_out,
        "recent": recent,
    }
