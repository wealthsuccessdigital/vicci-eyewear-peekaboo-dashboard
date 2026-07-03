#!/usr/bin/env python3
"""
build.py — AI Visibility Report Generator
Pulls data from AI Peekaboo API, generates LLM-powered action recommendations,
and injects everything into template.html to produce a self-contained report.

Usage:
    python3 build.py [--config config.json]
"""

import json
import os
import re
import subprocess
import sys
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import requests

# ─── LLM provider configs ─────────────────────────────────────────────────────

PROVIDER_CONFIGS = {
    "anthropic":  {
        "base_url": "https://api.anthropic.com/v1",
        "default_model": "claude-sonnet-4-6",
        "extra_headers": {"anthropic-version": "2023-06-01"},
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o",
        "extra_headers": {},
    },
    "google": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "gemini-2.0-flash",
        "extra_headers": {},
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "gemini-2.0-flash",
        "extra_headers": {},
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
        "extra_headers": {},
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "anthropic/claude-sonnet-4-6",
        "extra_headers": {},
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "default_model": "mistral-large-latest",
        "extra_headers": {},
    },
    "claude-cli": {
        "default_model": "claude-sonnet-4-6",
    },
}


def call_llm(provider, api_key, model, system_prompt, user_prompt, base_url=None):
    """Call any OpenAI-compatible LLM provider and return the text response."""
    if provider == "claude-cli":
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        import json as _json, tempfile as _tmp, os as _os
        _settings_path = "/tmp/empty_settings.json"
        if not _os.path.exists(_settings_path):
            with open(_settings_path, "w") as _f:
                _json.dump({"env": {"ANTHROPIC_API_KEY": None}}, _f)
        cmd = ["claude", "-p", "-", "--settings", _settings_path]
        if model:
            cmd += ["--model", model]
        result = subprocess.run(cmd, input=full_prompt, capture_output=True, text=True, check=True)
        return result.stdout.strip()

    try:
        from openai import OpenAI
    except ImportError:
        print("Error: 'openai' package not installed. Run: pip install openai")
        sys.exit(1)

    pconf = PROVIDER_CONFIGS.get(provider, {})
    resolved_url = base_url or pconf.get("base_url", "https://api.openai.com/v1")
    extra_headers = pconf.get("extra_headers", {})

    client = OpenAI(api_key=api_key, base_url=resolved_url, default_headers=extra_headers)
    response = client.chat.completions.create(
        model=model,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()


# ─── Config ──────────────────────────────────────────────────────────────────

def load_config(path="config.json"):
    if not os.path.exists(path):
        print(f"Error: config file not found at {path}")
        sys.exit(1)
    with open(path) as f:
        cfg = json.load(f)

    # Backward compat: old anthropic_api_key field
    if "anthropic_api_key" in cfg and "llm_api_key" not in cfg:
        cfg["llm_api_key"] = cfg["anthropic_api_key"]
        cfg.setdefault("llm_provider", "anthropic")

    # Environment variable overrides (for CI / GitHub Actions)
    if os.environ.get('AIPEEKABOO_API_KEY'):
        cfg['aipeekaboo_api_key'] = os.environ['AIPEEKABOO_API_KEY']
    if os.environ.get('LLM_API_KEY'):
        cfg['llm_api_key'] = os.environ['LLM_API_KEY']
    if os.environ.get('LLM_PROVIDER'):
        cfg['llm_provider'] = os.environ['LLM_PROVIDER']
    if os.environ.get('OUTPUT_FILE'):
        cfg['output_file'] = os.environ['OUTPUT_FILE']

    required = ["aipeekaboo_api_key", "brands"]
    if cfg.get("llm_provider", "anthropic") != "claude-cli":
        if not cfg.get("llm_api_key"):
            required.append("llm_api_key")
    for key in required:
        if key not in cfg:
            print(f"Error: missing required config key: {key}")
            sys.exit(1)
    cfg.setdefault("llm_api_key", "none")

    # Guard against path traversal in output_file
    _out = cfg.get("output_file", "report.html")
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _out_abs = os.path.normpath(os.path.join(_script_dir, _out))
    if not _out_abs.startswith(_script_dir):
        print(f"Error: output_file path escapes project directory: {_out}")
        sys.exit(1)

    return cfg


# ─── API helpers ─────────────────────────────────────────────────────────────

BASE_URL = "https://www.aipeekaboo.com/api/v1"


def api_get(api_key, path, params=None, retries=5):
    """GET request with rate-limit retry (reads X-RateLimit-Reset header)."""
    headers = {"X-API-Key": api_key}
    url = BASE_URL + path
    for attempt in range(retries):
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        if resp.status_code == 429:
            reset = resp.headers.get("X-RateLimit-Reset")
            if reset:
                wait = max(1, int(reset) - int(time.time()))
            else:
                wait = 60
            print(f"  Rate limited. Waiting {wait}s before retry...")
            time.sleep(wait)
            continue
        elif resp.status_code in (500, 502, 503, 504):
            wait = 2 * (attempt + 1)
            print(f"  Warning: server error {resp.status_code}, retrying in {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"Failed after {retries} retries: GET {path}")


def fetch_all_prompts(api_key, brand_id):
    """Fetch all prompts for a brand with pagination."""
    prompts = []
    page = 1
    while True:
        data = api_get(api_key, f"/brands/{brand_id}/prompts",
                       params={"limit": 200, "page": page})
        batch = data.get("prompts") or data.get("data") or []
        prompts.extend(batch)
        pagination = data.get("pagination", {})
        if not pagination.get("hasMore", False):
            break
        page += 1
    return prompts


_fetch_semaphore = threading.Semaphore(4)  # max 4 concurrent prompt fetches
_fetch_lock = threading.Lock()
_last_fetch_time = [0.0]
_MIN_FETCH_INTERVAL = 3.2  # seconds between calls to stay under 20 req/min

def fetch_prompt_detail(api_key, brand_id, prompt_id):
    """Fetch full prompt detail including history with sources and entities."""
    with _fetch_semaphore:
        with _fetch_lock:
            now = time.time()
            wait = _MIN_FETCH_INTERVAL - (now - _last_fetch_time[0])
            if wait > 0:
                time.sleep(wait)
            _last_fetch_time[0] = time.time()
        # HTTP call is outside the lock but inside the semaphore
        return api_get(api_key, f"/brands/{brand_id}/prompts/{prompt_id}",
                       params={"include_full_response": "true", "time_range": "90d"})


# ─── Classification helpers ───────────────────────────────────────────────────

DOMAIN_CAT_MAP = {
    # Social
    "reddit.com": "Social Platform",
    "quora.com": "Social Platform",
    "linkedin.com": "Social Platform",
    "twitter.com": "Social Platform",
    "x.com": "Social Platform",
    "facebook.com": "Social Platform",
    "instagram.com": "Social Platform",
    "tiktok.com": "Social Platform",
    "pinterest.com": "Social Platform",
    # Video
    "youtube.com": "Video Platform",
    "vimeo.com": "Video Platform",
    "dailymotion.com": "Video Platform",
    # Reviews
    "g2.com": "Review Site",
    "capterra.com": "Review Site",
    "trustpilot.com": "Review Site",
    "producthunt.com": "Review Site",
    "yelp.com": "Review Site",
    "glassdoor.com": "Review Site",
    "tripadvisor.com": "Review Site",
    "trustradius.com": "Review Site",
    "getapp.com": "Review Site",
    "softwareadvice.com": "Review Site",
    # Publishing
    "medium.com": "Publishing Platform",
    "substack.com": "Publishing Platform",
    "wordpress.com": "Publishing Platform",
    # Developer
    "github.com": "Developer Platform",
    "stackoverflow.com": "Developer Platform",
    "dev.to": "Developer Platform",
    "news.ycombinator.com": "Developer Platform",
    # eCommerce platforms
    "apps.shopify.com": "eCommerce Platform",
    "shopify.com": "eCommerce Platform",
    "amazon.com": "eCommerce Platform",
    "amazon.co.uk": "eCommerce Platform",
    "etsy.com": "eCommerce Platform",
    "ebay.com": "eCommerce Platform",
    # Tech news
    "techcrunch.com": "Tech News",
    "wired.com": "Tech News",
    "theverge.com": "Tech News",
    "venturebeat.com": "Tech News",
    # Business news
    "forbes.com": "Business News",
    "bloomberg.com": "Business News",
    "businessinsider.com": "Business News",
    "wsj.com": "Business News",
    "ft.com": "Business News",
    "economist.com": "Business News",
    # General news
    "reuters.com": "News",
    "apnews.com": "News",
    "bbc.com": "News",
    "cnn.com": "News",
    "theguardian.com": "News",
}

# Sets used for fast domain membership checks in classify_url
_SOCIAL_DOMAINS = {
    "reddit.com", "quora.com", "twitter.com", "x.com",
    "linkedin.com", "facebook.com", "instagram.com", "tiktok.com", "pinterest.com",
}
_VIDEO_DOMAINS = {"youtube.com", "vimeo.com", "dailymotion.com"}

# Path segment lists — order within each list is irrelevant; order of the
# if-elif chain in classify_url determines precedence.

# Product detail pages
_PRODUCT_SEGS = [
    "/product/", "/products/", "/produto/", "/productos/", "/produkt/",
    "/produit/", "/produits/", "/item/", "/items/", "/pd/", "/pdp/",
    "/buy/", "/shop/product", "/store/product",
]

# URL path regex for product ID patterns common across e-commerce platforms
_PRODUCT_PATH_RE = re.compile(
    r"(?:"
    r"/p[-_/][a-z0-9]"       # /p-xxx, /p_xxx, /p/xxx  (El Corte Inglés, generic)
    r"|/-/p-"                 # /-/p-  (Decathlon variant)
    r"|/_/[a-z]-p-"           # /_/R-p-  (Decathlon variant)
    r"|/dp/[a-z0-9]{8,}"      # /dp/B08XYZ123  (Amazon ASIN)
    r"|/ref=[a-z]"            # /ref=sr_  (Amazon)
    r"|/sku/[a-z0-9]"         # /sku/12345
    r")",
    re.IGNORECASE,
)

# URL query parameter names that confirm a product detail page
_PRODUCT_QUERY_RE = re.compile(
    r"[?&](productid|product_id|pid|ean|sku|itemid|item_id|variantid|variant_id)=",
    re.IGNORECASE,
)

# Category / collection / browse pages
_CATEGORY_SEGS = [
    "/category/", "/categories/", "/categoria/", "/categorias/",
    "/collection/", "/collections/",
    "/dept/", "/department/", "/departments/",
    "/browse/",
    "/shop/", "/store/",
    # Sport & apparel retail
    "/sport/", "/sports/", "/desporto/", "/deporte/", "/esporte/",
    "/women/", "/men/", "/kids/", "/children/", "/homme/", "/femme/",
    "/sale/", "/deals/", "/offers/", "/outlet/", "/promo/",
    # Common taxonomy segments
    "/clothing/", "/shoes/", "/footwear/", "/accessories/",
    "/electronics/", "/home/", "/garden/", "/furniture/",
    "/fitness/", "/outdoor/", "/running/", "/cycling/",
]

# Editorial / blog content
_BLOG_SEGS = [
    "/blog/", "/articles/", "/article/", "/post/", "/posts/",
    "/editorial/", "/column/", "/columns/",
    "/insights/", "/resources/", "/resource/",
    "/learn/", "/education/", "/content/",
    "/thought-leadership/", "/perspectives/",
]

# Press / news sections on brand sites
_NEWS_SEGS = [
    "/news/", "/press/", "/press-release/", "/press-releases/",
    "/media/", "/media-center/", "/newsroom/",
    "/announcement/", "/announcements/",
]

# Documentation / help centres
_DOC_SEGS = [
    "/docs/", "/doc/", "/documentation/",
    "/help/", "/support/",
    "/faq/", "/faqs/",
    "/knowledge-base/", "/kb/",
    "/manual/", "/getting-started/",
]

# Pricing pages
_PRICING_SEGS = ["/pricing", "/plans", "/prices", "/tarifs", "/precos", "/preços"]

# Review / ratings sections
_REVIEW_SEGS = ["/review/", "/reviews/", "/ratings/", "/testimonials/", "/opinions/", "/avis/"]


def classify_domain(domain):
    if domain in DOMAIN_CAT_MAP:
        return DOMAIN_CAT_MAP[domain]
    # Partial suffix match (catches subdomains like news.example.com)
    for known, label in DOMAIN_CAT_MAP.items():
        if domain.endswith("." + known):
            return label
    # .ai TLD — likely an AI/SaaS product (conservative: only exact TLD, not "ai" substring)
    if domain.endswith(".ai"):
        return "AI/SaaS"
    return "Industry Blog"


def classify_url(url, domain, title=""):
    title = (title or "").lower()
    path = urlparse(url).path.lower()
    query = urlparse(url).query.lower()
    t = title + " " + url.lower()   # combined text used for content_type matching

    # ── Page type (domain_type) ────────────────────────────────────────────────
    # Checks run in priority order: specific domains first, then path signals,
    # then depth-based fallbacks.

    if any(d in domain for d in _SOCIAL_DOMAINS):
        domain_type = "social_media"

    elif any(d in domain for d in _VIDEO_DOMAINS):
        domain_type = "video"

    # Product detail pages — explicit path segments (highest confidence)
    elif any(seg in path for seg in _PRODUCT_SEGS):
        domain_type = "product_page"

    # Product detail pages — URL path ID patterns (e.g. /p-UUID, /-/p-, /dp/ASIN)
    elif _PRODUCT_PATH_RE.search(path):
        domain_type = "product_page"

    # Product detail pages — query parameters (e.g. ?productId=, ?ean=)
    elif _PRODUCT_QUERY_RE.search(query):
        domain_type = "product_page"

    # Blog / editorial content
    elif any(seg in path for seg in _BLOG_SEGS):
        domain_type = "blog_article"

    # Documentation / help
    elif any(seg in path for seg in _DOC_SEGS):
        domain_type = "documentation"

    # Pricing pages
    elif any(seg in path for seg in _PRICING_SEGS):
        domain_type = "pricing_page"

    # Press / news sections
    elif any(seg in path for seg in _NEWS_SEGS):
        domain_type = "news_article"

    # Review / ratings sections
    elif any(seg in path for seg in _REVIEW_SEGS):
        domain_type = "review_page"

    # Category / collection pages — after product and editorial checks
    elif any(seg in path for seg in _CATEGORY_SEGS):
        domain_type = "category_page"

    # Homepage — root or near-root path
    elif path in ("/", ""):
        domain_type = "homepage"

    # Shallow paths (≤2 actual segments) with no other signal → section landing
    elif path.count("/") <= 3:
        domain_type = "category_page"

    else:
        domain_type = "blog_article"

    # ── Content type ───────────────────────────────────────────────────────────
    # Keyword signals override page-type-based defaults where the title/URL
    # give a stronger content-type signal.

    if any(kw in t for kw in ["vs ", " versus ", "comparison", " alternative", "alternatives", "compare "]):
        content_type = "comparison"

    elif any(kw in t for kw in ["how to ", " guide", " tutorial", "step-by-step", "step by step"]):
        content_type = "how_to_guide"

    elif any(kw in t for kw in ["best ", "top ", " tools", " apps", " software", "ranked", "roundup", "top-"]):
        content_type = "listicle_roundup"

    elif any(kw in t for kw in [" review", " reviews", "tested", "hands-on", "hands on", "unboxing", " rating", " ratings"]):
        content_type = "product_review"

    elif any(kw in t for kw in ["case study", "success story", "customer story", "case-study"]):
        content_type = "case_study"

    elif any(kw in t for kw in [" report", " study", " survey", " research", " statistics", " stats"]):
        content_type = "research_report"

    elif any(kw in t for kw in ["press release", "press-release", "announces", "launches", "new launch"]):
        content_type = "press_release"

    # Fall back to page-type-derived content type
    elif domain_type == "social_media":
        content_type = "forum_thread"
    elif domain_type == "video":
        content_type = "video"
    elif domain_type == "product_page":
        content_type = "product_page"
    elif domain_type == "category_page":
        content_type = "category_page"
    elif domain_type == "news_article":
        content_type = "news_article"
    elif domain_type == "review_page":
        content_type = "product_review"
    elif domain_type == "pricing_page":
        content_type = "pricing_page"
    elif domain_type == "documentation":
        content_type = "documentation"
    elif domain_type == "homepage":
        content_type = "brand_homepage"
    else:
        content_type = "blog_article"

    return domain_type, content_type


def extract_domain(url):
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return url


def infer_intent(text):
    """Rule-based searchIntent inference for prompts where the API returns null.
    Maps to the same values the template expects: COMMERCIAL, TRANSACTIONAL,
    INVESTIGATIONAL, INFORMATIONAL, NAVIGATIONAL, BRANDED, SENTIMENT.
    """
    t = text.lower()
    if re.search(r'\b(compar|vs\b|versus|alternat|review|difference between|instead of|better than|pros.and.cons)\b', t):
        return "INVESTIGATIONAL"
    if re.search(r'\b(buy|pric(e|ing|ed)|cost|purchas|sign.?up|get.started|free.trial|demo|subscri(be|ption))\b', t):
        return "TRANSACTIONAL"
    if re.search(r'\b(login|log.?in|sign.?in|homepage|official.site|download.app)\b', t):
        return "NAVIGATIONAL"
    if re.search(r'\b(best|top\s*\d*|leading|recommend|which\s.{1,60}(platform|tool|software|solution|system|service|app)|who\s(offers|provides|has))\b', t):
        return "COMMERCIAL"
    if re.search(r'\b(reddit|opinion|reputation|what.do.people|community|forum|feedback|think.of|experience.with)\b', t):
        return "SENTIMENT"
    return "INFORMATIONAL"


def normalize_comp_name(name):
    """Normalize competitor display name for deduplication (e.g. 'Otterly AI' == 'otterly.ai')."""
    n = name.strip().lower()
    n = re.sub(r'\.(ai|com|io|co|org|net|app)$', '', n)
    n = re.sub(r'\s+ai$', '', n)
    n = re.sub(r'ai$', '', n)  # handles "OtterlyAI" → "Otterly"
    n = re.sub(r'\s+(digital|agency|media|platform|labs?|technologies?|solutions?)$', '', n)
    return re.sub(r'[\s\-]', '', n)


def extract_competitors_llm(full_responses, brand_name, provider, api_key, model, base_url=None):
    """
    LLM-based competitor extraction from full AI response texts.
    full_responses: list of (text, model_key, p_idx, e_idx) tuples
    Returns: list of (competitor_name, model_key, p_idx, e_idx) tuples — only real company/product names.
    """
    valid = [(t, mk, pi, ei) for t, mk, pi, ei in full_responses if t and len(t.strip()) > 80]
    if not valid:
        return []

    _non_comp = {
        "google", "youtube", "chatgpt", "gemini", "perplexity", "bing", "meta",
        "facebook", "instagram", "tiktok", "twitter", "linkedin", "openai",
        "anthropic", "claude", "gpt", "copilot", "wordpress", "shopify",
        "woocommerce", "semrush", "ahrefs", "moz", "hubspot", "salesforce",
        "mailchimp", "amazon", "sortlist", "clutch", "goodfirms", "thomasnet",
    }

    batch_size = 8
    results = []
    results_lock = threading.Lock()
    total_batches = (len(valid) + batch_size - 1) // batch_size
    completed = [0]
    completed_lock = threading.Lock()

    system_prompt = (
        f"Extract competitor company and product names from AI-generated responses about {brand_name}.\n\n"
        f"Return ONLY a JSON object mapping response number (string key) to an array of company/brand names.\n"
        f"Example: {{\"1\": [\"JR Automation\", \"Cognex\"], \"2\": [\"Siemens\", \"FANUC America\"]}}\n\n"
        f"Rules:\n"
        f"- Include: manufacturers, automation companies, tooling suppliers, integrators — any named company or product\n"
        f"- Include names mentioned inline in prose, not just in structured lists\n"
        f"- Use the most canonical short name (e.g. 'Cognex' not 'Cognex Corporation'; 'FANUC' not 'FANUC Robotics')\n"
        f"- Exclude: AI/search platforms (Google, ChatGPT, Gemini, Perplexity, OpenAI, Bing, Meta)\n"
        f"- Exclude: SEO/marketing tools (Semrush, Ahrefs, HubSpot, Salesforce, Mailchimp)\n"
        f"- Exclude: generic phrases, section headers, descriptions, bullet labels — only real proper company names\n"
        f"- Return ONLY valid JSON. No markdown fences, no explanation."
    )

    def process_batch(batch_num, batch):
        sections = [f"[RESPONSE {j+1}]\n{text[:2000]}" for j, (text, _, _, _) in enumerate(batch)]
        batch_text = "\n\n---\n\n".join(sections)
        try:
            raw = call_llm(provider, api_key, model, system_prompt, batch_text, base_url)
            raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip(), flags=re.MULTILINE)
            extracted = json.loads(raw)
        except Exception as e:
            with completed_lock:
                completed[0] += 1
                print(f"    NLP {completed[0]}/{total_batches} — batch {batch_num} failed: {e}")
            return
        batch_results = []
        for j, (text, mk, pi, ei) in enumerate(batch):
            names = extracted.get(str(j + 1), [])
            for name in names:
                name = str(name).strip().rstrip('.,;:')
                if not name or len(name) < 3 or len(name) > 70:
                    continue
                lower = name.lower()
                if any(skip in lower for skip in _non_comp):
                    continue
                batch_results.append((name, mk, pi, ei))
        with results_lock:
            results.extend(batch_results)
        with completed_lock:
            completed[0] += 1
            if completed[0] % 10 == 0 or completed[0] == total_batches:
                print(f"    NLP progress: {completed[0]}/{total_batches} batches done")

    batches = [(i // batch_size + 1, valid[i:i + batch_size]) for i in range(0, len(valid), batch_size)]
    print(f"    Running {total_batches} NLP batches with 8 parallel workers...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_batch, bn, b) for bn, b in batches]
        for f in as_completed(futures):
            pass  # results collected inside process_batch via lock

    return results


def comp_domain_from_name(name):
    known = {
        "google": "google.com",
        "microsoft": "microsoft.com",
        "salesforce": "salesforce.com",
        "hubspot": "hubspot.com",
        "shopify": "shopify.com",
        "klaviyo": "klaviyo.com",
        "zendesk": "zendesk.com",
        "intercom": "intercom.com",
        "mailchimp": "mailchimp.com",
        "gorgias": "gorgias.com",
        "tidio": "tidio.com",
        "dynamic yield": "dynamicyield.com",
        "yotpo": "yotpo.com",
        "okendo": "okendo.io",
        "attentive": "attentive.com",
        "postscript": "postscript.io",
        "recart": "recart.com",
        "privy": "privy.com",
    }
    lower = name.lower()
    for k, v in known.items():
        if k in lower:
            return v
    slug = re.sub(r"[^a-z0-9]", "", lower)
    return slug + ".com"


# ─── Data processing ─────────────────────────────────────────────────────────

def _brand_context(text, brand_name, window=600):
    """Extract a text window centred on the brand mention in the full response."""
    if not text:
        return ""
    idx = text.find(brand_name)
    if idx == -1:
        # Try case-insensitive
        lower = text.lower()
        idx = lower.find(brand_name.lower())
    if idx == -1:
        return text[:window]
    start = max(0, idx - window // 3)
    end = min(len(text), idx + 2 * window // 3)
    excerpt = text[start:end].strip()
    if start > 0:
        excerpt = "…" + excerpt
    if end < len(text):
        excerpt = excerpt + "…"
    return excerpt


def process_brand_data(api_key, brand_cfg, llm_cfg=None):
    """Fetch and process all data for a single brand."""
    brand_id = brand_cfg["id"]
    brand_name = brand_cfg["name"]

    print(f"  Fetching prompts for {brand_name}...")
    prompts_raw = fetch_all_prompts(api_key, brand_id)
    print(f"  Processing {len(prompts_raw)} prompts...")

    prompts_out = []
    all_citations = []       # (url, title, model_key)
    all_entities = []        # (name, entity_type, model_key)
    all_full_responses = []  # (text, model_key, p_idx, e_idx) for LLM NLP pass
    sentiment_mentions = []
    raw_prompt_history = []  # for time-range filtering in the frontend

    # Pre-fetch all prompt details in parallel (rate-limited via semaphore in fetch_prompt_detail)
    print(f"  Parallel-fetching {len(prompts_raw)} prompt details (4 workers, rate-limited)...")
    details_map = {}

    def _fetch_one(p_idx_p):
        p_idx, p = p_idx_p
        prompt_id = p.get("id") or p.get("promptId")
        try:
            detail = fetch_prompt_detail(api_key, brand_id, prompt_id)
            if (p_idx + 1) % 10 == 0 or p_idx + 1 == len(prompts_raw):
                print(f"    Fetched {p_idx + 1}/{len(prompts_raw)} prompts...")
            return p_idx, detail
        except Exception as e:
            print(f"    Warning: could not fetch prompt {prompt_id}: {e}")
            return p_idx, p

    with ThreadPoolExecutor(max_workers=4) as executor:
        for p_idx, detail in executor.map(_fetch_one, enumerate(prompts_raw)):
            if detail is not None:
                details_map[p_idx] = detail

    print(f"  Processing {len(prompts_raw)} prompt details...")
    for p_idx, p in enumerate(prompts_raw):
        prompt_id = p.get("id") or p.get("promptId")
        prompt_text = p.get("promptText") or p.get("text") or ""
        search_intent = p.get("searchIntent") or infer_intent(p.get("promptText") or p.get("text") or "")

        detail = details_map.get(p_idx, p)

        history = (detail.get("data") or {}).get("history") or detail.get("history") or []

        models_data = {}
        scores = []
        mentions_count = 0
        raw_entries = []  # per-entry raw data for this prompt

        for e_idx, entry in enumerate(history):
            model_key = entry.get("aiModel") or entry.get("model", "unknown")
            mentioned = entry.get("mentioned", False)
            score = entry.get("score", 0) or 0
            rank = entry.get("rank")
            sentiment = entry.get("sentiment")
            if sentiment:
                sentiment = sentiment.lower()
                if sentiment not in ("positive", "negative", "neutral", "uncertain"):
                    sentiment = "neutral"

            response_text = (
                entry.get("response") or
                entry.get("fullResponse") or
                entry.get("responseText") or ""
            )
            snippet = response_text[:300] if response_text else ""

            models_data[model_key] = {
                "mentioned": mentioned,
                "score": score,
                "rank": rank,
                "sentiment": sentiment,
                "snippet": snippet,
            }

            if mentioned:
                mentions_count += 1
                scores.append(score)

                if sentiment:
                    full_resp = (
                        entry.get("fullResponse") or
                        entry.get("response") or
                        entry.get("responseText") or ""
                    )
                    sentiment_mentions.append({
                        "prompt": prompt_text,
                        "model": model_key,
                        "rank": rank,
                        "score": score,
                        "sentiment": sentiment,
                        "reason": next(
                            (b.get("mentionSummary", "") for b in (entry.get("brandMentions") or [])
                             if (b.get("entityName") or "").lower() == brand_name.lower()),
                            entry.get("mentionSummary", "") or entry.get("sentimentReason") or ""
                        )[:400],
                        "context": _brand_context(full_resp, brand_name, 600) or snippet[:400],
                        "competitors": [
                            e.get("entityName") or e.get("name", "")
                            for e in (entry.get("brandMentions") or entry.get("entities") or [])
                            if (e.get("type") or e.get("entityType") or "").lower() in ("competitor", "untracked")
                        ],
                    })

            sources = (
                entry.get("sources") or
                entry.get("citedSources") or []
            )
            for src in sources:
                url = src.get("url", "")
                title = src.get("title") or urlparse(url).path or url
                if url:
                    all_citations.append((url, title, model_key))

            for ent in (entry.get("brandMentions") or entry.get("entities") or []):
                ent_type = (ent.get("type") or ent.get("entityType") or "").lower()
                if ent_type in ("competitor", "untracked"):
                    all_entities.append((ent.get("entityName") or ent.get("name", ""), "competitor", model_key))

            # Collect full response for LLM NLP extraction pass (runs after history loop)
            full_resp = entry.get("fullResponse") or ""
            if full_resp:
                all_full_responses.append((full_resp, model_key, p_idx, e_idx))

            # ── Raw entry for frontend time-range filtering ───────────────
            entry_date = entry.get("date", "")
            raw_comps = [
                bm.get("entityName") or bm.get("name", "")
                for bm in (entry.get("brandMentions") or [])
                if (bm.get("type") or "").lower() in ("competitor", "untracked")
                and (bm.get("entityName") or bm.get("name", ""))
            ]
            raw_entry_reason = ""
            raw_entry_context = ""
            if mentioned:
                raw_entry_reason = next(
                    (b.get("mentionSummary", "") for b in (entry.get("brandMentions") or [])
                     if (b.get("entityName") or "").lower() == brand_name.lower()),
                    entry.get("mentionSummary", "") or ""
                )[:400]
                raw_entry_context = _brand_context(
                    entry.get("fullResponse") or entry.get("response") or "", brand_name, 600
                ) or (entry.get("responseSnippet") or "")[:400]
            raw_srcs = []
            for src in (entry.get("sources") or entry.get("citedSources") or []):
                src_url = src.get("url", "")
                if not src_url:
                    continue
                src_title = src.get("title") or src_url
                src_domain = extract_domain(src_url)
                src_dt, src_ct = classify_url(src_url, src_domain, src_title)
                raw_srcs.append({
                    "u": src_url, "t": src_title[:100], "pt": src_dt, "ct": src_ct
                })
            raw_entries.append({
                "date": entry_date,
                "model": model_key,
                "hit": mentioned,
                "sc": score,
                "rk": rank,
                "snt": sentiment,
                "rsn": raw_entry_reason,
                "ctx": raw_entry_context,
                "srcs": raw_srcs,
                "comps": raw_comps,
            })

        avg_score = round(sum(scores) / len(scores), 2) if scores else 0.0
        best_score = max(scores) if scores else 0

        raw_prompt_history.append({
            "id": prompt_id,
            "text": prompt_text,
            "intent": search_intent,
            "entries": raw_entries,
        })

        prompts_out.append({
            "id": prompt_id,
            "text": prompt_text,
            "intent": search_intent,
            "avgScore": avg_score,
            "bestScore": best_score,
            "mentions": mentions_count,
            "totalRuns": len(history),
            "models": models_data,
        })

    # ── LLM NLP competitor extraction pass ───────────────────────────────────
    entities_from_api = len([e for e in all_entities if e[1] == "competitor"])
    llm_comps = []
    if llm_cfg and all_full_responses:
        nlp_sample = all_full_responses[:240]
        print(f"  Running LLM NLP extraction on {len(nlp_sample)} responses (capped from {len(all_full_responses)})...")
        llm_comps = extract_competitors_llm(
            nlp_sample,
            brand_name,
            llm_cfg.get("provider"),
            llm_cfg.get("api_key"),
            llm_cfg.get("model"),
            llm_cfg.get("base_url"),
        )
        for name, mk, _pi, _ei in llm_comps:
            all_entities.append((name, "competitor", mk))
        entities_from_llm = len(llm_comps)
        print(f"  Competitor extraction: {entities_from_api} from API brandMentions + {entities_from_llm} from LLM NLP")
    else:
        print(f"  Competitor extraction: {entities_from_api} from API brandMentions (no LLM pass)")

    # ── Citations aggregation ─────────────────────────────────────────────────
    url_data = {}
    for url, title, model_key in all_citations:
        if url not in url_data:
            url_data[url] = {"title": title, "count": 0, "models": set(), "mc": defaultdict(int)}
        url_data[url]["count"] += 1
        url_data[url]["models"].add(model_key)
        url_data[url]["mc"][model_key] += 1

    domain_counts = defaultdict(int)
    domain_url_list = defaultdict(list)
    domain_type_counts = defaultdict(int)
    content_type_counts = defaultdict(int)

    _url_cache = {}  # url -> (domain, page_type, content_type)
    dcat = {}
    for url, info in url_data.items():
        if url not in _url_cache:
            domain = extract_domain(url)
            dt, ct = classify_url(url, domain, info["title"])
            _url_cache[url] = (domain, dt, ct)
        domain, dt, ct = _url_cache[url]
        domain_counts[domain] += info["count"]
        dcat[domain] = classify_domain(domain)
        domain_type_counts[dt] += info["count"]
        content_type_counts[ct] += info["count"]
        domain_url_list[domain].append({
            "url": url,
            "title": info["title"],
            "count": info["count"],
            "models": sorted(info["models"]),
            "mc": dict(info["mc"]),
            "pageType": dt,
            "contentType": ct,
        })

    top_domains = sorted(domain_counts.items(), key=lambda x: -x[1])[:20]
    top_listicles = []
    for domain, urls in domain_url_list.items():
        for u in urls:
            if u["contentType"] == "listicle_roundup":
                top_listicles.append({"domain": domain, "url": u["url"], "title": u["title"], "count": u["count"]})
    top_listicles = sorted(top_listicles, key=lambda x: -x["count"])[:10]

    citations_out = {
        "total": sum(info["count"] for info in url_data.values()),
        "uniqueUrls": len(url_data),
        "uniqueDomains": len(domain_counts),
        "domainTypes": sorted(
            [{"type": k, "count": v} for k, v in domain_type_counts.items()],
            key=lambda x: -x["count"]
        ),
        "contentTypes": sorted(
            [{"type": k, "count": v} for k, v in content_type_counts.items()],
            key=lambda x: -x["count"]
        ),
        "topDomains": [{"domain": d, "count": c} for d, c in top_domains],
        "topListicles": top_listicles,
    }

    top_40_domains = [d for d, _ in sorted(domain_counts.items(), key=lambda x: -x[1])[:40]]
    durl_brand = {}
    for domain in top_40_domains:
        urls_sorted = sorted(domain_url_list[domain], key=lambda x: -x["count"])[:12]
        durl_brand[domain] = urls_sorted

    # ── Sentiment summary ────────────────────────────────────────────────────
    sent_counts = {"positive": 0, "neutral": 0, "negative": 0, "uncertain": 0}
    for m in sentiment_mentions:
        s = m.get("sentiment") or "neutral"
        sent_counts[s] = sent_counts.get(s, 0) + 1

    sentiment_out = {
        "total_mentions": len(sentiment_mentions),
        **sent_counts,
        "mentions": sentiment_mentions,
    }

    # ── Model citations ──────────────────────────────────────────────────────
    model_cit = defaultdict(lambda: {
        "total": 0,
        "domains": defaultdict(int),
        "domain_types": defaultdict(int),
        "content_types": defaultdict(int),
    })
    for url, title, model_key in all_citations:
        if url in _url_cache:
            domain, dt, ct = _url_cache[url]
        else:
            domain = extract_domain(url)
            dt, ct = classify_url(url, domain, title)
            _url_cache[url] = (domain, dt, ct)
        model_cit[model_key]["total"] += 1
        model_cit[model_key]["domains"][domain] += 1
        model_cit[model_key]["domain_types"][dt] += 1
        model_cit[model_key]["content_types"][ct] += 1

    model_citations_out = {}
    for mk, info in model_cit.items():
        top_doms = sorted(info["domains"].items(), key=lambda x: -x[1])[:10]
        model_citations_out[mk] = {
            "total": info["total"],
            "uniqueDomains": len(info["domains"]),
            "topDomains": [{"domain": d, "count": c} for d, c in top_doms],
            "domainTypes": sorted(
                [{"type": k, "count": v} for k, v in info["domain_types"].items()],
                key=lambda x: -x["count"]
            ),
            "contentTypes": sorted(
                [{"type": k, "count": v} for k, v in info["content_types"].items()],
                key=lambda x: -x["count"]
            ),
        }

    # ── Competitor name deduplication ────────────────────────────────────────
    norm_groups = defaultdict(list)
    for name, etype, model_key in all_entities:
        if not name:
            continue
        norm = normalize_comp_name(name)
        norm_groups[norm].append((name, etype, model_key))

    name_to_canonical = {}
    for norm, entries in norm_groups.items():
        name_counts = defaultdict(int)
        for n, _, _ in entries:
            name_counts[n] += 1
        canonical = max(
            name_counts.keys(),
            key=lambda n: (
                name_counts[n],
                n[0].isupper() if n else False,
                '.' not in n,
                not n.lower().endswith('.ai'),
                len(n),
            ),
        )
        for n, _, _ in entries:
            name_to_canonical[n] = canonical

    all_entities = [(name_to_canonical.get(n, n), et, mk) for n, et, mk in all_entities]

    # ── Competitors aggregation (after dedup so modelMentions uses canonical names) ──
    comp_data = defaultdict(lambda: {
        "mentions": 0, "scores": [], "models": set(),
        "sentiments": [], "model_counts": defaultdict(int),
    })
    for name, etype, model_key in all_entities:
        if not name:
            continue
        comp_data[name]["mentions"] += 1
        comp_data[name]["models"].add(model_key)
        comp_data[name]["model_counts"][model_key] += 1

    competitors_out = []
    for name, info in comp_data.items():
        sents = info["sentiments"]
        top_sent = max(set(sents), key=sents.count) if sents else "neutral"
        competitors_out.append({
            "name": name,
            "mentions": info["mentions"],
            "avgScore": 0,
            "topSentiment": top_sent,
            "models": sorted(info["models"]),
            "modelMentions": dict(info["model_counts"]),
            "summaries": [],
        })
    competitors_out.sort(key=lambda x: -x["mentions"])

    comp_domains = {c["name"]: comp_domain_from_name(c["name"]) for c in competitors_out}

    # Apply canonical names to raw history and sentiment mention competitors
    for rp in raw_prompt_history:
        for re_entry in rp["entries"]:
            re_entry["comps"] = [name_to_canonical.get(c, c) for c in re_entry["comps"]]

    for sm in sentiment_mentions:
        sm["competitors"] = [name_to_canonical.get(c, c) for c in sm["competitors"]]

    # Inject LLM NLP competitors into raw_prompt_history entries so they appear
    # in Prompts tab Top Competitors, Citations competitor filter, and time-range filtering.
    for name, mk, p_idx, e_idx in llm_comps:
        canonical = name_to_canonical.get(name, name)
        if not canonical:
            continue
        entry_comps = raw_prompt_history[p_idx]["entries"][e_idx]["comps"]
        if canonical not in entry_comps:
            entry_comps.append(canonical)

    all_run_dates = sorted({e["date"] for p in raw_prompt_history for e in p["entries"] if e.get("date")})

    return {
        "prompts": prompts_out,
        "citations": citations_out,
        "competitors": competitors_out,
        "sentiment": sentiment_out,
        "modelCitations": model_citations_out,
        "durl": durl_brand,
        "dcat": dcat,
        "comp_domains": comp_domains,
        "raw_history": {
            "runDates": all_run_dates,
            "prompts": raw_prompt_history,
        },
    }


# ─── Actions generation ───────────────────────────────────────────────────────

def build_actions_prompt(brand_name, brand_domain, data):
    prompts_list = data["prompts"]
    total_prompts = len(prompts_list)
    mentioned_prompts = sum(1 for p in prompts_list if p["mentions"] > 0)
    vis_pct = round(100 * mentioned_prompts / total_prompts) if total_prompts else 0

    cit = data["citations"]
    top_domains_str = ", ".join(
        f"{d['domain']} ({d['count']})"
        for d in cit["topDomains"][:10]
    )

    competitors = data["competitors"][:5]
    comp_str = ", ".join(
        f"{c['name']} ({c['mentions']} mentions)"
        for c in competitors
    )

    content_types = cit["contentTypes"][:5]
    total_ct = sum(x["count"] for x in cit["contentTypes"]) or 1
    ct_str = ", ".join(
        f"{x['type']} {round(100*x['count']/total_ct)}%"
        for x in content_types
    )

    model_breakdown_lines = []
    model_mentions = defaultdict(lambda: {"mentioned": 0, "total": 0})
    for p in prompts_list:
        for mk, md in p["models"].items():
            model_mentions[mk]["total"] += 1
            if md.get("mentioned"):
                model_mentions[mk]["mentioned"] += 1
    for mk, counts in model_mentions.items():
        pct = round(100 * counts["mentioned"] / counts["total"]) if counts["total"] else 0
        model_breakdown_lines.append(
            f"{mk}: {pct}% visibility, mentioned in {counts['mentioned']} prompts"
        )

    top_listicles = cit.get("topListicles", [])[:5]
    listicles_str = ", ".join(
        f"{l['domain']} ({l['count']} cites)"
        for l in top_listicles
    )

    return f"""Analyze this AI visibility data for {brand_name} and generate exactly 6 prioritized action recommendations as a JSON array.

Brand: {brand_name} ({brand_domain})
Prompt visibility: {vis_pct}% ({mentioned_prompts} of {total_prompts} prompts trigger a mention)
Total AI citations in this space: {cit['total']} across {cit['uniqueDomains']} unique domains
Top cited domains: {top_domains_str}
Top competitors by AI mentions: {comp_str}
Dominant content types: {ct_str}
Model breakdown: {'; '.join(model_breakdown_lines)}
Top listicle/roundup articles (highest-priority inclusion targets): {listicles_str}

Return a JSON array of exactly 6 objects. Each object must have these exact fields:
{{
  "priority": "high" or "medium",
  "effort": "High effort" or "Med effort",
  "cat": one of "Visibility", "Content", "Citation Strategy", "Competitive",
  "icon": one of exactly: "alert", "list", "play", "zap", "chat", "map", "search", "shield", "target" -- NO emojis, these are string keys only,
  "title": "one clear directive -- what the brand should do (max 90 chars)",
  "signals": ["2-3 specific data points from the data above that make this urgent"],
  "favDomains": ["2-4 specific domain names relevant to this action"],
  "why": "one paragraph explaining the business reason, using specific numbers from the data",
  "steps": ["step 1", "step 2", "step 3", "step 4"],
  "outcome": "what improvement to expect and realistic timeframe",
  "platDomains": ["3-5 domain names relevant to this action for favicon display"]
}}

Rules:
- No em dashes anywhere. Use commas, colons, or reword.
- Every domain name, citation count, and competitor name must come from the data above.
- Write for an SEO/AEO practitioner -- someone who knows what a listicle, schema markup, and Reddit citation are.
- Frame recommendations around topic/entity ownership and getting listed in cited sources -- never say "target this prompt" or "submit to AI models".
- At least 2 of the 6 actions must have a step starting with "this week" or "today".
- Outcome timeframes: listicle inclusion = 60-90 days for citations; YouTube = 2-4 weeks; schema markup = 45-90 days.
- Return only the JSON array, no other text."""


def generate_actions(cfg, brand_name, brand_domain, data):
    provider = cfg.get("llm_provider", "anthropic")
    api_key = cfg.get("llm_api_key")
    pconf = PROVIDER_CONFIGS.get(provider, {})
    model = cfg.get("llm_model") or pconf.get("default_model", "gpt-4o")
    base_url = cfg.get("llm_base_url")

    print(f"  Generating actions for {brand_name} (via {provider} / {model})...")

    system = (
        "You are an expert SEO and AEO (Answer Engine Optimization) strategist. "
        "You generate highly specific, data-driven action recommendations for marketing teams. "
        "Your output is always JSON. Never use em dashes (--). "
        "Write in plain language for an SEO/AEO practitioner."
    )

    raw = call_llm(provider, api_key, model, system, build_actions_prompt(brand_name, brand_domain, data), base_url)

    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"```$", "", raw).strip()

    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        # Try stripping markdown code fences if the LLM wrapped the JSON
        stripped = raw.strip().strip('`')
        if stripped.startswith('json'):
            stripped = stripped[4:].strip()
        try:
            return json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            print(f"  Warning: could not parse actions JSON, using empty actions")
            return []


# ─── Template injection ───────────────────────────────────────────────────────

def brand_toggle_html(brands):
    parts = []
    for i, b in enumerate(brands):
        cls = "bt-btn active" if i == 0 else "bt-btn"
        fav_url = f"https://www.google.com/s2/favicons?domain={b['domain']}&sz=32"
        key_safe = b['key'].replace("'", "\\'")
        onclick = f"setBrand('{key_safe}',this)"
        parts.append(
            f'<button class="{cls}" onclick="{onclick}">'
            f'<img src="{fav_url}" onerror="this.style.display=\'none\'">{b["name"]}'
            f"</button>"
        )
    return "".join(parts)


def js_obj_literal(py_obj):
    return json.dumps(py_obj, ensure_ascii=False)


def inject_template(template_path, output_path, brands, D, DURL, DCAT, comp_domains,
                    BRAND_CFG, ACTIONS, RAW_HISTORY, report_title=None):
    with open(template_path, encoding="utf-8") as f:
        html = f.read()

    if not report_title:
        brand_names = " & ".join(b["name"] for b in brands)
        report_title = f"AI Visibility Report: {brand_names}"

    replacements = {
        "%%REPORT_TITLE%%": report_title,
        "%%BRAND_TOGGLE%%": brand_toggle_html(brands),
        "%%DATA%%": js_obj_literal(D),
        "%%DOMAIN_URLS%%": js_obj_literal(DURL),
        "%%DOMAIN_CATEGORIES%%": js_obj_literal(DCAT),
        "%%COMP_DOMAINS%%": js_obj_literal(comp_domains),
        "%%BRAND_CFG%%": js_obj_literal(BRAND_CFG),
        "%%ACTIONS%%": js_obj_literal(ACTIONS),
        "%%DEFAULT_BRAND%%": brands[0]["key"],
        "%%RAW_HISTORY%%": js_obj_literal(RAW_HISTORY),
    }

    for placeholder, value in replacements.items():
        html = html.replace(placeholder, value)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    config_path = "config.json"
    if len(sys.argv) > 2 and sys.argv[1] == "--config":
        config_path = sys.argv[2]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, config_path)
    template_path = os.path.join(script_dir, "template.html")

    cfg = load_config(config_path)
    api_key = cfg["aipeekaboo_api_key"]
    brands_cfg = cfg["brands"]
    output_file = cfg.get("output_file", "report.html")
    output_path = os.path.join(script_dir, output_file)

    provider = cfg.get("llm_provider", "openai")
    llm_api_key = cfg.get("llm_api_key") or cfg.get("anthropic_api_key")
    llm_model = cfg.get("llm_model") or PROVIDER_CONFIGS.get(provider, {}).get("default_model")
    llm_base_url = cfg.get("llm_base_url")
    llm_cfg = {
        "provider": provider,
        "api_key": llm_api_key,
        "model": llm_model,
        "base_url": llm_base_url,
    }

    if not os.path.exists(template_path):
        print(f"Error: template.html not found at {template_path}")
        sys.exit(1)

    D = {
        "prompts": {},
        "citations": {},
        "competitors": {},
        "sentiment": {},
        "modelCitations": {},
    }
    DURL = {}
    DCAT = {}
    comp_domains_all = {}
    BRAND_CFG = {}
    ACTIONS = {}
    RAW_HISTORY = {}

    for b in brands_cfg:
        brand_name = b["name"]
        brand_key = b["key"]
        brand_domain = b["domain"]

        print(f"\nProcessing brand: {brand_name}")

        brand_data = process_brand_data(api_key, b, llm_cfg=llm_cfg)

        D["prompts"][brand_name] = brand_data["prompts"]
        D["citations"][brand_name] = brand_data["citations"]
        D["competitors"][brand_name] = brand_data["competitors"]
        D["sentiment"][brand_name] = brand_data["sentiment"]
        D["modelCitations"][brand_name] = brand_data["modelCitations"]

        DURL[brand_name] = brand_data["durl"]
        DCAT.update(brand_data["dcat"])
        comp_domains_all.update(brand_data["comp_domains"])

        BRAND_CFG[brand_key] = {
            "key": brand_name,
            "name": brand_name,
            "url": brand_domain,
        }

        ACTIONS[brand_key] = generate_actions(cfg, brand_name, brand_domain, brand_data)
        RAW_HISTORY[brand_key] = brand_data["raw_history"]

    print(f"\nWriting report to {output_file}...")
    inject_template(
        template_path,
        output_path,
        brands_cfg,
        D,
        DURL,
        DCAT,
        comp_domains_all,
        BRAND_CFG,
        ACTIONS,
        RAW_HISTORY,
        report_title=cfg.get("report_title"),
    )
    print(f"Done! Open {output_file} in your browser.")


if __name__ == "__main__":
    main()
