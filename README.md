# Vicci Eyewear — AI Visibility Report

A self-contained, single-file HTML dashboard that visualizes Vicci Eyewear's AI visibility (AEO) performance across LLM platforms — Google AI Overviews, Google AI Mode, ChatGPT, Gemini, and Perplexity (Sonar).

**[View the live report →](../../)**

## What this is

This dashboard tracks how often, and how favorably, Vicci Eyewear shows up when AI models answer eyewear-related prompts. It's built for the AEO/SEO team to monitor visibility trends, benchmark against competitors, and spot which prompts and models are worth optimizing for next.

The dashboard reads as a Peekaboo-style AI visibility report:

- **Overview** — headline metrics (AI Visibility %, model runs, best score, total citations, sentiment score, average position), a visibility-over-time chart vs. top competitors, competitor mention rankings, sentiment breakdown, and per-model coverage
- **Prompts** — the individual prompts being tracked and how each performed
- **Sentiment** — how Vicci is talked about when it does get mentioned
- **Competitors** — head-to-head mention volume against Warby Parker, Ray-Ban, Zenni, Eyebuydirect, Prada, LensCrafters, Gucci, Peepers, and others
- **Citations** — the underlying source URLs each model pulled from, with domain and content-type breakdowns
- **Actions** — recommended next steps based on the data

All views support filtering by date range (last 7/14/20 days, last run, custom), AI model, and prompt intent.

## How it works

This is a **static HTML file** — there's no backend, no build step, and no live API calls. All report data (prompt runs, model responses, hit/miss status, sentiment, citations, and source URLs) is embedded directly in the page as JSON, and the page's own JavaScript handles filtering, sorting, and chart rendering entirely in the browser.

That means:

- It can be hosted anywhere that serves static files (GitHub Pages, S3, etc.) with zero server-side setup
- It works offline once loaded — no external API dependency at view time
- **The data is a snapshot as of the last export**, not a live feed. To reflect new AI visibility runs, the source data has to be regenerated and the HTML re-exported/re-deployed (see below)

## Updating the report

1. Pull the latest visibility data for the tracked prompts and date range
2. Regenerate the report export
3. Replace `index.html` (or the report file) in this repo with the new version
4. Commit and push — GitHub Pages will redeploy automatically

There's no in-browser "refresh data" button; updating always means re-exporting and re-deploying the file.

## Repo contents

| File | Purpose |
|---|---|
| `index.html` | The full report — self-contained, includes markup, styles, data, and rendering logic |

## Notes

- Built for internal use by the Vicci Eyewear SEO/AEO team (Wealth Success Digital)
- Best viewed on desktop; layout is not optimized for mobile
- If you're looking to combine this with GA4 and Google Search Console data in one place, see the team's unified Peekaboo + GA4 + GSC reporting effort — this repo covers the AI visibility layer only
