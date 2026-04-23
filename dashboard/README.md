# Arena Dashboard

Single-file HTML dashboard for visualizing CombBandits arena results.

## View locally

```
python3 -m http.server 8000 --directory dashboard
```

Then open <http://localhost:8000/>.

## Regenerate data after a new arena run

```
python3 dashboard/build_data.py
```

This rebundles every `arena_results/arena_*.json` into `dashboard/data.js` so the static page can render it without a backend.

## What's shown

- Overall mean regret ranking
- Win-rate distribution (which algorithm wins each random config)
- Per-corruption-type performance (adversarial / consistent_wrong / partial_overlap / uniform)
- Robustness scatter (mean vs worst case) with the Pareto frontier highlighted
- Normalized regret vs CUCB
- Speed vs quality tradeoff
- Full leaderboard table

## Hosting

Push to a `gh-pages` branch or enable Pages from `/dashboard` to make the dashboard public.
