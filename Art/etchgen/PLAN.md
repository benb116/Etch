# etchgen: Ground-up rethink of the image → lineset algorithm

## Context

This repo drives a physical Etch-a-Sketch (Pi + steppers turning the knobs) plus two web renderers (`rPi/public/index.html`, `docs/index.html` → lineograph.com). Art is a "lineset": `{"name", "pxSpeed", "pxPerRev", "points": [[x,y],...]}` — one continuous polyline, never lifting the pen. Diagonal segments are allowed (both motors run simultaneously).

The current generator (`Art/newImageGen.py` → `Art/imageParser.py` / `Art/graphMods.py` / `Art/artUtils.py`) works but has structural problems:

- **Per-pixel graph**: every Canny pixel is a node with unit edges → 100k+ node multigraphs; `ConnectSubgraphs` (morphological border-growing per component) and `FinalizeGraph` (all-pairs Dijkstra + optimal matching, gated to ≤300 odd nodes) are the bottlenecks.
- **Visible artifact lines**: disjoint components get bridged with naked straight lines across white space; Eulerization duplicates edges via ad-hoc distance-threshold heuristics.
- **Fixed 45° hatching in 5 discrete tone bins** — limited shading fidelity across image types.
- **No preview of the final path**, no upload UI; params are hardcoded literals; the matplotlib `SliderFigure` shows only the raster mask (not the resulting line) and is disabled.

Priorities: **(1) speed of the algorithm, (2) minimize extra/connector lines added, (3) quality of the drawing**, with **screens + high-quality art as the primary target** (no line-budget/drawing-time constraint). Preview + parameter tuning when an image is added. Free to fully depart from the old algorithm.

## Decisions

1. **Platform**: new clean Python package + a small Flask web app (matches existing stack: `rPi/server.py` Flask, vanilla-JS SVG renderers) for upload → live sliders → preview of the *actual final single line* → save JSON to `rPi/public/art/`. Old `Art/` scripts left untouched.
2. **Pluggable style architecture**; build: (a) flagship *contours + flow-guided tone hatching*, (b) *squiggle scanlines* (inherently one line, zero connectors). Streamline-only "pencil sketch" comes nearly free from (a)'s tracer.
3. **Quality-first**: since physical drawing time no longer matters, *retracing existing lines is free* (invisible) — the solver should aggressively prefer retracing over drawing new visible connector lines.
4. **Output contract unchanged**; absorb `reduceSteps.js`'s simplify/scale/center into the Python postprocess (1-decimal coords, RDP) since 13 MB JSONs are a real browser problem.

## Architecture

New package `Art/etchgen/` + preview app `Art/preview/`. Core principle: **work in the polyline/vector domain, not per-pixel graphs**. A drawing is `list[Polyline]` (each an Nx2 float32 numpy array) until the very end; the solver operates on a *topological* graph whose nodes are only endpoints/junctions (thousands, not 100k+).

```
Art/etchgen/
  params.py      # single param schema: name, type, range, default, stage, tooltip → drives UI sliders AND CLI flags
  preprocess.py  # load, EXIF, grayscale, resize to working res, levels/gamma, CLAHE, bilateral denoise
  flow.py        # orientation field via structure tensor (+ optional smoothing iterations)
  structure.py   # XDoG/Canny → cleanup → thinning → skeleton→polyline tracing → RDP
  tone.py        # tone renderers (hatch/flow streamlines, squiggle)
  solver.py      # topological graph, component joining, ink-aware Eulerization, Hierholzer
  route.py       # cost-field A* connector routing ("hide" connectors along dark/drawn areas)
  geometry.py    # polyline ops: resample, RDP (numpy), Chaikin, KD-tree snapping
  metrics.py     # ink/retrace/connector lengths, point count, est. draw time
  serialize.py   # scale/center to canvas, round, lineset JSON
  pipeline.py    # orchestrator with per-stage caching keyed by param-subset hashes
  cli.py         # CLI entry point
  tests/         # pytest suites + bench.py
Art/preview/
  server.py      # Flask app
  static/index.html, app.js, style.css
requirements.txt # numpy, opencv-contrib-python (ximgproc.thinning), scipy, networkx, Pillow, Flask
```

## Flagship algorithm (contours + flow-guided tone), step by step

**0. Preprocess** (`preprocess.py`): grayscale float32; resize max-dim to working resolution (preview 640, final 1280+ — "detail" slider); levels (black/white point, gamma sliders); optional CLAHE (`cv2.createCLAHE`) for flat photos; optional bilateral filter (cartoons/logos skip). Output: `gray` + blurred `tone_map` (darkness 0..1).

**1. Orientation field** (`flow.py`): structure tensor — `Ix,Iy` via Sobel; `Jxx,Jxy,Jyy` Gaussian-smoothed; orientation `0.5*atan2(2Jxy, Jxx−Jyy) + π/2` + coherence. Chosen over Kang's full iterative ETF: 3 convolutions ≈ 10× faster, quality sufficient for hatching direction; an optional 1–2 vector-smoothing iterations behind a param if needed later.

**2. Structure lines** (`structure.py`):
- **XDoG** (default; params σ, φ, ε; k=1.6) — much cleaner, better-connected artistic linework than Canny; Canny kept as a fallback method param for high-contrast/logo inputs.
- Binary cleanup: 3×3 closing, drop components < min-area (param).
- Thin to 1px: `cv2.ximgproc.thinning`.
- Trace skeleton → polylines: classify pixels by 8-neighbor count (endpoint=1, junction≥3, both via one convolution); walk paths from endpoints/junctions consuming pixels; closed loops get an arbitrary start. Junction pixels become *shared graph nodes*.
- Per-polyline RDP simplify (tol ~1px).

**3. Tone renderer** (`tone.py`) — one core engine, the **variable-density streamline tracer** (Jobard–Lefer): trace streamlines through a direction field with RK2 steps, killing a line when it comes within `dsep(x,y)` of an already-traced line (occupancy grid). `dsep` maps from tone: `lerp(s_min, s_max, brightness)`; regions brighter than a cutoff get no lines. Modes from the same engine:
- **Fixed-angle hatch**: constant direction field at angle θ (slider) → classic parallel hatching with *continuously varying spacing* (no more 5 discrete bins).
- **Flow-guided hatch** (default): direction = orientation field → strokes follow form (hair, contours, drapery).
- **Crosshatch layer**: second pass with field rotated ~90° active only where darkness > threshold (slider).
- **Endpoint snapping**: KD-tree over densely-resampled structure polylines; hatch endpoints within `r_snap` (~local spacing) snap onto the nearest structure point, creating shared T-junction nodes. This is the big component-count killer — most hatch lines terminate at region boundaries which coincide with contour lines.

**4. Solver** (`solver.py`) — the part that makes it one line with minimal visible extra ink:
- **Topological multigraph**: nodes = endpoints/junctions keyed by a 0.5px spatial-hash grid; edges carry their polyline + arc length.
- **Component joining**: MST over components (cross-component nearest pairs via KD-tree/Delaunay over resampled polyline points). Attachment-point scoring prefers **degree-1 endpoints** (a bridge landing on two endpoints makes both even — connects AND Eulerizes in one move). Connector geometry: straight segment in preview; **cost-field A\* route in final** (`route.py`: downsampled grid, cost = 1 + β·brightness, with already-drawn lines rasterized as near-zero cost) so bridges hug dark regions/existing lines instead of slashing across white space; routed path RDP-simplified.
- **Ink-aware Eulerization** (Chinese Postman with a twist): collect odd-degree nodes; build a sparse candidate graph (k≈10 Euclidean nearest via KD-tree); pair cost = `min(retrace_cost = graph shortest-path length, bridge_cost = euclidean × visibility_penalty(mean brightness under segment))`. **visibility_penalty defaults high** — retracing is invisible, so prefer duplicating existing paths over new ink. Matching: min-weight matching on the sparse candidate graph (blossom) when |odd| ≤ ~400, greedy KD-tree nearest-pair matching above that ("matching quality" param). Apply: retrace → duplicate the shortest-path edges; bridge → add routed edge.
- **Leave 2 odd nodes** (Euler *path*): skip the most expensive matched pair.
- **Euler path**: iterative Hierholzer over the multigraph, stitching each edge's polyline in the right orientation → one continuous point list. Each output segment tagged ink/retrace/connector for metrics + preview overlay.

**5. Postprocess** (`geometry.py`, `serialize.py`): optional Chaikin smoothing (organic styles) → global RDP (tol slider ~0.3–1px) → scale/center to canvas (default 1500×865, 80px margin, matching `reduceSteps.js`) → round to 1 decimal → `{"name", "pxSpeed":50, "pxPerRev":200, "points"}`.

## Other styles

- **Squiggle scanlines**: N serpentine rows; sine amplitude/frequency modulated by sampled darkness (amplitude ≤ row pitch/2); rows joined at alternating ends → exactly one polyline by construction (bypasses the solver, zero connectors). Trivial and fast; very native to the medium.
- **Pencil sketch** (near-free): flow-guided streamlines with a wide `dsep` range and no structure layer.

## Preview app (`Art/preview/`)

- `POST /api/image` → session id (in-memory session store, image decoded + EXIF-corrected).
- `GET /api/params?style=` → param schema JSON (from `params.py`) → client auto-generates sliders.
- `POST /api/generate` `{session, style, params, quality: preview|final}` → `{points, segment_classes, metrics, stage_timings}`.
- `POST /api/save` `{session, name}` → runs final-quality render, writes `rPi/public/art/<name>.json`.
- Client (vanilla JS like the rest of the repo): drag-drop image, source-image/result side-by-side, debounced regenerate on slider change, SVG polyline render with **connector/retrace overlay in red/blue** (literally see the extra ink), stats bar (length, ink %, points, est. physical time as info), replay-animation button reusing the `stroke-dasharray/dashoffset` trick from `docs/index.html`.
- **Responsiveness via stage caching** (`pipeline.py`): stages keyed by hash of (image, relevant param subset) — moving a hatch slider reuses cached preprocess/structure; preview runs at 640px with straight connectors + greedy matching; "Final render" runs high-res + A* routing + blossom matching.

## Performance targets

- **Preview**: regenerate < ~1s at 640px working resolution so sliders feel live.
- **Final render**: must scale to large sources — up to ~3000×2000 working resolution, with 1–2 minutes acceptable. Scaling tactics: everything raster is numpy/OpenCV (O(pixels), fine at 6MP); the streamline tracer's occupancy grid is resolution-independent per line-count; the solver works on the *topological* graph so its cost scales with line count, not pixels — greedy KD-tree matching kicks in automatically above the blossom threshold; A* connector routing runs on a downsampled grid regardless of source size. The "detail" param decouples working resolution from source resolution.
- No modin, no per-pixel graphs. Python-loop hot spots (skeleton walk, streamline integration, Hierholzer) are linear with small constants; if 3000px profiling shows the tracer dominating, vectorize the RK2 stepper to advance all active streamlines per iteration (batch numpy).

## Verification

1. **Solver invariant tests** (pytest, most important): on random polyline soups and on real samples — output is one continuous polyline (consecutive stitched points coincide), every input polyline's geometry appears in the output, ≤2 odd nodes pre-Hierholzer, metrics add up (ink + retrace + connector = total).
2. **Style smoke tests** on diverse `Art/Pics` samples (Mario=cartoon, stevejobs=portrait, batman=logo, brain=texture): JSON parses, matches format contract, point count within sane bounds.
3. **E2E**: run the CLI on a sample → load through the `docs/index.html` / `rPi/public/index.html` renderer to confirm it draws as one line; run preview server, upload → tune → save → confirm file lands in `rPi/public/art/` and renders.
4. **Bench**: `tests/bench.py` prints per-stage timings for all samples/styles.

## Phasing

1. Package skeleton, `geometry.py` + `solver.py` core with invariant tests (the heart of the rethink).
2. Preprocess + structure + streamline tone engine + CLI → first end-to-end flagship renders.
3. Flask preview app + slider UI + overlay/metrics.
4. Squiggle style; large-image (3000×2000) profiling pass, vectorize tracer if needed.
5. A* connector routing + blossom matching (final-quality mode), Chaikin smoothing.
6. README for `Art/etchgen`, requirements.txt, golden outputs refreshed.

## Key references synthesized

Chinese Postman formulation for single-stroke drawing (Etch-a-Snap prior art), Kang et al. *Coherent Line Drawing* (ETF/FDoG → structure-tensor stand-in), XDoG thresholding, Jobard–Lefer evenly-spaced streamlines (variable-density tracer), vpype linemerge/linesort model (endpoint merging before solving), StippleGen/TSP-art (considered; rejected as primary style — TSP solve is the slow path and tone quality per compute is worse than streamlines).
