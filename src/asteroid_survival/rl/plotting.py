from __future__ import annotations

import json
from html import escape
from pathlib import Path


def _evaluated_stage(record: dict) -> tuple[dict, str]:
    """Return the stage evaluated for training, across solo and team log schemas."""
    if "current" in record:
        stage = record["current"]
        return stage, str(stage.get("name", f"stage {record.get('training_stage', 0) + 1}"))
    if "stages" in record:
        index = int(record.get("training_stage", len(record["stages"]) - 1))
        if 0 <= index < len(record["stages"]):
            stage = record["stages"][index]
            return stage, str(stage.get("name", f"stage {index + 1}"))
    return record, str(record.get("name", "run"))


def _rate(stage: dict, metric: str) -> float | None:
    fields = ({"completion": ("completion_rate", "success_rate"),
               "survival": ("survival_fraction", "mean_alive_ship_time_fraction")})
    for field in fields[metric]:
        value = stage.get(field)
        if value is not None:
            return float(value)
    return None


def _focused_rate_graph(records: list[dict], run_dir: str | Path, x_field: str,
                        x_label: str, view: str) -> str:
    selected = (("completion", "Completion", "#2563eb"),
                ("survival", "Survival", "#dc2626"))
    if view != "both":
        selected = tuple(item for item in selected if item[0] == view)

    rows = []
    for record in records:
        stage, name = _evaluated_stage(record)
        rows.append((int(record[x_field]), name, stage))
    available = [(metric, label, color) for metric, label, color in selected
                 if any(_rate(stage, metric) is not None for _, _, stage in rows)]
    if not available:
        requested = " or ".join(metric for metric, _, _ in selected)
        raise SystemExit(f"evaluation log has no {requested} values")

    width, height = 1200, 620
    x0, y0, panel_w, panel_h = 95, 90, 1025, 420
    x_values = [step for step, _, _ in rows]
    min_x, max_x = min(x_values), max(x_values)
    span = max(1, max_x - min_x)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" data-view="{view}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="40" y="38" font-size="24" font-family="sans-serif">'
        f'Asteroids training progress — {escape(Path(run_dir).name)}</text>',
    ]
    for tick in range(5):
        rate = 1.0 - tick / 4
        y = y0 + tick * panel_h / 4
        parts.append(f'<path d="M{x0},{y:.1f} H{x0 + panel_w}" stroke="#e2e8f0"/>')
        parts.append(f'<text x="{x0 - 12}" y="{y + 4:.1f}" text-anchor="end" '
                     f'font-size="12" font-family="sans-serif">{rate:.0%}</text>')
    parts.append(f'<path d="M{x0},{y0} V{y0 + panel_h} H{x0 + panel_w}" '
                 'fill="none" stroke="#64748b"/>')

    # Stage changes explain deliberate drops when a policy advances to a harder rung.
    previous_name = None
    for step, name, _ in rows:
        if name == previous_name:
            continue
        x = x0 + panel_w * (step - min_x) / span
        if previous_name is not None:
            parts.append(f'<path d="M{x:.1f},{y0} V{y0 + panel_h}" stroke="#94a3b8" '
                         'stroke-width="1" stroke-dasharray="4 3"/>')
        parts.append(f'<text x="{x + 5:.1f}" y="{y0 + 18}" font-size="11" '
                     f'font-family="sans-serif" fill="#475569">{escape(name)}</text>')
        previous_name = name

    for metric_index, (metric, label, color) in enumerate(available):
        segments: list[list[str]] = [[]]
        for step, _, stage in rows:
            value = _rate(stage, metric)
            if value is None:
                if segments[-1]:
                    segments.append([])
                continue
            x = x0 + panel_w * (step - min_x) / span
            y = y0 + panel_h * (1.0 - min(max(value, 0.0), 1.0))
            segments[-1].append(f"{x:.1f},{y:.1f}")
        for points in segments:
            if points:
                parts.append(f'<polyline data-metric="{metric}" points="{" ".join(points)}" '
                             f'fill="none" stroke="{color}" stroke-width="3"/>')
        legend_x = x0 + metric_index * 190
        parts.append(f'<path d="M{legend_x},{y0 - 28} h28" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x + 36}" y="{y0 - 24}" font-size="13" '
                     f'font-family="sans-serif" fill="{color}">{label}</text>')

    parts.append(f'<text x="{x0 + panel_w / 2}" y="{y0 + panel_h + 38}" '
                 'text-anchor="middle" font-size="13" font-family="sans-serif">'
                 f'{x_label} {min_x:,}–{max_x:,}</text>')
    parts.append('</svg>\n')
    return "".join(parts)


def plot_progress(run_dir: str | Path, output: str | Path, view: str = "both") -> Path:
    """Write a dependency-free SVG of frozen held-out evaluation metrics."""
    source = Path(run_dir) / "evaluation.jsonl"
    if not source.exists():
        raise SystemExit(f"no evaluation log found at {source}")
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    if not records:
        raise SystemExit(f"evaluation log is empty: {source}")
    if view not in {"completion", "survival", "both", "all"}:
        raise ValueError(f"unknown graph view: {view}")
    curriculum = "stages" in records[0]
    x_field = ("environment_steps"
               if all(record.get("environment_steps") is not None for record in records)
               else "episode")
    x_label = "Environment decisions" if x_field == "environment_steps" else "Episodes"
    if view != "all":
        svg = _focused_rate_graph(records, run_dir, x_field, x_label, view)
        destination = Path(output)
        if destination.suffix.lower() != ".svg":
            destination = destination.with_suffix(".svg")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(svg, encoding="utf-8")
        return destination

    names = ([stage["name"] for stage in max(records, key=lambda r: len(r.get("stages", [])))["stages"]]
             if curriculum else [Path(run_dir).name])
    metrics = (("completion_rate", "Completion rate", 1.0),
               ("mean_wave", "Mean wave", None),
               ("mean_accuracy", "Accuracy", 1.0),
               ("mean_mean_wave_clear_time", "Mean wave clear time (s)", None))
    colors = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c")
    width, height = 1200, 800
    panels = []
    for metric_index, (field, label, ceiling) in enumerate(metrics):
        x0 = 80 + (metric_index % 2) * 580
        y0 = 80 + (metric_index // 2) * 360
        panel_w, panel_h = 500, 270
        series = []
        for stage_index, name in enumerate(names):
            values = []
            for record in records:
                if curriculum and stage_index >= len(record["stages"]):
                    continue
                stage = record["stages"][stage_index] if curriculum else record
                fallback = (stage.get("mean_survival_time", 0.0)
                            if field == "mean_mean_wave_clear_time" else 0.0)
                values.append((int(record[x_field]), float(stage.get(field, fallback))))
            series.append((name, values))
        maximum = ceiling or max(1.0, max(value for _, values in series for _, value in values))
        x_values = [int(record[x_field]) for record in records]
        min_x, max_x = min(x_values), max(x_values)
        span = max(1, max_x - min_x)
        panels.append(f'<text x="{x0}" y="{y0 - 22}" font-size="18" '
                      f'font-family="sans-serif">{escape(label)}</text>')
        panels.append(f'<path d="M{x0},{y0} V{y0 + panel_h} H{x0 + panel_w}" '
                      'fill="none" stroke="#64748b"/>')
        panels.append(f'<text x="{x0 - 10}" y="{y0 + 5}" text-anchor="end" '
                      f'font-size="12" font-family="sans-serif">{maximum:.2g}</text>')
        panels.append(f'<text x="{x0 - 10}" y="{y0 + panel_h}" text-anchor="end" '
                      'font-size="12" font-family="sans-serif">0</text>')
        for record in records:
            action = record.get("champion_action")
            if action not in {"improved", "rollback"}:
                continue
            x = x0 + panel_w * (int(record[x_field]) - min_x) / span
            color = "#059669" if action == "improved" else "#dc2626"
            panels.append(f'<path d="M{x:.1f},{y0} V{y0 + panel_h}" '
                          f'stroke="{color}" stroke-width="1" stroke-dasharray="4 3"/>')
        for stage_index, (name, values) in enumerate(series):
            points = []
            for step, value in values:
                x = x0 + panel_w * (step - min_x) / span
                y = y0 + panel_h * (1.0 - min(max(value / maximum, 0.0), 1.0))
                points.append(f"{x:.1f},{y:.1f}")
            color = colors[stage_index % len(colors)]
            panels.append(f'<polyline points="{" ".join(points)}" fill="none" '
                          f'stroke="{color}" stroke-width="2"/>')
            if metric_index == 0:
                panels.append(f'<text x="{x0 + 10 + (stage_index % 2) * 245}" '
                              f'y="{y0 + 18 + (stage_index // 2) * 18}" font-size="11" '
                              f'font-family="sans-serif" fill="{color}">{escape(name)}</text>')
        panels.append(f'<text x="{x0 + panel_w / 2}" y="{y0 + panel_h + 30}" '
                      'text-anchor="middle" font-size="12" font-family="sans-serif">'
                      f'{x_label} {min_x:,}–{max_x:,}</text>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>'
           f'<text x="40" y="38" font-size="24" font-family="sans-serif">'
           f'Asteroids training progress — {escape(Path(run_dir).name)}</text>'
           + "".join(panels) + "</svg>\n")
    destination = Path(output)
    if destination.suffix.lower() != ".svg":
        destination = destination.with_suffix(".svg")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(svg, encoding="utf-8")
    return destination
