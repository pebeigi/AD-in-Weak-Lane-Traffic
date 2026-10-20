# -*- coding: utf-8 -*-
"""
Shared filenames for RAU and SDPM trajectory-comparison figures.

Plots are mirrored into ``comparison_visualizations/`` beside the Lebanon scripts
(default) using::

    run_{run_id}_direction_{direction_id}_id_{vehicle_id}_{MODEL}.png

Examples::

    run_1_direction_2_id_4375_RAU.png
    run_1_direction_2_id_4375_SDPM.png
"""

from __future__ import annotations

import os


def comparison_plot_filename(run_id, direction_id, agent_id, model_slug: str) -> str:
    """Stable, filesystem-safe name for sorting and diffing."""
    r = "__" if run_id is None else str(run_id).replace(".", "_").replace("/", "_")
    d = "__" if direction_id is None else str(direction_id).replace(".", "_").replace("/", "_")
    return f"run_{r}_direction_{d}_id_{int(agent_id)}_{model_slug}.png"


def resolve_comparison_output_dir(cli_value: str | None, script_dir: str) -> str | None:
    """Interpret argparse value: None → default folder under ``script_dir``; \"\" → disabled."""
    if cli_value is not None and not str(cli_value).strip():
        return None
    if cli_value is None:
        return os.path.join(script_dir, "comparison_visualizations")
    if os.path.isabs(cli_value):
        return cli_value
    return os.path.join(script_dir, cli_value)


def save_figure_to_comparison(
    fig,
    comparison_dir: str | None,
    run_id,
    direction_id,
    agent_id,
    model_slug: str,
    bbox_inches: str | None = "tight",
) -> str | None:
    """Persist ``fig`` under ``comparison_dir`` if set. Returns written path or None."""
    if not comparison_dir:
        return None
    os.makedirs(comparison_dir, exist_ok=True)
    name = comparison_plot_filename(run_id, direction_id, agent_id, model_slug)
    path = os.path.join(comparison_dir, name)
    fig.savefig(path, bbox_inches=bbox_inches)
    return path
