from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def setup_matplotlib(font_size: int = 9):
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.titlesize": font_size + 1,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "legend.fontsize": font_size - 1,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )
    return plt


def save_figure(fig, output: str | Path, dpi: int = 300) -> None:
    output = ensure_parent(output)
    fig.savefig(output, dpi=dpi)


def format_float(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def latex_escape(text: Any) -> str:
    result = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result


def write_booktabs_table(
    path: str | Path,
    rows: list[dict[str, Any]],
    columns: list[str] | None = None,
    caption: str = "",
    label: str = "",
) -> None:
    path = ensure_parent(path)
    if not rows:
        raise ValueError("Cannot write LaTeX table with no rows.")
    columns = columns or list(rows[0].keys())
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{table}[t]\n\\centering\n")
        handle.write("\\begin{tabular}{" + "l" * len(columns) + "}\n")
        handle.write("\\toprule\n")
        handle.write(" & ".join(latex_escape(col) for col in columns) + " \\\\\n")
        handle.write("\\midrule\n")
        for row in rows:
            handle.write(" & ".join(latex_escape(row.get(col, "")) for col in columns) + " \\\\\n")
        handle.write("\\bottomrule\n\\end{tabular}\n")
        if caption:
            handle.write(f"\\caption{{{latex_escape(caption)}}}\n")
        if label:
            handle.write(f"\\label{{{latex_escape(label)}}}\n")
        handle.write("\\end{table}\n")
