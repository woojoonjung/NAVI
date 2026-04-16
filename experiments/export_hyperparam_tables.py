#!/usr/bin/env python3
"""
Build LaTeX + plain-text tables from masked-prediction and row-classification
hyperparam JSON logs (paired by run_id).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _variant_row_key(model_name: str, domain: str) -> str:
    """Strip navi_{domain}_ prefix; default checkpoint -> 'base'."""
    d = domain.lower()
    exact = f"navi_{d}"
    if model_name == exact:
        return "base"
    prefix = f"navi_{d}_"
    if model_name.startswith(prefix):
        return model_name[len(prefix) :] or "base"
    return model_name


def _escape_tex(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def _fmt_mp_cell(v) -> str:
    if v is None:
        return "---"
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return str(v)


def _fmt_rc_cell(mean, std) -> str:
    if mean is None or std is None:
        return "---"
    try:
        return f"{float(mean):.4f} $\\pm$ {float(std):.4f}"
    except (TypeError, ValueError):
        return "---"


def _fmt_rc_cell_txt(mean, std) -> str:
    if mean is None or std is None:
        return "---"
    try:
        return f"{float(mean):.4f} +/- {float(std):.4f}"
    except (TypeError, ValueError):
        return "---"


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _mp_variant_dict(mp_doc: dict) -> dict:
    if isinstance(mp_doc.get("variants"), dict):
        return mp_doc["variants"]
    skip = {"experiment", "domain", "run_id", "checkpoint_epoch"}
    return {k: v for k, v in mp_doc.items() if k not in skip and isinstance(v, dict)}


def collect_variant_keys(mp_movie: dict, mp_product: dict) -> list[str]:
    vm = _mp_variant_dict(mp_movie)
    vp = _mp_variant_dict(mp_product)
    keys = set()
    for model_name in vm:
        keys.add(_variant_row_key(model_name, "Movie"))
    for model_name in vp:
        keys.add(_variant_row_key(model_name, "Product"))

    def sort_key(k: str):
        return (0, "") if k == "base" else (1, k)

    return sorted(keys, key=sort_key)


def build_masked_prediction_tables(
    mp_movie: dict,
    mp_product: dict,
) -> tuple[str, str]:
    vm = _mp_variant_dict(mp_movie)
    vp = _mp_variant_dict(mp_product)
    rows = collect_variant_keys(mp_movie, mp_product)

    def lookup(domain: str, row_key: str, metric: str):
        d = domain.lower()
        full = f"navi_{d}" if row_key == "base" else f"navi_{d}_{row_key}"
        src = vm if domain == "Movie" else vp
        ent = src.get(full)
        if not ent:
            return None
        return ent.get(metric)

    # --- LaTeX ---
    lines_tex = []
    lines_tex.append("% Requires: \\usepackage{booktabs}")
    lines_tex.append("\\begin{tabular}{l|cc|cc}")
    lines_tex.append("\\toprule")
    lines_tex.append(
        " & \\multicolumn{2}{c|}{Movie} & \\multicolumn{2}{c}{Product} \\\\"
    )
    lines_tex.append("Variant & Header & Value & Header & Value \\\\")
    lines_tex.append("\\midrule")
    for rk in rows:
        label = "(base)" if rk == "base" else rk
        hm = lookup("Movie", rk, "header_accuracy")
        vm_ = lookup("Movie", rk, "value_accuracy")
        hp = lookup("Product", rk, "header_accuracy")
        vp_ = lookup("Product", rk, "value_accuracy")
        lines_tex.append(
            f"{_escape_tex(label)} & {_fmt_mp_cell(hm)} & {_fmt_mp_cell(vm_)} & "
            f"{_fmt_mp_cell(hp)} & {_fmt_mp_cell(vp_)} \\\\"
        )
    lines_tex.append("\\bottomrule")
    lines_tex.append("\\end{tabular}")
    tex = "\n".join(lines_tex)

    # --- TXT ---
    w_var = max(len("(base)"), max((len(rk if rk != "base" else "(base)") for rk in rows), default=8))
    colw = 14
    sep = " | "
    header = (
        f"{'Variant':<{w_var}}{sep}"
        f"{'M_Header':>{colw}}{sep}{'M_Value':>{colw}}{sep}"
        f"{'P_Header':>{colw}}{sep}{'P_Value':>{colw}}"
    )
    bar = "-" * len(header)
    lines_txt = [header, bar]
    for rk in rows:
        label = "(base)" if rk == "base" else rk
        hm = lookup("Movie", rk, "header_accuracy")
        vm_ = lookup("Movie", rk, "value_accuracy")
        hp = lookup("Product", rk, "header_accuracy")
        vp_ = lookup("Product", rk, "value_accuracy")
        lines_txt.append(
            f"{label:<{w_var}}{sep}"
            f"{_fmt_mp_cell(hm):>{colw}}{sep}{_fmt_mp_cell(vm_):>{colw}}{sep}"
            f"{_fmt_mp_cell(hp):>{colw}}{sep}{_fmt_mp_cell(vp_):>{colw}}"
        )
    txt = "\n".join(lines_txt)
    return tex, txt


CLASSIFIERS = ["xgboost", "catboost", "lr", "tabpfn"]


def build_row_classification_tables(rc_movie: dict, rc_product: dict) -> tuple[str, str]:
    """JSON shape: { variant_name: { 'xgboost_movie': {mean, std, scores}, ... } }"""

    def row_keys(data: dict, domain: str) -> set[str]:
        out = set()
        for model_name in data:
            if model_name.startswith("_"):
                continue
            out.add(_variant_row_key(model_name, domain))
        return out

    keys = row_keys(rc_movie, "Movie") | row_keys(rc_product, "Product")

    def sort_key(k: str):
        return (0, "") if k == "base" else (1, k)

    rows = sorted(keys, key=sort_key)

    def lookup(domain: str, row_key: str, clf: str):
        d = domain.lower()
        full = f"navi_{d}" if row_key == "base" else f"navi_{d}_{row_key}"
        src = rc_movie if domain == "Movie" else rc_product
        ent = src.get(full)
        if not ent:
            return None, None
        key = f"{clf}_{d}"
        block = ent.get(key)
        if not block:
            return None, None
        return block.get("mean"), block.get("std")

    # LaTeX: 1 + 4 + 4 columns
    tex_lines = []
    tex_lines.append("% Requires: \\usepackage{booktabs}")
    tex_lines.append("\\begin{tabular}{l|" + "cccc|" + "cccc" + "}")
    tex_lines.append("\\toprule")
    tex_lines.append(
        " & \\multicolumn{4}{c|}{Movie} & \\multicolumn{4}{c}{Product} \\\\"
    )
    sub = " & ".join(
        [f"\\multicolumn{{1}}{{c}}{{\\small {c}}}" for c in CLASSIFIERS]
        + [f"\\multicolumn{{1}}{{c}}{{\\small {c}}}" for c in CLASSIFIERS]
    )
    tex_lines.append("Variant & " + sub + " \\\\")
    tex_lines.append("\\midrule")
    for rk in rows:
        label = "(base)" if rk == "base" else rk
        cells = []
        for dom in ("Movie", "Product"):
            for clf in CLASSIFIERS:
                m, s = lookup(dom, rk, clf)
                cells.append(_fmt_rc_cell(m, s))
        tex_lines.append(_escape_tex(label) + " & " + " & ".join(cells) + " \\\\")
    tex_lines.append("\\bottomrule")
    tex_lines.append("\\end{tabular}")
    tex = "\n".join(tex_lines)

    w_var = max(12, max(len(rk if rk != "base" else "(base)") for rk in rows) if rows else 8)
    cw = 22
    # Two header rows: domain bands, then classifier names
    txt_lines = [
        f"{'':{w_var}} | {'Movie':^{len(CLASSIFIERS) * (cw + 3) - 3}} | {'Product':^{len(CLASSIFIERS) * (cw + 3) - 3}}",
        f"{'Variant':<{w_var}}"
        + " | " + " | ".join(f"{c:^{cw}}" for c in CLASSIFIERS)
        + " | " + " | ".join(f"{c:^{cw}}" for c in CLASSIFIERS),
        "-" * 200,
    ]
    for rk in rows:
        label = "(base)" if rk == "base" else rk
        parts = [f"{label:<{w_var}}"]
        for dom in ("Movie", "Product"):
            for clf in CLASSIFIERS:
                m, s = lookup(dom, rk, clf)
                parts.append(_fmt_rc_cell_txt(m, s).rjust(cw))
        txt_lines.append(" | ".join(parts))
    txt = "\n".join(txt_lines)

    return tex, txt


def main():
    p = argparse.ArgumentParser(description="Export hyperparam LaTeX/TXT tables from JSON logs")
    p.add_argument("--run_id", required=True, help="Shared run id used in JSON filenames")
    p.add_argument(
        "--logs_dir",
        type=Path,
        default=Path("experiments/logs"),
        help="Directory containing JSON files",
    )
    p.add_argument(
        "--out_prefix",
        type=Path,
        default=None,
        help="Output path prefix (default: logs_dir/hyperparam_tables_{run_id})",
    )
    args = p.parse_args()
    rid = args.run_id
    logd = args.logs_dir
    if not logd.is_absolute():
        logd = _REPO_ROOT / logd
    prefix = args.out_prefix
    if prefix is None:
        prefix = logd / f"hyperparam_tables_{rid}"
    else:
        if not prefix.is_absolute():
            prefix = _REPO_ROOT / prefix

    mp_m = logd / f"masked_prediction_hyperparam_sensitivity_movie_{rid}.json"
    mp_p = logd / f"masked_prediction_hyperparam_sensitivity_product_{rid}.json"
    rc_m = logd / f"row_classification_hyperparam_sensitivity_movie_{rid}.json"
    rc_p = logd / f"row_classification_hyperparam_sensitivity_product_{rid}.json"

    missing = [str(x) for x in (mp_m, mp_p, rc_m, rc_p) if not Path(x).exists()]
    if missing:
        print("Missing JSON files:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        sys.exit(1)

    mp_movie = load_json(mp_m)
    mp_product = load_json(mp_p)
    rc_movie = load_json(rc_m)
    rc_product = load_json(rc_p)

    tex_mp, txt_mp = build_masked_prediction_tables(mp_movie, mp_product)
    tex_rc, txt_rc = build_row_classification_tables(rc_movie, rc_product)

    for ext, body in ((".tex", tex_mp), (".txt", txt_mp)):
        path = Path(str(prefix) + "_masked_prediction" + ext)
        path.parent.mkdir(parents=True, exist_ok=True)
        if ext == ".tex":
            header = (
                "% Auto-generated hyperparameter masked-prediction table\n"
                f"% run_id: {rid}\n\n"
            )
        else:
            header = (
                "# Auto-generated hyperparameter masked-prediction table\n"
                f"# run_id: {rid}\n\n"
            )
        path.write_text(header + body + "\n", encoding="utf-8")
        print(f"Wrote {path}")

    for ext, body in ((".tex", tex_rc), (".txt", txt_rc)):
        path = Path(str(prefix) + "_row_classification" + ext)
        path.parent.mkdir(parents=True, exist_ok=True)
        if ext == ".tex":
            header = (
                "% Auto-generated hyperparameter row-classification table\n"
                f"% run_id: {rid}\n\n"
            )
        else:
            header = (
                "# Auto-generated hyperparameter row-classification table\n"
                f"# run_id: {rid}\n\n"
            )
        path.write_text(header + body + "\n", encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
