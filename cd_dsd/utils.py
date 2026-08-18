import csv
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def setup_dirs(cfg):
    for d in [cfg.CHECKPOINT_DIR, cfg.LOG_DIR, cfg.PLOT_DIR, cfg.RESULTS_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)
    logger.info(f"Output root: {cfg.OUTPUT_DIR}")


def save_results_csv(results, path):
    if not results:
        return
    rows = []
    for r in results:
        row = {k: v for k, v in r.items()
               if k not in ("factor_attributions", "pred_original",
                            "pred_corrected", "label_cols", "vis_path")}
        row.update({f"attr_{k}": v
                    for k, v in r.get("factor_attributions", {}).items()})
        label_cols = r.get("label_cols", [])
        for c, label in enumerate(label_cols):
            row[f"pred_orig_{label}"] = round(r["pred_original"][c], 4)
            row[f"pred_corr_{label}"] = round(r["pred_corrected"][c], 4)
        rows.append(row)

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    logger.info(f"Results saved → {path}")


def print_summary(results, domain):
    if not results:
        return
    u_total    = np.mean([r["u_total"]         for r in results])
    u_domain   = np.mean([r["u_domain"]        for r in results])
    u_semantic = np.mean([r["u_semantic"]      for r in results])
    dom_frac   = np.mean([r["domain_fraction"] for r in results])

    logger.info(f"\n{'='*60}")
    logger.info(f"  CD-DSD Summary — Domain: {domain}")
    logger.info(f"  Samples diagnosed : {len(results)}")
    logger.info(f"  Mean U_total      : {u_total:.4f}")
    logger.info(f"  Mean U_domain     : {u_domain:.4f}")
    logger.info(f"  Mean U_semantic   : {u_semantic:.4f}")
    logger.info(f"  Mean domain frac  : {dom_frac:.1%}")
    logger.info(f"{'='*60}")
