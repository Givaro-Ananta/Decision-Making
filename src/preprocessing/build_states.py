"""Configurable hotspot-state interface; active-state thresholds are pending."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


def assign_hotspot_state(
    hotspot_count,
    frp_sum,
    config: Mapping[str, object] | None = None,
):
    """Assign No Activity or defer active classes until thresholds are approved.

    Until a state rule is approved, active observations return ``pd.NA`` rather
    than receiving invented Low/Moderate/High labels. A future configured
    classifier may be supplied as ``config['classifier']``.
    """
    count = np.asarray(hotspot_count)
    total_frp = np.asarray(frp_sum, dtype=float)
    if count.shape != total_frp.shape:
        raise ValueError("hotspot_count and frp_sum must have matching shapes")
    classifier = config.get("classifier") if config is not None else None
    if classifier is not None:
        result = np.asarray(classifier(count, total_frp), dtype=object)
        if result.shape != count.shape:
            raise ValueError("Configured state classifier returned an unexpected shape")
        return result.item() if result.ndim == 0 else result
    result = np.full(count.shape, pd.NA, dtype=object)
    inactive = count == 0
    result[inactive] = 0
    return result.item() if result.ndim == 0 else result
