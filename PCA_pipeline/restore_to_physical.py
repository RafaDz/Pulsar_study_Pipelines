from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class RestoreContext:
    avgprof: np.ndarray
    row_offset: np.ndarray
    weighted_mean_profile: np.ndarray
    onmask: np.ndarray
    offmask: np.ndarray


def collapse_duplicate_mjds(data: np.ndarray, mjd: np.ndarray):
    order = np.argsort(mjd, kind="mergesort")
    data_sorted = data[order]
    mjd_sorted = mjd[order]

    mjd_unique, inverse_unique, counts = np.unique(
        mjd_sorted,
        return_inverse=True,
        return_counts=True,
    )

    data_unique = np.zeros((len(mjd_unique), data.shape[1]), dtype=float)
    for i in range(len(mjd_unique)):
        data_unique[i] = data_sorted[inverse_unique == i].mean(axis=0)

    return data_sorted, mjd_sorted, order, data_unique, mjd_unique, inverse_unique, counts


def build_psrcelery_preprocessing(
    data: np.ndarray,
    onmask: np.ndarray,
    offmask: np.ndarray,
):
    nsub, nbin = data.shape

    avgprof = np.median(data, axis=0)
    template_subtracted = data - np.tile(avgprof, nsub).reshape((nsub, nbin))

    row_offset = np.mean(template_subtracted[:, onmask], axis=1)
    subdata0 = template_subtracted - row_offset[:, None]

    offrms = np.std(subdata0[:, offmask], axis=1)
    weights = (1.0 / offrms**2).reshape(-1, 1)
    weighted_mean_profile = np.sum(subdata0 * weights, axis=0) / np.sum(weights)

    subdata = subdata0 - weighted_mean_profile
    return avgprof, row_offset, weighted_mean_profile, subdata


def build_restore_context(
    raw_data: np.ndarray,
    mjd: np.ndarray,
    onmask: np.ndarray,
    offmask: np.ndarray,
) -> RestoreContext:
    _, _, order, data_unique, _, inverse_unique, _ = collapse_duplicate_mjds(raw_data, mjd)

    avgprof, row_offset_unique, weighted_mean_profile, _ = build_psrcelery_preprocessing(
        data_unique,
        onmask,
        offmask,
    )

    row_offset_sorted = row_offset_unique[inverse_unique]
    row_offset = np.empty_like(row_offset_sorted)
    row_offset[order] = row_offset_sorted

    return RestoreContext(
        avgprof=avgprof,
        row_offset=row_offset,
        weighted_mean_profile=weighted_mean_profile,
        onmask=onmask,
        offmask=offmask,
    )


def restore_single_profile_on(
    profile_on_transformed: np.ndarray,
    obs_full_index: int,
    ctx: RestoreContext,
) -> np.ndarray:
    return (
        profile_on_transformed
        + ctx.weighted_mean_profile[ctx.onmask]
        + ctx.row_offset[obs_full_index]
        + ctx.avgprof[ctx.onmask]
    )