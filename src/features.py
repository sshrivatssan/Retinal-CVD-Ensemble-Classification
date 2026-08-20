from math import pi

import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix


def feats_lbp(g_norm: np.ndarray, mask=None, P=16, R=2):
    """Extract a uniform Local Binary Pattern (LBP) histogram."""

    # Compute uniform LBP
    lbp = local_binary_pattern(
        (g_norm * 255).astype(np.uint8),
        P,
        R,
        method="uniform"
    )

    n_bins = P + 2

    # Use pixels within the fundus mask
    vals = lbp[mask.astype(bool)] if mask is not None else lbp.ravel()

    # Compute normalized LBP histogram
    hist, _ = np.histogram(
        vals,
        bins=n_bins,
        range=(0, n_bins),
        density=True
    )

    return {
        f"lbp_u{P}_r{R}_{i}": float(hist[i])
        for i in range(n_bins)
    }


def feats_glcm(g_norm: np.ndarray, mask=None, levels=64):
    """Extract 14 Haralick features from GLCMs."""

    # Quantize image to 64 gray levels
    I = (g_norm * (levels - 1)).round().astype(np.uint8)

    # Set pixels outside the fundus mask to zero
    if mask is not None:
        I = I.copy()
        I[~mask.astype(bool)] = 0

    # GLCM distances and orientations
    dists = [1, 2]
    angles = [0, pi / 4, pi / 2, 3 * pi / 4]

    # Compute normalized symmetric GLCMs
    P = graycomatrix(
        I,
        distances=dists,
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True
    )

    def _safe_log2(x):
        """Compute log2 safely for zero-valued probabilities."""
        x = np.asarray(x, dtype=np.float64)

        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(x > 0, np.log2(x), 0.0)

        return out

    def _haralick_from_P(p):
        """Compute the 14 Haralick features from one GLCM."""

        p = p.astype(np.float64, copy=False)

        # Marginal probability distributions
        px = p.sum(axis=1)
        py = p.sum(axis=0)

        L = p.shape[0]
        idx = np.arange(L)

        # Marginal means
        ux = (idx * px).sum()
        uy = (idx * py).sum()

        # Marginal standard deviations
        sx = np.sqrt(
            ((idx - ux) ** 2 * px).sum() + 1e-12
        )
        sy = np.sqrt(
            ((idx - uy) ** 2 * py).sum() + 1e-12
        )

        i = idx[:, None]
        j = idx[None, :]

        # 1. Angular Second Moment (ASM)
        asm = (p ** 2).sum()

        # 2. Contrast
        contrast = (((i - j) ** 2) * p).sum()

        # 3. Correlation
        correlation = (
            (((i - ux) * (j - uy) * p).sum()) / (sx * sy)
            if (sx > 0 and sy > 0)
            else 1.0
        )

        # 4. Variance
        variance = (((i - ux) ** 2) * p).sum()

        # 5. Inverse Difference Moment (IDM)
        idm = (p / (1.0 + (i - j) ** 2)).sum()

        # Sum and difference probability distributions
        p_sum = np.zeros(2 * L - 1)
        p_diff = np.zeros(L)

        for ii in range(L):
            for jj in range(L):
                val = p[ii, jj]
                p_sum[ii + jj] += val
                p_diff[abs(ii - jj)] += val

        # Entropy terms
        HXY = -(p * _safe_log2(p)).sum()
        HX = -(px * _safe_log2(px)).sum()
        HY = -(py * _safe_log2(py)).sum()

        pxpy = px[:, None] * py[None, :]
        HXY1 = -(p * _safe_log2(pxpy)).sum()
        HXY2 = -(pxpy * _safe_log2(pxpy)).sum()

        k = np.arange(2 * L - 1)

        # 6. Sum Average
        sum_avg = (k * p_sum).sum()

        # Sum entropy is also used in the Sum Variance calculation
        sum_entropy = -(p_sum * _safe_log2(p_sum)).sum()

        # 7. Sum Variance
        sum_var = ((k - sum_entropy) ** 2 * p_sum).sum()

        # 8. Sum Entropy
        # Stored in sum_entropy above

        # 9. Entropy
        entropy = HXY

        d = np.arange(L)
        dm = (d * p_diff).sum()

        # 10. Difference Variance
        diff_var = (((d - dm) ** 2) * p_diff).sum()

        # 11. Difference Entropy
        diff_entropy = -(p_diff * _safe_log2(p_diff)).sum()

        # 12. Information Measure of Correlation I (IMC1)
        imc1 = (
            (HXY - HXY1) / max(HX, HY)
            if max(HX, HY) > 0
            else 0.0
        )

        # 13. Information Measure of Correlation II (IMC2)
        imc2 = np.sqrt(
            max(
                0.0,
                1.0 - np.exp(-2.0 * (HXY2 - HXY))
            )
        )

        # 14. Maximal Correlation Coefficient (MCC)
        Q = np.zeros((L, L))

        nz_px = np.where(px == 0, 1, px)
        nz_py = np.where(py == 0, 1, py)

        for ii in range(L):
            for jj in range(L):
                num = (
                    p[ii, :] * p[jj, :] / nz_py
                ).sum()

                Q[ii, jj] = num / nz_px[ii]

        try:
            eig = np.linalg.eigvals(Q)
            eig = np.sort(np.real(eig))

            mcc = (
                np.sqrt(max(0.0, eig[-2]))
                if len(eig) >= 2
                else 0.0
            )

        except np.linalg.LinAlgError:
            mcc = 0.0

        return {
            "asm": asm,
            "contrast": contrast,
            "correlation": correlation,
            "variance": variance,
            "idm": idm,
            "sum_avg": sum_avg,
            "sum_var": sum_var,
            "sum_entropy": sum_entropy,
            "entropy": entropy,
            "diff_var": diff_var,
            "diff_entropy": diff_entropy,
            "imc1": imc1,
            "imc2": imc2,
            "mcc": mcc
        }

    # Extract features from each distance-orientation GLCM
    feats = []

    for di in range(len(dists)):
        for ai in range(len(angles)):
            feats.append(
                _haralick_from_P(P[:, :, di, ai])
            )

    # Aggregate each Haralick feature using mean and standard deviation
    agg = {}

    for k in feats[0].keys():
        vals = np.array([f[k] for f in feats])

        agg[f"glcm_{k}_mean"] = float(vals.mean())
        agg[f"glcm_{k}_std"] = float(vals.std())

    return agg