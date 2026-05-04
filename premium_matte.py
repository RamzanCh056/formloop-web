"""Premium refinements before GIF: smoother alpha edges, red/gym despill, optional logo with soft shadow."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _alpha_grad_mag(a: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(a.astype(np.float64))
    return np.hypot(gx, gy)


def refine_alpha_edges(a: np.ndarray, *, blur_sigma: float = 0.28) -> np.ndarray:
    """Slight Gaussian mix on alpha only along high-gradient boundary; preserves solid interior & hair tails."""
    import cv2

    a = np.clip(a.astype(np.float64), 0.0, 1.0)
    g = _alpha_grad_mag(a)
    gp = g[g > 1e-6]
    g_thr = float(np.percentile(gp, 97)) if gp.size else 1e-3
    edge_w = np.clip(g / (g_thr * 1.8 + 1e-6), 0.0, 1.0)
    band = (a > 0.02) & (a < 0.995)
    interior = a > 0.92
    # Light weight — strong edge blur smears gym into the silhouette on motion frames.
    w = edge_w * band.astype(np.float64) * (~interior).astype(np.float64) * 0.22
    ab = cv2.GaussianBlur(a, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    out = (1.0 - w) * a + w * ab
    return np.clip(out, 0.0, 1.0)


def boost_alpha_thin_specular(bgr: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Lift alpha on bright, low-saturation regions (specular bar metal) where RVM often under-matts."""
    import cv2

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    v = np.std(bgr.astype(np.float32), axis=-1)
    a = np.clip(a.astype(np.float64), 0.0, 1.0)
    out = a.copy()
    mask = (gray > 178.0) & (v < 36.0) & (a > 0.10) & (a < 0.72)
    out[mask] = np.clip(out[mask] * 1.12 + 0.04, 0.0, 1.0)
    # Mid-gray metal: require existing matte so gym floor never floods.
    mid = (
        (gray > 115.0)
        & (gray < 205.0)
        & (v < 48.0)
        & (a > 0.12)
        & (a < 0.65)
    )
    out[mid] = np.clip(out[mid] * 1.22 + 0.08, 0.0, 1.0)
    out[mid] = np.maximum(out[mid], 0.22)
    return out


def boost_alpha_dark_equipment(bgr: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Lift alpha on darker rubber/metal plates and black dumbbells (mid alpha, low variance)."""
    import cv2

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    v = np.std(bgr.astype(np.float32), axis=-1)
    a = np.clip(a.astype(np.float64), 0.0, 1.0)
    out = a.copy()
    mask = (gray > 8.0) & (gray < 158.0) & (v < 54.0) & (a > 0.06) & (a < 0.78)
    out[mask] = np.clip(out[mask] * 1.18 + 0.06, 0.0, 1.0)
    very_dark = (gray <= 42.0) & (v < 48.0) & (a > 0.05) & (a < 0.82)
    out[very_dark] = np.clip(out[very_dark] * 1.12 + 0.07, 0.0, 1.0)
    return out


def boost_alpha_near_body_dark(
    bgr: np.ndarray, a: np.ndarray, *, ref_alpha: np.ndarray
) -> np.ndarray:
    """Recover alpha for very dark handheld objects touching the subject (RVM often kills them)."""
    import cv2

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    v = np.std(bgr.astype(np.float32), axis=-1)
    a = np.clip(a.astype(np.float64), 0.0, 1.0)
    ref = np.clip(ref_alpha.astype(np.float64), 0.0, 1.0)
    body = ((ref > 0.24).astype(np.uint8) * 255)
    k = max(11, int(min(bgr.shape[0], bgr.shape[1]) * 0.042) | 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    near = cv2.dilate(body, ker, iterations=1)
    near_f = (near.astype(np.float32) / 255.0) > 0.5
    dark = (gray < 105.0) & (gray > 4.0)
    low_a = (a < 0.32) & (a >= 0.04)
    mask = near_f & dark & low_a & (v < 48.0)
    out = a.copy()
    out[mask] = np.clip(np.maximum(out[mask] * 1.22 + 0.08, 0.38), 0.0, 1.0)
    return out


def boost_alpha_proximity_fill(
    bgr: np.ndarray, a: np.ndarray, *, ref_alpha: np.ndarray
) -> np.ndarray:
    """Fill near-zero alpha inside a thick dilation of RVM+gate foreground (reaches bar ends)."""
    import cv2

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    v = np.std(bgr.astype(np.float32), axis=-1)
    a = np.clip(a.astype(np.float64), 0.0, 1.0)
    ref = np.clip(ref_alpha.astype(np.float64), 0.0, 1.0)
    seed = ((ref > 0.18).astype(np.uint8) * 255)
    k = max(15, int(min(bgr.shape[0], bgr.shape[1]) * 0.068) | 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    prox = cv2.dilate(seed, ker, iterations=1)
    pf = (prox.astype(np.float32) / 255.0) > 0.5
    cand = (
        pf
        & (a < 0.32)
        & (gray > 12.0)
        & (gray < 222.0)
        & (v < 40.0)
    )
    out = a.copy()
    out[cand] = np.maximum(out[cand], 0.40)
    return out


def boost_handheld_gear_alpha(
    bgr: np.ndarray, a: np.ndarray, *, ref_alpha: np.ndarray
) -> np.ndarray:
    """Combine bright-metal + dark-equipment recovery for bars and weights."""
    a = boost_alpha_thin_specular(bgr, a)
    a = boost_alpha_dark_equipment(bgr, a)
    a = boost_alpha_near_body_dark(bgr, a, ref_alpha=ref_alpha)
    # Fill weak/near-zero alpha pockets inside the expanded subject region (bar ends, plates).
    a = boost_alpha_proximity_fill(bgr, a, ref_alpha=ref_alpha)
    return a


def contract_alpha_halo(a: np.ndarray, *, erode_iters: int = 0) -> np.ndarray:
    """Optional 1px erode trims fringe but destroys thin bars; default is no erode (alpha-only smooth)."""
    import cv2

    x = np.clip(a.astype(np.float64), 0.0, 1.0)
    if erode_iters <= 0:
        return np.clip(
            cv2.GaussianBlur(x.astype(np.float32), (0, 0), sigmaX=0.45, sigmaY=0.45),
            0.0,
            1.0,
        )
    u = (x * 255.0).astype(np.uint8)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    er = cv2.erode(u, ker, iterations=int(erode_iters))
    out = er.astype(np.float64) / 255.0
    out = cv2.GaussianBlur(out.astype(np.float32), (0, 0), sigmaX=0.5, sigmaY=0.5).astype(
        np.float64
    )
    return np.clip(out, 0.0, 1.0)


def desaturate_fringe_bgr(bgr: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Kill colored spill on edges: grayscale blend + darken uncertain alpha (less gym bleed)."""
    import cv2

    af = np.clip(a.astype(np.float64), 0.0, 1.0)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    fringe = (af > 0.03) & (af < 0.58)
    out = bgr.astype(np.float64)
    if np.any(fringe):
        w = np.zeros_like(af)
        w[fringe] = np.clip(1.0 - (af[fringe] - 0.03) / 0.55, 0.22, 1.0) ** 0.82
        w3 = w[..., None]
        out = out * (1.0 - w3) + gray[..., None] * w3
    low = af < 0.42
    if np.any(low):
        dim = np.clip(af[low] / 0.42, 0.0, 1.0) ** 1.45
        out[low] *= dim[..., None]
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def despill_red_black_edges(rgb: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Reduce red bleed and harsh black fringing common on gym footage + black comps."""
    import cv2

    x = rgb.astype(np.float32)
    edge = ((a > 0.04) & (a < 0.96)).astype(np.float32)[..., None]
    r, gch, b = x[..., 0], x[..., 1], x[..., 2]
    mx = np.maximum(gch, b)
    red_ex = np.maximum(0.0, r - mx - 8.0)
    r2 = r - edge.squeeze() * 0.62 * red_ex
    lum = 0.299 * r2 + 0.587 * gch + 0.114 * b
    crush = np.maximum(0.0, 38.0 - lum) * edge.squeeze() * 0.42
    r2 = r2 - crush
    gch = gch - crush * 0.25
    b = b - crush * 0.25
    x2 = np.stack([r2, gch, b], axis=-1)
    return np.clip(x2, 0.0, 255.0).astype(np.uint8)


def _over_straight(
    drgb: np.ndarray, da: np.ndarray, srgb: np.ndarray, sa: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Porter-Duff source-over, straight RGB. Shapes HxWx3, HxW, HxWx3, HxW; values 0–1 float."""
    sa3 = sa[..., None]
    da3 = da[..., None]
    oa = sa + da * (1.0 - sa)
    oa3 = np.maximum(oa[..., None], 1e-6)
    orgb = (srgb * sa3 + drgb * da3 * (1.0 - sa3)) / oa3
    return np.clip(orgb, 0.0, 1.0), np.clip(oa, 0.0, 1.0)


def composite_logo_premium(
    bgr: np.ndarray,
    alpha_u8: np.ndarray,
    logo_path: Path,
    *,
    width_frac: float = 0.34,
    margin_bottom_px: int = 36,
    shadow_opacity: float = 0.42,
    shadow_blur: int = 23,
    shadow_offset: tuple[int, int] = (4, 6),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Composite a PNG logo (with alpha) centered near bottom + soft shadow (straight-alpha correct).
    Returns (bgr, alpha_u8) for PNG / ffmpeg.
    """
    import cv2

    logo = cv2.imread(str(logo_path), cv2.IMREAD_UNCHANGED)
    if logo is None:
        raise FileNotFoundError(logo_path)
    if logo.shape[2] == 3:
        logo = np.dstack([logo, np.full(logo.shape[:2], 255, dtype=np.uint8)])
    lh, lw = logo.shape[:2]
    H, W = bgr.shape[:2]
    target_w = max(32, int(W * float(width_frac)))
    scale = target_w / float(lw)
    target_h = max(16, int(lh * scale))
    logo_r = cv2.resize(logo, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    lg_bgr = logo_r[..., :3].astype(np.float32) / 255.0
    lg_a = logo_r[..., 3].astype(np.float32) / 255.0

    drgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    da = alpha_u8.astype(np.float32) / 255.0

    k = max(3, shadow_blur | 1)
    dil = cv2.dilate((lg_a * 255).astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(
        np.float32
    ) / 255.0
    sh = np.clip(cv2.GaussianBlur(dil, (k, k), 0) * shadow_opacity, 0.0, 1.0)
    ox, oy = shadow_offset
    x_logo = (W - target_w) // 2
    y_logo = H - target_h - margin_bottom_px
    x_logo = int(np.clip(x_logo, 0, max(0, W - target_w)))
    y_logo = int(np.clip(y_logo, 0, max(0, H - target_h)))
    xs = int(np.clip(x_logo + ox, 0, W - target_w))
    ys = int(np.clip(y_logo + oy, 0, H - target_h))
    xe, ye = xs + target_w, ys + target_h

    sh_full = np.zeros((H, W), dtype=np.float32)
    sh_full[ys:ye, xs:xe] = sh
    black = np.zeros((H, W, 3), dtype=np.float32)
    drgb, da = _over_straight(drgb, da, black, sh_full)

    xl1, yl1 = x_logo + target_w, y_logo + target_h
    lg_full_rgb = np.zeros((H, W, 3), dtype=np.float32)
    lg_full_a = np.zeros((H, W), dtype=np.float32)
    lg_full_rgb[y_logo:yl1, x_logo:xl1] = lg_bgr
    lg_full_a[y_logo:yl1, x_logo:xl1] = lg_a
    drgb, da = _over_straight(drgb, da, lg_full_rgb, lg_full_a)

    bgr_o = (np.clip(drgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    bgr_o = cv2.cvtColor(bgr_o, cv2.COLOR_RGB2BGR)
    return bgr_o, (np.clip(da, 0.0, 1.0) * 255.0).astype(np.uint8)


def temporal_consistency_alpha(
    bgr: np.ndarray,
    a: np.ndarray,
    a_prev: np.ndarray,
    *,
    bgr_prev: np.ndarray | None = None,
    gate: np.ndarray | None = None,
    lift_strength: float = 0.42,
) -> np.ndarray:
    """
    Recover one-frame alpha dropouts on handheld gear without dragging silhouettes.
    Only lifts uncertain, low-texture pixels when previous-frame alpha was confidently foreground.
    """
    import cv2

    cur = np.clip(a.astype(np.float64), 0.0, 1.0)
    prev = np.clip(a_prev.astype(np.float64), 0.0, 1.0)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    color_var = np.std(bgr.astype(np.float32), axis=-1)
    prev_warp = prev
    motion = None
    if bgr_prev is not None:
        prev_gray = cv2.cvtColor(bgr_prev, cv2.COLOR_BGR2GRAY).astype(np.float32)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=21,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        h, w = gray.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        map_x = grid_x - flow[..., 0]
        map_y = grid_y - flow[..., 1]
        prev_warp = cv2.remap(
            prev.astype(np.float32),
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(np.float64)
        motion = cv2.magnitude(flow[..., 0], flow[..., 1]).astype(np.float32)

    # Candidate flicker: previously solid-ish foreground, suddenly weak alpha now.
    flicker = (prev_warp > 0.38) & (cur < 0.36) & (cur > 0.01)
    # Keep this conservative: mostly darker / low-texture equipment regions.
    gear_like = (gray > 6.0) & (gray < 176.0) & (color_var < 62.0)
    cand = flicker & gear_like
    # Also allow recovery around previous-frame silhouette (helps motion-blurred plates).
    prev_fg = (prev_warp > 0.34).astype(np.uint8) * 255
    k = max(9, int(min(bgr.shape[0], bgr.shape[1]) * 0.02) | 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    near_prev = cv2.dilate(prev_fg, ker, iterations=1) > 0
    cand &= near_prev
    if motion is not None:
        # motion blur / fast movement often causes one-frame alpha collapse; recover there.
        cand &= (motion > 0.9)
    if gate is not None:
        cand &= (np.clip(gate.astype(np.float64), 0.0, 1.0) > 0.16)

    out = cur.copy()
    if np.any(cand):
        lifted = np.clip((1.0 - lift_strength) * cur[cand] + lift_strength * prev_warp[cand], 0.0, 1.0)
        out[cand] = np.maximum(out[cand], lifted)
        out[cand] = np.maximum(out[cand], 0.34)
    # Wider safety net: if current alpha is almost gone but previous was strong nearby, borrow some.
    prev_d = cv2.dilate((prev_warp > 0.34).astype(np.uint8) * 255, ker, iterations=1) > 0
    hard_drop = (cur < 0.10) & (prev_warp > 0.52) & prev_d
    if gate is not None:
        hard_drop &= (np.clip(gate.astype(np.float64), 0.0, 1.0) > 0.12)
    if np.any(hard_drop):
        out[hard_drop] = np.maximum(out[hard_drop], np.clip(prev_warp[hard_drop] * 0.76, 0.0, 1.0))

    # Stable deep background can still be lightly damped by previous frame.
    deep_bg = (out < 0.14) & (prev < 0.14)
    bg_mix = np.clip(0.90 * out + 0.10 * prev, 0.0, 1.0)
    out = np.where(deep_bg, bg_mix, out)
    return out


def finalize_transparent_bgra(
    bgr: np.ndarray,
    a: np.ndarray,
    *,
    gate: np.ndarray | None = None,
    alpha_cutoff: float = 24.0 / 255.0,
    gate_sharpen_eps: float = 0.038,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Straight RGBA for GIF: zero alpha → zero RGB. Aggressive fringe rejection for clean comps.
    """
    af = np.clip(a.astype(np.float64), 0.0, 1.0)
    if gate is not None:
        g = np.clip(gate.astype(np.float64), 0.0, 1.0)
        denom = max(1e-6, 1.0 - gate_sharpen_eps)
        g = np.clip((g - gate_sharpen_eps) / denom, 0.0, 1.0)
        af = np.minimum(af, g)
    af = np.power(np.clip(af, 0.0, 1.0), 1.12)
    af = np.clip((af - 0.022) / 0.978, 0.0, 1.0)
    af = np.where(af < float(alpha_cutoff), 0.0, af)
    premul = bgr.astype(np.float64) * af[..., None]
    bgr_out = np.zeros_like(bgr, dtype=np.float64)
    valid = af >= float(alpha_cutoff)
    div = np.maximum(af[valid, None], float(alpha_cutoff))
    bgr_out[valid] = np.clip(premul[valid] / div, 0.0, 255.0)
    au8 = (np.clip(af, 0.0, 1.0) * 255.0).astype(np.uint8)
    return bgr_out.astype(np.uint8), au8


def write_premium_rgba_sequence(
    fg_mp4: Path,
    alpha_mp4: Path,
    out_dir: Path,
    *,
    refine: bool = True,
    gear_alpha_boost: bool = False,
    yolo_model: object | None = None,
    yolo_use_seg_gate: bool = True,
    yolo_pad_frac: float = 0.34,
    yolo_inference_imgsz: int = 1280,
    logo_path: Path | None = None,
    logo_width_frac: float = 0.34,
    logo_margin_bottom: int = 36,
    temporal_consistency: bool = True,
) -> int:
    """Decode fg + alpha videos, refine, optional logo; write `out_dir / %06d.png` RGBA. Returns frame count."""
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    cap_f = cv2.VideoCapture(str(fg_mp4))
    cap_a = cv2.VideoCapture(str(alpha_mp4))
    if not cap_f.isOpened() or not cap_a.isOpened():
        raise RuntimeError("Could not open fg or alpha video")
    n = 0
    gate_ema: np.ndarray | None = None
    a_prev: np.ndarray | None = None
    fgr_prev: np.ndarray | None = None
    try:
        while True:
            rf, fgr = cap_f.read()
            ra, al = cap_a.read()
            if not rf or not ra:
                break
            if fgr.shape[:2] != al.shape[:2]:
                raise ValueError("fg and alpha frame size mismatch")
            a = cv2.cvtColor(al, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            gate_mul: np.ndarray | None = None
            if yolo_model is not None:
                from subject_gate import (
                    primary_person_seg_soft_mask_bgr,
                    primary_person_soft_mask_bgr,
                )

                if yolo_use_seg_gate:
                    gate_new = primary_person_seg_soft_mask_bgr(
                        fgr,
                        yolo_model,
                        imgsz=int(yolo_inference_imgsz),
                    ).astype(np.float32)
                    gate_exp = 1.02
                else:
                    gate_new = primary_person_soft_mask_bgr(
                        fgr, yolo_model, pad_frac=yolo_pad_frac
                    ).astype(np.float32)
                    gate_exp = 1.06
                if gate_ema is None:
                    gate_ema = gate_new.copy()
                else:
                    gate_ema = np.clip(
                        0.58 * gate_new.astype(np.float64)
                        + 0.42 * gate_ema.astype(np.float64),
                        0.0,
                        1.0,
                    ).astype(np.float32)
                gate_mul = np.power(
                    np.clip(gate_ema.astype(np.float64), 0.0, 1.0), gate_exp
                ).astype(np.float32)
                a = a.astype(np.float64) * gate_mul.astype(np.float64)
                a = np.where(gate_mul < 0.11, 0.0, a)
                a = np.clip(a, 0.0, 1.0).astype(np.float32)
            a_ref = np.clip(a.astype(np.float64), 0.0, 1.0).copy()
            if gear_alpha_boost:
                a = boost_handheld_gear_alpha(fgr, a, ref_alpha=a_ref)
            if refine:
                a = refine_alpha_edges(a)
                rgb = cv2.cvtColor(fgr, cv2.COLOR_BGR2RGB)
                rgb = despill_red_black_edges(rgb, a)
                fgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if gate_mul is not None:
                a = np.minimum(a, gate_mul.astype(np.float64))
            a = contract_alpha_halo(a)
            if temporal_consistency and (a_prev is not None):
                a = temporal_consistency_alpha(fgr, a, a_prev, bgr_prev=fgr_prev, gate=gate_mul)
            if logo_path is not None:
                au8_pre = (np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8)
                fgr, au8_pre = composite_logo_premium(
                    fgr,
                    au8_pre,
                    logo_path,
                    width_frac=logo_width_frac,
                    margin_bottom_px=logo_margin_bottom,
                )
                a = au8_pre.astype(np.float32) / 255.0
            fgr = desaturate_fringe_bgr(fgr, a)
            cutoff = (14.0 / 255.0) if gear_alpha_boost else (24.0 / 255.0)
            fgr, au8 = finalize_transparent_bgra(fgr, a, gate=gate_mul, alpha_cutoff=cutoff)
            a_prev = au8.astype(np.float64) / 255.0
            fgr_prev = fgr.copy()
            bgra = np.dstack([fgr[..., 0], fgr[..., 1], fgr[..., 2], au8])
            cv2.imwrite(str(out_dir / f"{n:06d}.png"), bgra)
            n += 1
    finally:
        cap_f.release()
        cap_a.release()
    if n == 0:
        raise RuntimeError("no frames written")
    return n
