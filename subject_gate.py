"""Gate matting alpha to the primary person (YOLO) so background people / gear are dropped."""

from __future__ import annotations

from typing import Any

import numpy as np


def _expand_xyxy(
    xyxy: np.ndarray,
    *,
    w: int,
    h: int,
    pad_frac: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    span = max(bw, bh, w * 0.08, h * 0.08)
    pm = float(pad_frac) * span
    xi1 = int(np.clip(np.floor(x1 - pm), 0, w - 1))
    yi1 = int(np.clip(np.floor(y1 - pm), 0, h - 1))
    xi2 = int(np.clip(np.ceil(x2 + pm), 0, w))
    yi2 = int(np.clip(np.ceil(y2 + pm), 0, h))
    return xi1, yi1, xi2, yi2


def _box_center(xyxy: np.ndarray) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _same_xyxy(a: np.ndarray, b: np.ndarray, *, tol: float = 2.0) -> bool:
    return bool(np.max(np.abs(a - b)) < tol)


def _pick_primary_xyxy(
    pool: list[tuple[float, np.ndarray, float]],
    *,
    w: int,
    h: int,
    center_bias_area_ratio: float = 0.76,
) -> np.ndarray:
    """
    Prefer the largest box; if another box is within center_bias_area_ratio of that area,
    pick whoever is closer to the frame center (typical main subject in phone gym shots).
    """
    work = sorted(pool, key=lambda t: t[0], reverse=True)
    max_area = work[0][0]
    if max_area <= 1.0:
        return work[0][1]
    cx_img, cy_img = 0.5 * w, 0.48 * h
    close_idx = [i for i, t in enumerate(work) if t[0] >= max_area * center_bias_area_ratio]
    if len(close_idx) < 2:
        return work[0][1]

    def dist2(i: int) -> float:
        bx, by = _box_center(work[i][1])
        dx, dy = bx - cx_img, by - cy_img
        return dx * dx + dy * dy

    best = min(close_idx, key=dist2)
    return work[best][1]


def primary_person_soft_mask_bgr(
    bgr: np.ndarray,
    yolo_model: Any,
    *,
    pad_frac: float = 0.32,
    other_suppress_pad_frac: float = 0.26,
    blur_sigma: float = 4.0,
    detect_conf: float = 0.18,
    primary_min_conf: float = 0.24,
    center_bias_area_ratio: float = 0.76,
    others_dilate_iter: int = 2,
) -> np.ndarray:
    """
    Primary COCO person (largest, or most centered when two bodies are similar size), padded.
    All other persons (even low-conf partial detections) are carved out with extra dilation so
    limbs / seated people behind the athlete are suppressed.

    If no detections, returns all-ones (no gating).
    """
    import cv2

    h, w = bgr.shape[:2]
    rgb = bgr[:, :, ::-1]
    results = yolo_model(rgb, verbose=False, conf=float(detect_conf), iou=0.5)
    scored: list[tuple[float, np.ndarray, float]] = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        for b in r.boxes:
            cid = int(b.cls[0].item())
            if cid != 0:
                continue
            conf = float(b.conf[0].item()) if b.conf is not None else 1.0
            if conf < detect_conf:
                continue
            xyxy = b.xyxy[0].detach().cpu().numpy().astype(np.float64)
            x1, y1, x2, y2 = xyxy
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            area = bw * bh
            scored.append((area, xyxy, conf))
    if not scored:
        return np.ones((h, w), dtype=np.float32)

    strong = [t for t in scored if t[2] >= primary_min_conf]
    pool = strong if strong else scored
    primary = _pick_primary_xyxy(
        pool, w=w, h=h, center_bias_area_ratio=center_bias_area_ratio
    )

    xi1, yi1, xi2, yi2 = _expand_xyxy(primary, w=w, h=h, pad_frac=pad_frac)
    primary_hard = np.zeros((h, w), dtype=np.float32)
    primary_hard[yi1:yi2, xi1:xi2] = 1.0

    others_hard = np.zeros((h, w), dtype=np.float32)
    for _area, xyxy, _cf in scored:
        if _same_xyxy(xyxy, primary):
            continue
        ox1, oy1, ox2, oy2 = _expand_xyxy(
            xyxy, w=w, h=h, pad_frac=other_suppress_pad_frac
        )
        others_hard[oy1:oy2, ox1:ox2] = 1.0

    if np.max(others_hard) > 0.0 and others_dilate_iter > 0:
        od = (others_hard > 0.5).astype(np.uint8) * 255
        rk = max(19, int(min(h, w) * 0.048) | 1)
        rker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rk, rk))
        od = cv2.dilate(od, rker, iterations=int(others_dilate_iter))
        others_hard = np.clip(od.astype(np.float32) / 255.0, 0.0, 1.0)

    k = max(3, int(blur_sigma * 2) | 1)
    p = cv2.GaussianBlur(primary_hard, (k, k), sigmaX=blur_sigma, sigmaY=blur_sigma)
    if np.max(others_hard) > 0.0:
        # Tighter blur on "others" so suppression stays sharp — reduces semi-transparent bg people.
        sig_o = float(blur_sigma) * 0.58
        ko = max(3, int(sig_o * 2) | 1)
        o = cv2.GaussianBlur(others_hard, (ko, ko), sigmaX=sig_o, sigmaY=sig_o)
        gate = p * np.clip(1.0 - o, 0.0, 1.0)
    else:
        gate = p
    return np.clip(gate, 0.0, 1.0)


def primary_person_seg_soft_mask_bgr(
    bgr: np.ndarray,
    yolo_model: Any,
    *,
    detect_conf: float = 0.26,
    edge_blur_sigma: float = 1.05,
    other_suppress_strength: float = 0.98,
    imgsz: int | None = None,
    morph_close_scale: float = 0.017,
    equip_dilate_iter: int = 1,
) -> np.ndarray:
    """
    Pixel-accurate person mask from YOLO-seg (largest COCO person, others subtracted).
    Morphological close reduces holes (e.g. between legs) where gym bleeds through; a small
    dilate keeps hands / bar ends that sit outside the raw seg silhouette.
    Falls back to padded box gate if the model has no mask head or no detections.
    """
    import cv2

    h, w = bgr.shape[:2]
    rgb = bgr[:, :, ::-1]
    infer_sz = int(imgsz) if imgsz is not None else int(min(1280, max(640, max(h, w))))
    results = yolo_model(
        rgb,
        verbose=False,
        conf=float(detect_conf),
        iou=0.45,
        classes=[0],
        imgsz=infer_sz,
    )

    masks_hw: list[tuple[float, np.ndarray]] = []
    for r in results:
        boxes = getattr(r, "boxes", None)
        masks = getattr(r, "masks", None)
        if boxes is None or len(boxes) == 0:
            continue
        if masks is None:
            continue
        data = getattr(masks, "data", None)
        if data is None:
            continue
        md = data.detach().float().cpu().numpy()
        if md.ndim != 3:
            continue
        nh, nw = int(md.shape[1]), int(md.shape[2])
        n_inst = int(md.shape[0])
        for i in range(n_inst):
            if int(boxes.cls[i].item()) != 0:
                continue
            if boxes.conf is not None and float(boxes.conf[i].item()) < detect_conf:
                continue
            mi = md[i]
            if nh != h or nw != w:
                mi = cv2.resize(mi, (w, h), interpolation=cv2.INTER_LINEAR)
            mi = np.clip(mi.astype(np.float32), 0.0, 1.0)
            masks_hw.append((float(mi.sum()), mi))

    if not masks_hw:
        return primary_person_soft_mask_bgr(bgr, yolo_model, detect_conf=detect_conf)

    masks_hw.sort(key=lambda t: t[0], reverse=True)
    max_area = masks_hw[0][0]
    close_idx = [j for j, t in enumerate(masks_hw) if t[0] >= max_area * 0.72]
    if len(close_idx) >= 2:
        cx_i, cy_i = 0.5 * w, 0.48 * h

        def cen_score(mm: np.ndarray) -> float:
            ys, xs = np.where(mm > 0.45)
            if ys.size < 8:
                return 1e12
            mx, my = float(xs.mean()), float(ys.mean())
            return (mx - cx_i) ** 2 + (my - cy_i) ** 2

        pi = int(min(close_idx, key=lambda j: cen_score(masks_hw[j][1])))
    else:
        pi = 0
    primary = masks_hw[pi][1].copy()

    others = np.zeros((h, w), dtype=np.float32)
    for j, (_ar, mm) in enumerate(masks_hw):
        if j == pi:
            continue
        others = np.maximum(others, mm)

    gate = primary * (1.0 - float(other_suppress_strength) * others)
    gate = np.clip(gate, 0.0, 1.0)
    hard = (gate > 0.48).astype(np.uint8) * 255
    kc = max(5, int(min(h, w) * float(morph_close_scale)) | 1)
    if kc % 2 == 0:
        kc += 1
    ker_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc))
    hard = cv2.morphologyEx(hard, cv2.MORPH_CLOSE, ker_c, iterations=1)
    if equip_dilate_iter > 0:
        kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        hard = cv2.dilate(hard, kd, iterations=int(equip_dilate_iter))
    gate_u = hard.astype(np.float32) / 255.0
    kb = max(3, int(edge_blur_sigma * 2) | 1)
    gate_u = cv2.GaussianBlur(
        gate_u, (kb, kb), sigmaX=edge_blur_sigma, sigmaY=edge_blur_sigma
    )
    return np.clip(gate_u.astype(np.float32), 0.0, 1.0)
