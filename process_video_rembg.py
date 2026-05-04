#!/usr/bin/env python3
"""
Gym-oriented background removal — **rembg** (ONNX). **Default model is isnet-general-use**
(~180 MB download): salient foreground = person *and* held gear. **Not** the ~1 GB
**birefnet-general** unless you pass **--hq** (better edges, huge first-time download).

**First run:** We download weights **once in the main process** before parallel workers start,
so you do not get several processes all pulling the same file (slow / confusing progress).

**Speed:** `--fast` → **u2netp** (tiny ~5 MB, 320² inference — fastest; thin metal may suffer).
**--hq** → **birefnet-general** (~1 GB). **birefnet-general-lite** (~224 MB) is available via
`--model birefnet-general-lite`.

Avoid **u2net_human_seg** / **birefnet-portrait** for gym: they bias body-only and can drop props.

**Dumbbells “disappearing”:** Default **--alpha-ema 0** (temporal blend off). Raise only if you
prefer smoother edges over crisp fast motion.

**Post-pass:** **gym-refine** (light-UI + red bleed) is on by default; skip with **`--no-gym-refine`**
for faster runs. Temporal **max** alpha defaults **off** (`--temporal-max-radius 0`) — it stabilizes
thin bars but **ghosts** on fast limbs; use 1–2 only if needed. **GIF** defaults to **transparent**
(alpha); use **`--gif-opaque`** for white-backdrop GIF like before.

Install (RVM venv; Python 3.10–3.12 if wheels fail on 3.14):
  pip install \"rembg[cli]\" onnxruntime pillow

Run:
  python process_video_rembg.py --input input.mp4
  python process_video_rembg.py --fast
  python process_video_rembg.py --hq

If `python` is Homebrew/system without rembg, this script re-runs using ../venv/bin/python
when that venv exists (same as process_video.py).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _reexec_with_project_venv_if_needed() -> None:
    """Use rvm_backend/venv when rembg is not importable (venv python may symlink to Homebrew)."""
    try:
        import rembg  # noqa: F401
        return
    except ImportError:
        pass
    root = Path(__file__).resolve().parent
    venv_home = (root.parent / "venv").resolve()
    in_this_venv = Path(sys.prefix).resolve() == venv_home
    if in_this_venv:
        return
    for name in ("python3", "python"):
        vpy = venv_home / "bin" / name
        if vpy.is_file():
            os.execv(str(vpy), [str(vpy), str(Path(__file__).resolve()), *sys.argv[1:]])
            return


_reexec_with_project_venv_if_needed()

import argparse
import multiprocessing
import shutil
import subprocess
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO

# PyMatting (only with --alpha-matting) can emit noisy numerical warnings per frame.
warnings.filterwarnings("ignore", message=".*[Cc]holesky.*")
warnings.filterwarnings("ignore", message=".*positive-definiteness.*")
warnings.filterwarnings("ignore", message=".*PERFORMANCE WARNING.*")

# argparse sentinel: resolve default from --fast / --hq when --model omitted
_DEFAULT_MODEL_FLAG = object()


def _ensure_model_cached(model_name: str) -> None:
    """Ensure ONNX exists under rembg’s data dir; run once in parent before workers spawn.

    rembg’s ``download_models()`` always prints “Downloading model: …” even when the file
    is already cached (pooch skips the network). We call the session’s downloader directly
    and only say “download” when the file is missing or looks truncated.
    """
    from rembg.sessions import sessions

    sc = sessions.get(model_name)
    if sc is None:
        raise ValueError(f"Unknown rembg model: {model_name!r}")

    home = Path(sc.u2net_home()).expanduser()
    onnx_path = home / f"{sc.name()}.onnx"
    # IS-Net ~179MB; tiny models ~5MB — treat very small files as incomplete.
    min_bytes = 4 * 1024 * 1024
    cached = onnx_path.is_file() and onnx_path.stat().st_size >= min_bytes

    if cached:
        print(f"Using cached ONNX for {model_name!r}: {onnx_path}", flush=True)
    else:
        print(
            f"Downloading ONNX for {model_name!r} (one-time) into {home} …",
            flush=True,
        )
    sc.download_models()


def _video_fps(path: Path) -> float:
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    s = r.stdout.strip()
    if "/" in s:
        num, den = s.split("/")
        return float(num) / float(den)
    return float(s)


# --- multiprocessing (one ONNX session per worker; must be top-level for spawn) ---
_POOL_SESSION = None


def _pool_init(model_name: str) -> None:
    global _POOL_SESSION
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("ORT_INTRA_OP_NUM_THREADS", "2")
    from rembg import new_session

    _POOL_SESSION = new_session(model_name)


def _segment_frame(
    p: Path,
    dir_fg: Path,
    dir_a: Path,
    max_side: int,
    alpha_matting: bool,
    session,
) -> None:
    from rembg import remove
    from PIL import Image
    import numpy as np

    raw_full = Image.open(p).convert("RGB")
    ow, oh = raw_full.size
    long_edge = max(ow, oh)
    if max_side > 0 and long_edge > max_side:
        scale = max_side / long_edge
        nw, nh = max(1, int(ow * scale)), max(1, int(oh * scale))
        small = raw_full.resize((nw, nh), Image.Resampling.LANCZOS)
    else:
        small = raw_full

    buf = BytesIO()
    small.save(buf, format="PNG")
    try:
        if alpha_matting:
            out_b = remove(buf.getvalue(), session=session, alpha_matting=True)
        else:
            out_b = remove(buf.getvalue(), session=session)
    except TypeError:
        out_b = remove(buf.getvalue(), session=session)

    img = Image.open(BytesIO(out_b)).convert("RGBA")
    if img.size != (ow, oh):
        img = img.resize((ow, oh), Image.Resampling.LANCZOS)
    arr = np.asarray(img)
    rgb = arr[:, :, :3]
    a = arr[:, :, 3]
    stem = p.stem
    dir_fg.mkdir(parents=True, exist_ok=True)
    dir_a.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(dir_fg / f"{stem}.png")
    Image.fromarray(a, mode="L").save(dir_a / f"{stem}.png")


def _pool_worker_task(task: tuple) -> None:
    p_str, dfg, da, max_side, alpha_matting = task
    _segment_frame(Path(p_str), Path(dfg), Path(da), max_side, alpha_matting, _POOL_SESSION)


def _temporal_max_alpha_with_fg_fix(
    dir_in: Path,
    dir_fg: Path,
    dir_a: Path,
    paths_in_order: list[Path],
    radius: int,
) -> None:
    """Per-pixel max alpha over ±radius frames; copy original RGB into fg where alpha rises."""
    if radius <= 0 or len(paths_in_order) < 2:
        return
    import numpy as np
    from PIL import Image

    stems = [p.stem for p in paths_in_order]
    alphas_u8 = [np.asarray(Image.open(dir_a / f"{s}.png")) for s in stems]
    stack = np.stack(alphas_u8, axis=0).astype(np.float32) / 255.0
    n = stack.shape[0]
    out = np.empty_like(stack)
    for i in range(n):
        lo, hi = max(0, i - radius), min(n, i + radius + 1)
        out[i] = np.max(stack[lo:hi], axis=0)

    for i, p in enumerate(paths_in_order):
        stem = stems[i]
        a_old = stack[i]
        a_new = out[i]
        rises = a_new > a_old + (1.0 / 255.0)
        fg = np.asarray(Image.open(dir_fg / f"{stem}.png").convert("RGB")).copy()
        orig = np.asarray(Image.open(dir_in / f"{stem}.png").convert("RGB"))
        fg[rises] = orig[rises]
        Image.fromarray((np.clip(a_new, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L").save(
            dir_a / f"{stem}.png"
        )
        Image.fromarray(fg, mode="RGB").save(dir_fg / f"{stem}.png")


def _gym_refine_from_original(
    dir_in: Path,
    dir_fg: Path,
    dir_a: Path,
    paths_in_order: list[Path],
    *,
    protect_light_ui: bool,
    bleed_red_suppress: bool,
) -> None:
    """Use source RGB to fix on-video UI and crush spurious red gym bleed in uncertain regions."""
    import numpy as np
    from PIL import Image
    from tqdm.auto import tqdm

    for p in tqdm(paths_in_order, desc="gym-refine"):
        stem = p.stem
        orig = np.asarray(Image.open(dir_in / f"{stem}.png").convert("RGB"))
        fg = np.asarray(Image.open(dir_fg / f"{stem}.png").convert("RGB")).copy()
        ap = np.asarray(Image.open(dir_a / f"{stem}.png"))
        a = ap.astype(np.float32) / 255.0

        if protect_light_ui:
            mx = np.maximum(np.maximum(orig[..., 0], orig[..., 1]), orig[..., 2]).astype(np.float32)
            mn = np.minimum(np.minimum(orig[..., 0], orig[..., 1]), orig[..., 2]).astype(np.float32)
            sat = (mx - mn) / np.maximum(mx, 1.0)
            is_ui = (mx >= 198.0) & (sat < 0.14)
            try:
                from scipy.ndimage import binary_dilation

                is_ui = binary_dilation(is_ui, iterations=6)
            except ImportError:
                pass
            fg[is_ui] = orig[is_ui]
            a[is_ui] = 1.0

        if bleed_red_suppress:
            r = orig[..., 0].astype(np.float32)
            g = orig[..., 1].astype(np.float32)
            b = orig[..., 2].astype(np.float32)
            is_red = (
                (r > 112.0)
                & (r > g + 18.0)
                & (r > b + 18.0)
                & (g < 132.0)
                & (b < 132.0)
            )
            uncertain = (a > 0.04) & (a < 0.72)
            mask = is_red & uncertain
            a[mask] = np.clip(a[mask] * 0.12, 0.0, 1.0)

        Image.fromarray(fg, mode="RGB").save(dir_fg / f"{stem}.png")
        Image.fromarray((np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L").save(
            dir_a / f"{stem}.png"
        )


def _apply_alpha_ema(dir_a: Path, paths_in_order: list[Path], alpha_ema: float) -> None:
    if alpha_ema <= 0:
        return
    import numpy as np
    from PIL import Image

    prev: np.ndarray | None = None
    for p in paths_in_order:
        stem = p.stem
        ap = dir_a / f"{stem}.png"
        a = np.asarray(Image.open(ap), dtype=np.float32) / 255.0
        if prev is not None:
            a = alpha_ema * prev + (1.0 - alpha_ema) * a
        prev = a.copy()
        Image.fromarray((np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L").save(ap)


def _write_rgba_frames(
    dir_fg: Path,
    dir_a: Path,
    dir_out: Path,
    paths_in_order: list[Path],
) -> None:
    """Merge straight RGB + alpha into RGBA PNGs for transparent GIF / previews."""
    from PIL import Image
    from tqdm.auto import tqdm

    dir_out.mkdir(parents=True, exist_ok=True)
    for p in tqdm(paths_in_order, desc="rgba"):
        stem = p.stem
        rgb = Image.open(dir_fg / f"{stem}.png").convert("RGB")
        al = Image.open(dir_a / f"{stem}.png").convert("L")
        r, g, b = rgb.split()
        Image.merge("RGBA", (r, g, b, al)).save(dir_out / f"{stem}.png")


def _premul_backdrop_rgb(
    dir_fg: Path,
    dir_a: Path,
    dir_out: Path,
    paths_in_order: list[Path],
    backdrop_rgb: tuple[int, int, int] = (255, 255, 255),
    *,
    defringe: bool = True,
) -> None:
    """Composite on flat backdrop; optional decontamination of edge RGB vs backdrop (reduces halos)."""
    import numpy as np
    from PIL import Image

    bg = np.asarray(backdrop_rgb, dtype=np.float32)
    dir_out.mkdir(parents=True, exist_ok=True)
    for p in paths_in_order:
        stem = p.stem
        rgb = np.asarray(Image.open(dir_fg / f"{stem}.png").convert("RGB"), dtype=np.float32)
        a = np.asarray(Image.open(dir_a / f"{stem}.png"), dtype=np.float32) / 255.0
        a3 = np.clip(a[..., None], 0.0, 1.0)
        if defringe:
            lo, hi = 0.03, 0.92
            edge = (a[..., None] > lo) & (a[..., None] < hi)
            a_safe = np.maximum(a3, 1e-4)
            rgb_d = (rgb - (1.0 - a3) * bg) / a_safe
            rgb_d = np.clip(rgb_d, 0, 255)
            rgb = np.where(edge, rgb_d, rgb)
        comp = np.clip(rgb * a3 + bg * (1.0 - a3), 0, 255).astype(np.uint8)
        Image.fromarray(comp, mode="RGB").save(dir_out / f"{stem}.png")


def main() -> None:
    try:
        from rembg import new_session
        from tqdm.auto import tqdm
    except ImportError as e:
        vpip = Path(__file__).resolve().parent.parent / "venv" / "bin" / "python3"
        hint = ""
        if vpip.is_file():
            hint = (
                f"\nOr install into the project venv:\n"
                f'  "{vpip}" -m pip install -r requirements_rembg.txt\n'
            )
        print(
            "Missing dependency (rembg + pillow + tqdm + numpy + onnxruntime):\n"
            '  python3 -m pip install "rembg[cli]" onnxruntime pillow tqdm numpy\n'
            f"{hint}"
            "If rembg fails on Python 3.14, use Python 3.10–3.12 in a venv.\n"
            f"Import error: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Gym-style BG removal (rembg): person + held equipment → white backdrop + GIF."
    )
    parser.add_argument("--input", type=Path, default=None, help="Source video")
    parser.add_argument(
        "--model",
        type=str,
        default=_DEFAULT_MODEL_FLAG,
        help="rembg model id (default: isnet-general-use; or set by --fast / --hq). "
        "Gym-friendly: isnet-general-use, birefnet-general-lite, birefnet-general. "
        "Avoid for props: u2net_human_seg, birefnet-portrait, silueta.",
    )
    parser.add_argument(
        "--hq",
        action="store_true",
        help="Use birefnet-general (~1 GB download, best quality). Ignored if you pass --model.",
    )
    parser.add_argument(
        "--alpha-ema",
        type=float,
        default=0.0,
        help="Temporal alpha blend with previous frame (0=off, default — best for dumbbells/bar speed). "
        "Higher = smoother but can erase fast thin motion.",
    )
    parser.add_argument("--fg", type=Path, default=None, help="Output foreground mp4 (default fg_rembg.mp4)")
    parser.add_argument("--alpha", type=Path, default=None, help="Output alpha mp4 (default alpha_rembg.mp4)")
    parser.add_argument("--temp", type=Path, default=None, help="White backdrop mp4 (default temp_rembg.mp4)")
    parser.add_argument("--gif", type=Path, default=None, help="Output GIF (default output_rembg.gif)")
    parser.add_argument("--palette", type=Path, default=None, help="Palette png (default palette_rembg.png)")
    parser.add_argument(
        "--background",
        type=str,
        default="white",
        choices=("white", "lightgray"),
        help="Backdrop color",
    )
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument(
        "--alpha-matting",
        action="store_true",
        help="PyMatting refinement (much slower on CPU, ~3–8s/frame; small edge gains). Default is off.",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=0,
        help="Max long edge before rembg (0 = full frame, recommended for metal bars). "
        "Lowering saves little CPU (model still runs 1024²) but can thin small gear.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel rembg processes (default ~ half of CPU cores, max 8).",
    )
    parser.add_argument("--sequential", action="store_true", help="Single process (no pool).")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="When --model omitted: use u2netp (~5 MB, very fast; rougher on thin bars). --hq wins if both set.",
    )
    parser.add_argument(
        "--gif-width",
        type=int,
        default=560,
        help="GIF max width in px (taller = thinner bars survive palette; default 560).",
    )
    parser.add_argument(
        "--temporal-max-radius",
        type=int,
        default=0,
        help="Max alpha over ±N neighbor frames (default 0 — avoids motion ghosting). Use 1–2 only for thin bars that flicker.",
    )
    parser.add_argument(
        "--no-gym-refine",
        action="store_true",
        help="Skip light-UI + red-bleed passes (faster; less cleanup on overlays/gym red).",
    )
    parser.add_argument(
        "--gif-opaque",
        action="store_true",
        help="GIF on solid backdrop (old behavior). Default: transparent GIF from alpha.",
    )
    parser.add_argument(
        "--no-protect-light-ui",
        action="store_true",
        help="Do not force opaque light/white on-video overlays (title cards burn back in from source).",
    )
    parser.add_argument(
        "--no-bleed-red-suppress",
        action="store_true",
        help="Do not down-weight uncertain saturated-red pixels (gym machine bleed between limbs).",
    )
    parser.add_argument(
        "--no-defringe",
        action="store_true",
        help="Skip white-backdrop edge decontamination (slight halo may return).",
    )
    args = parser.parse_args()

    if args.model is _DEFAULT_MODEL_FLAG:
        if args.hq and args.fast:
            print("Note: --hq and --fast together → using birefnet-general (--hq wins).", flush=True)
        if args.hq:
            args.model = "birefnet-general"
        elif args.fast:
            args.model = "u2netp"
        else:
            args.model = "isnet-general-use"

    cpu_n = os.cpu_count() or 4
    if args.sequential:
        num_workers = 1
    elif args.workers is not None:
        num_workers = max(1, args.workers)
    else:
        num_workers = max(2, min(8, max(2, cpu_n // 2)))

    src = args.input
    if src is None:
        for candidate in (root / "input.mp4", root / "gym.mp4"):
            if candidate.is_file():
                src = candidate
                break
        if src is None:
            print("Error: pass --input or add input.mp4 / gym.mp4", file=sys.stderr)
            sys.exit(1)
    else:
        src = (root / src).resolve() if not src.is_absolute() else src
    if not src.is_file():
        print(f"Error: not found: {src}", file=sys.stderr)
        sys.exit(1)

    fg_mp4 = (root / args.fg) if args.fg else root / "fg_rembg.mp4"
    alpha_mp4 = (root / args.alpha) if args.alpha else root / "alpha_rembg.mp4"
    temp_mp4 = (root / args.temp) if args.temp else root / "temp_rembg.mp4"
    output_gif = (root / args.gif) if args.gif else root / "output_rembg.gif"
    palette_png = (root / args.palette) if args.palette else root / "palette_rembg.png"
    backdrop_rgb: tuple[int, int, int] = (255, 255, 255) if args.background == "white" else (240, 240, 240)

    fps = _video_fps(src)
    fps_s = f"{fps:.6f}".rstrip("0").rstrip(".")

    work = Path(tempfile.mkdtemp(prefix="rembg_frames_", dir=str(root)))
    frames_in = work / "in"
    frames_fg = work / "fg"
    frames_a = work / "a"
    frames_in.mkdir()
    frames_fg.mkdir()
    frames_a.mkdir()
    frames_comp = work / "comp"
    frames_rgba = work / "rgba"

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(src),
                str(frames_in / "%06d.png"),
            ],
            check=True,
        )
        paths = sorted(frames_in.glob("*.png"))
        if not paths:
            print("Error: no frames extracted", file=sys.stderr)
            sys.exit(1)

        ms = int(args.max_side) if args.max_side else 0
        tasks = [(str(p), str(frames_fg), str(frames_a), ms, args.alpha_matting) for p in paths]

        _ensure_model_cached(args.model)

        if num_workers <= 1:
            session = new_session(args.model)
            for p in tqdm(paths, desc="rembg"):
                _segment_frame(p, frames_fg, frames_a, ms, args.alpha_matting, session)
        else:
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=num_workers,
                mp_context=ctx,
                initializer=_pool_init,
                initargs=(args.model,),
            ) as ex:
                ch = max(1, min(12, len(tasks) // max(1, num_workers * 2)))
                list(
                    tqdm(
                        ex.map(_pool_worker_task, tasks, chunksize=ch),
                        total=len(tasks),
                        desc="rembg",
                    )
                )

        _temporal_max_alpha_with_fg_fix(
            frames_in, frames_fg, frames_a, paths, args.temporal_max_radius
        )
        _apply_alpha_ema(frames_a, paths, args.alpha_ema)
        if not args.no_gym_refine:
            _gym_refine_from_original(
                frames_in,
                frames_fg,
                frames_a,
                paths,
                protect_light_ui=not args.no_protect_light_ui,
                bleed_red_suppress=not args.no_bleed_red_suppress,
            )
        _premul_backdrop_rgb(
            frames_fg,
            frames_a,
            frames_comp,
            paths,
            backdrop_rgb,
            defringe=not args.no_defringe,
        )

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                fps_s,
                "-i",
                str(frames_comp / "%06d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "18",
                str(temp_mp4),
            ],
            check=True,
        )
        print("Solid backdrop video created (premultiplied backdrop)", flush=True)

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                fps_s,
                "-i",
                str(frames_fg / "%06d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "15",
                str(fg_mp4),
            ],
            check=True,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                fps_s,
                "-i",
                str(frames_a / "%06d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "15",
                str(alpha_mp4),
            ],
            check=True,
        )
        print("rembg foreground + alpha videos created", flush=True)

        if args.no_gif:
            return

        gw = max(320, min(1280, args.gif_width))
        scale = f"scale={gw}:-1:flags=lanczos"

        if args.gif_opaque:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(temp_mp4),
                    "-vf",
                    f"fps=15,{scale},palettegen=stats_mode=full",
                    "-frames:v",
                    "1",
                    "-update",
                    "1",
                    str(palette_png),
                ],
                check=True,
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(temp_mp4),
                    "-i",
                    str(palette_png),
                    "-filter_complex",
                    f"fps=15,{scale}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
                    "-loop",
                    "0",
                    str(output_gif),
                ],
                check=True,
            )
            print("GIF created (opaque backdrop)", flush=True)
        else:
            _write_rgba_frames(frames_fg, frames_a, frames_rgba, paths)
            fc = (
                f"[0:v]fps=15,{scale},split[s0][s1];"
                f"[s0]palettegen=stats_mode=full:max_colors=255:reserve_transparent=1[p];"
                f"[s1][p]paletteuse=alpha_threshold=128:diff_mode=rectangle:"
                f"dither=bayer:bayer_scale=2"
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-framerate",
                    fps_s,
                    "-i",
                    str(frames_rgba / "%06d.png"),
                    "-filter_complex",
                    fc,
                    "-loop",
                    "0",
                    str(output_gif),
                ],
                check=True,
            )
            print("GIF created (transparent)", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"Command failed (exit {exc.returncode}): {exc}", file=sys.stderr)
        sys.exit(exc.returncode or 1)
