"""BASS / BASS-II video reader: telemetry overlay and flame signal, frame by frame.

The BASS video burns a telemetry overlay into every frame:

    FUEL   AIR   FAN   RAD
      -2    20    89   0.26
    02/19/14 16:27:15

FUEL is the fuel-flow reading, AIR the anemometer reading (cm/s, slow
response), FAN the fan setting on the display, RAD the radiometer reading. The
overlay uses a fixed-pitch font at a fixed position, so every character sits in
a known cell.

Digit templates need no manual labels: the units-of-seconds digit of the clock
advances once per second in a known order, so it labels itself. Every other
digit cell is classified by normalised correlation against those templates.

Output is one row per video frame. A value is null when its cell could not be
read with confidence (frames where the overlay is mid-update); nothing is
interpolated or filled.

Geometry was measured on 720x480 files of PSI-25. `clock_selfcheck` and
`clock_step_ok` are reported per file so a different layout is detected
instead of silently misread.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Overlay geometry (pixels, 720x480 source).
CELL_X0 = 185.0      # left edge of character cell k = 0
CELL_PITCH = 28.63   # horizontal pitch between cells
CELL_W = 24
ROWS_LABELS = (32, 45)
ROWS_VALUES = (50, 63)
ROWS_CLOCK = (68, 81)
OVERLAY_BOTTOM = 90  # flame ROI starts below this row
FRAME_SIZE = (720, 480)
VISIBLE_RIGHT = 716    # columns from here on are black padding

LABEL_CELLS = (0, 1, 2, 3, 5, 6, 7, 10, 11, 12, 15, 16, 17)   # FUEL AIR FAN RAD
# Character cells (index k) on the values row. The value is right-aligned in each field.
FIELD_CELLS = {"fuel": (0, 1, 2), "air": (5, 6, 7), "fan": (10, 11, 12)}
RAD_CELLS = (15, 17, 18)       # units, tenths, hundredths ('.' at 16); cell 18 is cut by the frame edge
# Clock row: MM/DD/YY HH:MM:SS starting at cell -1.
DATE_CELLS = (-1, 0, 2, 3, 5, 6)
TIME_CELLS = (8, 9, 11, 12, 14, 15)
SEC_UNITS_CELL = 15

BLANK_MAX = 90        # a cell whose brightest pixel is below this is blank
MIN_SCORE = 0.80      # minimum template correlation to accept a digit
MAX_SHIFT = 2         # horizontal tolerance (px) when matching a cell to a template
FLAME_LEVEL = 60      # max(R,G,B) above this counts as a lit pixel
DIM_LEVEL = 24        # lower level for dim blue flames (near the quenching limit)
CORE_LEVEL = 200      # saturated flame core
MIN_REGION_PX = 40    # smaller lit regions are noise
BODY_SHARE = 0.25     # flame body: pixels above this share of the frame's peak level
CORE_SHARE = 0.85     # hot core: pixels above this share of the peak
LIGHTS_ON_BG = 10.0   # median level above this means the chamber lights are on
_KERNEL = np.ones((3, 3), np.uint8)
EDGE_MAX_LOSS = 0.01  # the cut RAD hundredths digit is dropped if cropping costs more than this in the self-check


@dataclass(frozen=True)
class VideoRead:
    fps: float
    n_frames: int
    duration_s: float
    clock_selfcheck: float   # share of seconds-units digits read correctly against the clock's own sequence (overlay frames only)
    overlay_present: float   # share of frames that carry the overlay at all
    clock_step_ok: float     # share of consecutive readable frames where the clock advances by 0 or 1 s
    edge_digit_check: float  # same self-check with the digit cropped like the cut RAD hundredths cell
    edge_visible_px: int     # visible width of the cut cell, of 24
    overlay_dx: float
    overlay_dy: int
    readable: dict           # share of frames with a value, per field
    frames: dict             # column name -> numpy array, one entry per frame


def _cell_x(k: int, dx: float = 0.0) -> int:
    return int(round(CELL_X0 + dx + CELL_PITCH * k))


def overlay_offset(cap, n_samples: int = 80) -> tuple[float, int]:
    """Shift (dx, dy) of the overlay against the reference geometry.

    The overlay is not at the same pixel position in every recording session.
    The label row (FUEL AIR FAN RAD) never changes, so the per-pixel median over
    sampled frames isolates it even when the chamber lights are on in some frames.
    """
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, max(n - 1, 0), n_samples).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame[:ROWS_VALUES[0] + 10], cv2.COLOR_BGR2GRAY))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    lit = np.median(np.stack(frames), axis=0) > BLANK_MAX
    rows = np.nonzero(lit[:, 150:VISIBLE_RIGHT].sum(axis=1) > 20)[0]
    if rows.size == 0:
        raise ValueError("overlay not found: no label row")
    top = int(rows[0])
    dy = top - ROWS_LABELS[0]
    colsum = lit[top:top + (ROWS_LABELS[1] - ROWS_LABELS[0])].sum(axis=0).astype(float)
    best = (-1e9, 0.0)
    for dx in np.arange(-8, 12.01, 0.5):
        inside = np.zeros(colsum.size, bool)
        for k in LABEL_CELLS:
            x = _cell_x(k, dx)
            inside[max(x, 0):x + CELL_W] = True
        score = colsum[inside].sum() - colsum[~inside].sum()
        if score > best[0]:
            best = (score, float(dx))
    return best[1], dy


def _norm(p: np.ndarray) -> np.ndarray:
    p = p.astype(np.float32)
    p = p - p.mean(axis=(-1, -2), keepdims=True)
    return p / (np.sqrt((p ** 2).sum(axis=(-1, -2), keepdims=True)) + 1e-6)


def _clock_templates(t: np.ndarray, sec_units: np.ndarray):
    """Self-label the seconds-units cell. Labels are relative (clock order); `_anchor` fixes the rotation."""
    step = max(1, len(t) // 3000)            # phase search on a subsample
    ts, sn = t[::step], _norm(sec_units[::step])
    best = (-1.0, 0.0)
    for ph in np.arange(0, 1, 0.02):
        frac = (ts - ph) % 1
        keep = (frac > 0.2) & (frac < 0.8)
        lab = np.floor(ts - ph).astype(int) % 10
        q = 0.0
        for d in range(10):
            m = keep & (lab == d)
            if m.any():
                q += float((sn[m].mean(0) ** 2).sum())
        if q > best[0]:
            best = (q, ph)
    ph = best[1]
    frac = (t - ph) % 1
    keep = (frac > 0.2) & (frac < 0.8)
    lab = np.floor(t - ph).astype(int) % 10
    tm = np.stack([sec_units[keep & (lab == d)].astype(np.float32).mean(0) if (keep & (lab == d)).any()
                   else np.zeros(sec_units.shape[1:], np.float32) for d in range(10)])
    return tm, lab, keep


def _anchor(templates: np.ndarray) -> int:
    """Rotation r such that relative label d shows digit (d + r) % 10.

    Stroke density is layout independent: in this dot-matrix font '1' has the
    fewest lit pixels of all digits.
    """
    lit = np.array([(tpl > 0.5 * tpl.max()).sum() if tpl.max() > 0 else 1e9 for tpl in templates])
    return (1 - int(np.argmin(lit))) % 10


def _shifted(templates: np.ndarray, dx: int) -> np.ndarray:
    out = np.zeros_like(templates)
    if dx > 0:
        out[:, :, dx:] = templates[:, :, :-dx]
    elif dx < 0:
        out[:, :, :dx] = templates[:, :, -dx:]
    else:
        out[:] = templates
    return out


def _classify(patches: np.ndarray, templates: np.ndarray, rot: int, width: int | None = None):
    """Digit per patch (nan when not confident) and blank mask.

    Tolerates a horizontal misalignment of up to MAX_SHIFT px between a cell and
    the clock cell the templates come from. `width` crops cells cut by the frame edge.
    """
    if width is not None:
        patches = patches[:, :, :width]
    energy = patches.max(axis=(1, 2))
    pn = _norm(patches)
    score = np.full((len(patches), 10), -1.0, np.float32)
    for dx in range(-MAX_SHIFT, MAX_SHIFT + 1):
        tpl = _shifted(templates, dx)
        if width is not None:
            tpl = tpl[:, :, :width]
        score = np.maximum(score, np.einsum("nij,dij->nd", pn, _norm(tpl)))
    digit = ((score.argmax(1) + rot) % 10).astype(float)
    blank = energy < BLANK_MAX
    digit[(score.max(1) < MIN_SCORE) | blank] = np.nan
    return digit, blank


def _is_minus(patches: np.ndarray) -> np.ndarray:
    """A minus sign lights only the middle rows of the cell."""
    p = patches.astype(np.float32)
    peak = p.max(axis=(1, 2))
    lit = p > 0.5 * peak[:, None, None]
    rows = lit.sum(axis=2)
    inside = rows[:, 4:9].sum(axis=1)
    return (peak >= BLANK_MAX) & (inside >= 0.95 * np.maximum(rows.sum(axis=1), 1)) & (inside >= 12)


def _right_aligned(cells, templates, rot):
    """Integer from up to three right-aligned cells, with optional leading minus."""
    hundreds, tens, units = cells
    du, bu = _classify(units, templates, rot)
    dt, bt = _classify(tens, templates, rot)
    dh, bh = _classify(hundreds, templates, rot)
    mt, mh = _is_minus(tens), _is_minus(hundreds)
    value = du.copy()
    sign = np.ones(len(du))
    two = ~bt & ~mt                       # a digit in the tens cell
    value[two] = dt[two] * 10 + du[two]
    sign[mt] = -1
    three = two & ~bh & ~mh
    value[three] = dh[three] * 100 + value[three]
    sign[two & mh] = -1
    return value * sign


def flame_region(frame: np.ndarray):
    """Flame body and hot core below the overlay.

    The body is the connected region around the brightest pixels, cut at a level
    relative to the frame's peak (BODY_SHARE of the 99.9th percentile, never below
    DIM_LEVEL). A fixed low level would also take in the glow on the sample holder
    and reflections. The core is the part of the body within CORE_SHARE of the peak;
    its centroid is the point of highest luminous intensity.

    Returns (body_mask, stats) with stats = (body_area, core_cx, core_cy, x0, y0, w, h,
    body_cx, body_cy, core_area, peak, head_x, head_y, head_radius), or (None, None) when nothing reaches MIN_REGION_PX.
    No model: the chamber is dark, so levels and connected components are enough.
    """
    b, g, r = cv2.split(frame[OVERLAY_BOTTOM:])
    roi = cv2.GaussianBlur(cv2.max(cv2.max(b, g), r), (5, 5), 0)
    peak = float(np.percentile(roi, 99.9))
    if peak <= DIM_LEVEL:
        return None, None
    level = max(DIM_LEVEL, BODY_SHARE * peak)
    m = cv2.morphologyEx((roi > level).astype(np.uint8), cv2.MORPH_OPEN, _KERNEL)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None, None
    # the component that holds the most light, not the one with the most pixels
    light = np.bincount(labels.ravel(), weights=roi.ravel().astype(np.float64), minlength=n)
    light[0] = 0
    k = int(np.argmax(light))
    area = int(stats[k, cv2.CC_STAT_AREA])
    if area < MIN_REGION_PX:
        return None, None
    comp = labels == k
    core = comp & (roi >= CORE_SHARE * peak)
    ys, xs = np.nonzero(core)
    if xs.size == 0:
        ys, xs = np.nonzero(comp)
    full = np.zeros(frame.shape[:2], np.uint8)
    full[OVERLAY_BOTTOM:] = comp.astype(np.uint8)
    x0, y0, w, h = (int(stats[k, i]) for i in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
    # flame head: the largest circle that fits inside the body. Its centre is where the
    # flame is anchored (the round head around the burning sample), not the tail.
    dist = cv2.distanceTransform(np.pad(comp.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    _, head_r, _, (hx, hy) = cv2.minMaxLoc(dist)
    return full, (area, float(xs.mean()), float(ys.mean()) + OVERLAY_BOTTOM, x0, y0 + OVERLAY_BOTTOM, w, h,
                  float(cents[k][0]), float(cents[k][1]) + OVERLAY_BOTTOM, int(core.sum()), peak,
                  float(hx), float(hy) + OVERLAY_BOTTOM, float(head_r))


def read_video(path: str) -> VideoRead:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_decl = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or n_decl <= 0:
        raise ValueError("video cannot be opened (damaged or truncated file)")
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if size != FRAME_SIZE:
        raise ValueError(f"unexpected frame size {size}; overlay geometry is calibrated for {FRAME_SIZE}")
    dx, dy = overlay_offset(cap)
    rv = (ROWS_VALUES[0] + dy, ROWS_VALUES[1] + dy)
    rc = (ROWS_CLOCK[0] + dy, ROWS_CLOCK[1] + dy)
    value_ks = sorted({k for ks in FIELD_CELLS.values() for k in ks} | set(RAD_CELLS))
    clock_ks = sorted(set(DATE_CELLS) | set(TIME_CELLS))
    hv, hc = ROWS_VALUES[1] - ROWS_VALUES[0], ROWS_CLOCK[1] - ROWS_CLOCK[0]
    vals = np.zeros((n_decl, len(value_ks), hv, CELL_W), np.uint8)
    clk = np.zeros((n_decl, len(clock_ks), hc, CELL_W), np.uint8)
    lit = np.zeros(n_decl, np.int32)
    core = np.zeros(n_decl, np.int32)
    dim = np.zeros(n_decl, np.int32)
    blue = np.zeros(n_decl, np.float32)
    bg = np.zeros(n_decl, np.float32)
    geo = np.full((n_decl, 14), np.nan, np.float32)  # see flame_region
    i = 0
    while i < n_decl:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(frame[:rc[1]], cv2.COLOR_BGR2GRAY)
        for j, k in enumerate(value_ks):
            x = _cell_x(k, dx)
            patch = g[rv[0]:rv[1], x:x + CELL_W]
            vals[i, j, :, :patch.shape[1]] = patch
        for j, k in enumerate(clock_ks):
            x = _cell_x(k, dx)
            clk[i, j] = g[rc[0]:rc[1], x:x + CELL_W]
        b, gr, r = cv2.split(frame[OVERLAY_BOTTOM:])
        roi = cv2.max(cv2.max(b, gr), r)
        mask = roi > FLAME_LEVEL
        n = int(mask.sum())
        lit[i] = n
        core[i] = int((roi > CORE_LEVEL).sum())
        dmask = roi > DIM_LEVEL
        dim[i] = int(dmask.sum())
        blue[i] = float((b[dmask] > r[dmask]).mean()) if dim[i] else np.nan
        bg[i] = float(np.median(roi[::4, ::4]))
        if bg[i] <= LIGHTS_ON_BG:                    # with the lights on the whole chamber is lit
            _, st = flame_region(frame)
            if st is not None:
                geo[i] = st
        i += 1
    cap.release()
    n = i
    vals, clk, lit, core, bg, geo = vals[:n], clk[:n], lit[:n], core[:n], bg[:n], geo[:n]
    dim, blue = dim[:n], blue[:n]
    t = np.arange(n) / fps
    vi = {k: j for j, k in enumerate(value_ks)}
    ci = {k: j for j, k in enumerate(clock_ks)}

    sec_units = clk[:, ci[SEC_UNITS_CELL]]
    templates, rel, keep = _clock_templates(t, sec_units)
    rot = _anchor(templates)
    got, _ = _classify(sec_units, templates, rot)
    truth = (rel + rot) % 10
    present = sec_units.max(axis=(1, 2)) >= BLANK_MAX      # frames where the overlay is on screen
    overlay_present = float(present.mean())
    keep = keep & present
    selfcheck = float((got[keep] == truth[keep]).mean()) if keep.any() else 0.0
    edge_w = VISIBLE_RIGHT - _cell_x(RAD_CELLS[2], dx)
    got_edge, _ = _classify(sec_units, templates, rot, width=edge_w)
    edge_check = float((got_edge[keep] == truth[keep]).mean())

    cols = {"frame": np.arange(n), "t_s": t}
    for name, ks in FIELD_CELLS.items():
        cols[name] = _right_aligned([vals[:, vi[k]] for k in ks], templates, rot)
    cols["air_cm_s"] = cols.pop("air")
    cols["fan_display"] = cols.pop("fan")
    ru, _ = _classify(vals[:, vi[RAD_CELLS[0]]], templates, rot)
    rt, _ = _classify(vals[:, vi[RAD_CELLS[1]]], templates, rot)
    rh, _ = _classify(vals[:, vi[RAD_CELLS[2]]], templates, rot, width=edge_w)
    if edge_check < selfcheck - EDGE_MAX_LOSS:                    # too little of the last digit is visible in this file
        rh = np.full(n, np.nan)
    cols["rad"] = ru + rt / 10 + rh / 100              # null when any of the three digits is unread
    cols["rad_1dp"] = ru + rt / 10                     # does not depend on the cut digit

    def pair(a, b):
        da, _ = _classify(clk[:, ci[a]], templates, rot)
        db, _ = _classify(clk[:, ci[b]], templates, rot)
        return da * 10 + db
    mo, dd, yy = pair(-1, 0), pair(2, 3), pair(5, 6)
    hh, mi, ss = pair(8, 9), pair(11, 12), pair(14, 15)
    tod = hh * 3600 + mi * 60 + ss
    cols.update({"clock_month": mo, "clock_day": dd, "clock_year": yy, "clock_tod_s": tod})
    ok = np.isfinite(tod)
    d = np.diff(tod[ok])
    step_ok = float(((d == 0) | (d == 1)).mean()) if d.size else float("nan")

    cols.update({"dim_px": dim, "lit_px": lit, "core_px": core, "blue_share": blue,
                 "background": bg, "lights_on": bg > LIGHTS_ON_BG,
                 "flame_area_px": geo[:, 0],                          # flame body
                 "flame_cx": geo[:, 1], "flame_cy": geo[:, 2],        # centroid of the hot core (brightest part)
                 "flame_radius_px": np.sqrt(geo[:, 0] / np.pi),      # radius of the circle with the body's area
                 "flame_x0": geo[:, 3], "flame_y0": geo[:, 4], "flame_w": geo[:, 5], "flame_h": geo[:, 6],
                 "body_cx": geo[:, 7], "body_cy": geo[:, 8],          # geometric centroid of the body
                 "core_area_px": geo[:, 9], "peak_level": geo[:, 10],
                 "head_x": geo[:, 11], "head_y": geo[:, 12],          # centre of the largest inscribed circle
                 "head_radius_px": geo[:, 13],
                 "overlay_dx": np.full(n, dx, np.float32), "overlay_dy": np.full(n, dy, np.int16)})
    readable = {k: float(np.isfinite(cols[k]).mean())
                for k in ("fuel", "air_cm_s", "fan_display", "rad", "rad_1dp", "clock_tod_s")}
    return VideoRead(fps=float(fps), n_frames=n, duration_s=n / fps, clock_selfcheck=selfcheck, overlay_present=overlay_present,
                     clock_step_ok=step_ok, edge_digit_check=edge_check, edge_visible_px=int(edge_w),
                     overlay_dx=dx, overlay_dy=dy, readable=readable, frames=cols)
