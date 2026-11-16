"""Cell parsers. Each returns a list of Parsed values; nothing is inferred or corrected."""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field

NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
MISSING = {"", "none", "n/a", "na", "-", "?", "tbd", "nan", "null", "--"}
_UNIT = r"([a-zA-Z%°\"µ/^.]+(?: [a-zA-Z]+)?)?"  # no digits, at most two words
_RE_RANGE = re.compile(rf"^({NUM})\s*(?:-|–|to)\s*({NUM})\s*{_UNIT}$", re.I)
_RE_NUM = re.compile(rf"^({NUM})\s*{_UNIT}$")
_RE_FRACTION = re.compile(r'^(\d+)\s*/\s*(\d+)\s*("|in|inch)?')


@dataclass
class Parsed:
    value: float | bool | str | None = None
    lo: float | None = None
    hi: float | None = None
    qualifier: str = "eq"
    unit: str | None = None  # unit found inside the cell, overrides the column unit
    text: str = ""
    flags: list[str] = field(default_factory=list)


def clean(s) -> str:
    return re.sub(r"\s+", " ", str(s if s is not None else "")).strip()


def number(text: str, tokens: dict | None = None) -> Parsed:
    t = clean(text)
    p = Parsed(text=t)
    if tokens and t in tokens:
        spec = tokens[t]
        p.value, p.qualifier = spec.get("value"), spec.get("qualifier", "eq")
        p.flags.append(spec.get("flag", "token"))
        return p
    if t.lower() in MISSING:
        if t:
            p.flags.append("missing_token")
        return p
    body = t
    m = re.match(r"^(~|≈|approx\.?|about)\s*", body, re.I)
    if m:
        p.qualifier, body = "approx", body[m.end():]
        p.flags.append("approx")
    elif body[0] in "<>":
        p.qualifier = "lt" if body[0] == "<" else "gt"
        p.flags.append(f"censored_{p.qualifier}")
        body = body[1:].strip()
    body = re.sub(r"\s*\(.*\)\s*$", "", body)  # trailing "(note 3)"
    if body.lower() in MISSING:
        p.qualifier, p.flags = "eq", ["missing_token"]
        return p
    m = _RE_RANGE.match(body)
    if m and float(m.group(2)) >= 0 and not body.startswith("-"):
        p.lo, p.hi = sorted((float(m.group(1)), float(m.group(2))))
        p.value, p.qualifier, p.unit = (p.lo + p.hi) / 2, "range", (m.group(3) or None)
        p.flags.append("range_midpoint")
        return p
    m = _RE_NUM.match(body)
    if m:
        p.value, p.unit = float(m.group(1)), (m.group(2) or None)
        if re.match(r"^0\d", m.group(1)):
            p.flags.append("suspect_leading_zero")
        return p
    p.qualifier = "eq"
    p.flags = [f for f in p.flags if not f.startswith(("censored", "approx"))] + ["non_numeric"]
    return p


def sequence(text: str) -> list[Parsed]:
    """'89,62,50,41' / '66 > 52 > 45' / '5, 2, 1, <1' -> ordered elements. Parentheticals are dropped."""
    t = clean(text)
    if t.lower() in MISSING:
        return []
    body = re.sub(r"\([^)]*\)?", " ", t)
    parts = [x.strip() for x in re.split(r"\s*(?:,|;|>(?!\s*$)|\bthen\b)\s*|(?<=\d)\s{1,}(?=[\d.<])", body) if x and x.strip()]
    out = []
    for part in parts:
        p = Parsed(text=part)
        m = re.search(rf"(<)?\s*({NUM})(?:\s*-\s*({NUM}))?", part)
        if m:
            p.value = float(m.group(2))
            if m.group(3):
                p.lo, p.hi = sorted((p.value, float(m.group(3))))
                p.value, p.qualifier = (p.lo + p.hi) / 2, "range"
                p.flags.append("range_midpoint")
            if m.group(1):
                p.qualifier = "lt"
                p.flags.append("censored_lt")
            if re.search(r"\bpots?\b", part, re.I) or re.search(r"\bpots?\b", t, re.I) and p.value < 10:
                p.flags.append("potentiometer_setting")
        else:
            p.flags.append("non_numeric")
        out.append(p)
    return out


def multi(text: str, tokens: dict | None = None) -> list[Parsed]:
    """'21; 26.5; 40' or '0.5; 1.0; or 3.0' -> list of setpoints."""
    t = clean(text)
    parts = [re.sub(r"^or\s+", "", x.strip(), flags=re.I) for x in re.split(r";|,|\band\b|&", t) if x.strip()]
    return [number(x, tokens) for x in parts]


def enum(text: str, mapping: dict) -> Parsed:
    t = clean(text)
    p = Parsed(text=t)
    key = t.lower()
    if key in MISSING and key not in mapping:
        return p
    for k, v in mapping.items():
        if str(k).lower() == key:
            p.value = v
            return p
    for k, v in mapping.items():  # prefix match: 'yes, PCI-D'
        if key.startswith(str(k).lower()):
            p.value = v
            return p
    p.flags.append("enum_unmapped")
    return p


def text(value: str) -> Parsed:
    t = clean(value)
    return Parsed(value=t or None, text=t)


_DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%d-%b-%y", "%d-%b-%Y", "%Y-%b-%d")


def date(value: str) -> Parsed:
    t = clean(value).replace("`", "").replace("'", "")
    p = Parsed(text=clean(value))
    if not t:
        return p
    head = t.split(" ")[0].split("T")[0]
    for fmt in _DATE_FORMATS:
        try:
            d = dt.datetime.strptime(head.title() if "%b" in fmt else head, fmt).date()
        except ValueError:
            continue
        if not 1995 <= d.year <= 2026:
            p.flags.append("suspect_date")
            return p
        p.value = d.isoformat()
        return p
    p.flags.append("suspect_date")
    return p


def inches(text: str) -> float | None:
    m = _RE_FRACTION.match(clean(text))
    return int(m.group(1)) / int(m.group(2)) if m else None
