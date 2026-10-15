"""Reusable presentation helpers: global styles, headers, tiles and course cards.

All dataset text is HTML-escaped before it is placed in custom markup.
"""

from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from src.knowledge_based import CAVEAT_PREFIX

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root {
  --sb-surface: #ffffff;
  --sb-page: #f7f7f4;
  --sb-ink: #14161a;
  --sb-ink-2: #555a63;
  --sb-muted: #8a8f98;
  --sb-line: #e6e5df;
  --sb-accent: #1c5cab;
  --sb-accent-2: #2a78d6;
  --sb-accent-soft: #ebf2fc;
  --sb-warm: #c2410c;
  --sb-warm-soft: #fdf1ea;
  --sb-good: #157a3c;
  --sb-good-soft: #e9f6ee;
}
html, body, .stApp, .stApp p, .stApp label, .stApp li, .stApp input, .stApp textarea, .stApp button,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp summary, .stApp a, [class^="sb-"], [class*=" sb-"] { font-family: 'Inter', system-ui, -apple-system, sans-serif; }
/* never override Streamlit's icon font */
[data-testid="stIconMaterial"], .material-symbols-rounded { font-family: 'Material Symbols Rounded' !important; }
.block-container { max-width: 1120px; padding-top: 3.4rem; padding-bottom: 4rem; }
h1, h2, h3, h4 { letter-spacing: -0.015em; }
[data-testid="stSidebar"] { border-right: 1px solid var(--sb-line); }

/* headers */
.sb-eyebrow { font-size: 0.74rem; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase;
  color: var(--sb-accent-2); margin-bottom: 0.2rem; }
.sb-title { font-size: 2.05rem; font-weight: 800; line-height: 1.15; margin: 0 0 0.35rem; color: var(--sb-ink); }
.sb-subtitle { color: var(--sb-ink-2); font-size: 1rem; max-width: 720px; margin-bottom: 1.4rem; }
.sb-section { font-size: 1.15rem; font-weight: 700; color: var(--sb-ink); margin: 1.8rem 0 0.8rem; }

/* hero */
.sb-hero { position: relative; overflow: hidden; background: #0f2f5c; color: #fff; border-radius: 20px;
  padding: 2.8rem 2.6rem; margin-bottom: 1.6rem; }
.sb-hero::after { content: ""; position: absolute; right: -80px; top: -80px; width: 340px; height: 340px;
  border-radius: 50%; background: radial-gradient(circle, rgba(42,120,214,.55), rgba(42,120,214,0) 70%); }
.sb-hero h1 { color: #fff; font-size: 2.7rem; font-weight: 800; margin: 0 0 0.6rem; position: relative; z-index: 1; }
.sb-hero p { color: #cfdcf0; font-size: 1.1rem; max-width: 600px; margin: 0; position: relative; z-index: 1; line-height: 1.55; }

/* stat tiles */
.sb-tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 0 0 1.4rem; }
.sb-tile { background: var(--sb-surface); border: 1px solid var(--sb-line); border-radius: 14px; padding: 1rem 1.1rem; }
.sb-tile-value { font-size: 1.6rem; font-weight: 800; color: var(--sb-ink); line-height: 1.15; }
.sb-tile-label { font-size: 0.82rem; color: var(--sb-ink-2); margin-top: 0.2rem; }

/* feature cards (home) */
.sb-features { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 14px; margin-bottom: 0.4rem; }
.sb-feature { background: var(--sb-surface); border: 1px solid var(--sb-line); border-radius: 16px; padding: 1.3rem 1.3rem 1.1rem;
  margin-bottom: 0.3rem; }
.sb-feature-icon { width: 40px; height: 40px; border-radius: 12px; background: var(--sb-accent-soft); color: var(--sb-accent);
  display: flex; align-items: center; justify-content: center; font-size: 1.2rem; margin-bottom: 0.8rem; }
.sb-feature b { display: block; font-size: 1.02rem; margin-bottom: 0.3rem; color: var(--sb-ink); }
.sb-feature span { color: var(--sb-ink-2); font-size: 0.9rem; line-height: 1.5; }

/* course cards */
.sb-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(330px, 1fr)); gap: 14px; }
.sb-card { background: var(--sb-surface); border: 1px solid var(--sb-line); border-radius: 16px; padding: 1.1rem 1.2rem 1rem;
  display: flex; flex-direction: column; gap: 0.55rem; transition: box-shadow .15s ease, border-color .15s ease; }
.sb-card:hover { box-shadow: 0 6px 22px rgba(20,22,26,.07); border-color: #d6d4cc; }
.sb-card-head { display: flex; justify-content: space-between; align-items: center; gap: 10px; }
.sb-provider { font-size: 0.8rem; font-weight: 600; color: var(--sb-ink-2); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.sb-match { display: flex; align-items: center; gap: 7px; flex: none; font-size: 0.74rem; color: var(--sb-muted); }
.sb-match-bar { width: 64px; height: 6px; border-radius: 6px; background: #eeede8; overflow: hidden; display: inline-block; }
.sb-match-bar i { display: block; height: 100%; border-radius: 6px; background: var(--sb-accent-2); }
.sb-course-title { font-weight: 700; font-size: 1.04rem; line-height: 1.35; color: var(--sb-ink) !important; text-decoration: none !important; }
.sb-course-title:hover { color: var(--sb-accent) !important; }
.sb-pills { display: flex; flex-wrap: wrap; gap: 6px; }
.sb-pill { font-size: 0.74rem; font-weight: 600; padding: 3px 9px; border-radius: 999px; background: #f1f0eb; color: var(--sb-ink-2); }
.sb-pill-level { background: var(--sb-accent-soft); color: var(--sb-accent); }
.sb-chips { display: flex; flex-wrap: wrap; gap: 6px; }
.sb-chip { font-size: 0.76rem; padding: 3px 10px; border-radius: 999px; background: #f6f5f1; color: var(--sb-ink-2);
  border: 1px solid var(--sb-line); }
.sb-chip-accent { background: var(--sb-good-soft); color: var(--sb-good); border-color: #cbe8d5; }
.sb-chip-warm { background: var(--sb-warm-soft); color: var(--sb-warm); border-color: #f6d6c4; }
.sb-chip-blue { background: var(--sb-accent-soft); color: var(--sb-accent); border-color: #cfe0f6; }
.sb-why { font-size: 0.86rem; color: var(--sb-ink-2); line-height: 1.5; margin: 0; padding: 0; list-style: none; }
.sb-why li { padding-left: 1.05rem; position: relative; margin: 0 0 0.15rem !important; font-size: 0.85rem !important;
  line-height: 1.45 !important; color: var(--sb-ink-2); }
.sb-why li::before { content: "✓"; position: absolute; left: 0; color: var(--sb-good); font-weight: 700; }
.sb-why li.sb-note::before { content: "i"; color: var(--sb-warm); font-style: italic; font-family: Georgia, serif; }
.sb-link { margin-top: auto; font-size: 0.84rem; font-weight: 600; color: var(--sb-accent) !important; text-decoration: none !important; }
.sb-link:hover { text-decoration: underline; }

/* compact course row (skill gap, path) */
.sb-row { display: flex; justify-content: space-between; align-items: center; gap: 12px; padding: 0.55rem 0;
  border-top: 1px solid var(--sb-line); }
.sb-row:first-of-type { border-top: none; }
.sb-row-title { font-weight: 600; font-size: 0.93rem; color: var(--sb-ink) !important; text-decoration: none !important; }
.sb-row-title:hover { color: var(--sb-accent) !important; }
.sb-meta { color: var(--sb-ink-2); font-size: 0.82rem; margin-top: 0.1rem; }

/* skill-gap cards */
.sb-gap { background: var(--sb-surface); border: 1px solid var(--sb-line); border-radius: 16px; padding: 1rem 1.2rem 0.6rem; margin-bottom: 12px; }
.sb-gap-head { display: flex; justify-content: space-between; align-items: center; gap: 10px; margin-bottom: 0.3rem; flex-wrap: wrap; }
.sb-gap-name { font-weight: 700; font-size: 1.02rem; color: var(--sb-ink); }
.sb-gap-note { font-size: 0.8rem; color: var(--sb-muted); margin-bottom: 0.2rem; }
.sb-status-ready { background: var(--sb-good-soft); color: var(--sb-good); }
.sb-status-wait { background: var(--sb-warm-soft); color: var(--sb-warm); }

/* readiness */
.sb-ready { background: var(--sb-surface); border: 1px solid var(--sb-line); border-radius: 16px; padding: 1.3rem 1.4rem; }
.sb-ready-value { font-size: 2.6rem; font-weight: 800; color: var(--sb-ink); line-height: 1; }
.sb-ready-label { color: var(--sb-ink-2); font-size: 0.9rem; margin: 0.35rem 0 0.8rem; }
.sb-ready-bar { height: 10px; border-radius: 10px; background: #eeede8; overflow: hidden; }
.sb-ready-bar i { display: block; height: 100%; background: var(--sb-accent-2); border-radius: 10px; }
.sb-label { font-size: 0.74rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: var(--sb-muted);
  margin: 0.2rem 0 0.45rem; }

/* learning path timeline */
.sb-stage { display: flex; align-items: center; gap: 10px; margin: 1.6rem 0 0.7rem; }
.sb-stage b { font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--sb-accent-2); }
.sb-stage span { flex: 1; height: 1px; background: var(--sb-line); }
.sb-step { display: flex; gap: 14px; background: var(--sb-surface); border: 1px solid var(--sb-line); border-radius: 16px;
  padding: 1rem 1.2rem; margin-bottom: 10px; }
.sb-step-num { flex: none; width: 34px; height: 34px; border-radius: 50%; background: var(--sb-accent); color: #fff;
  font-weight: 700; display: flex; align-items: center; justify-content: center; font-size: 0.92rem; }
.sb-step-body { flex: 1; min-width: 0; }
.sb-step-skill { font-weight: 700; font-size: 1.02rem; color: var(--sb-ink); display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.sb-step-why { font-size: 0.84rem; color: var(--sb-muted); margin-top: 0.45rem; }


@media (max-width: 640px) {
  .sb-hero { padding: 1.6rem 1.4rem; } .sb-hero h1 { font-size: 2rem; }
  .sb-title { font-size: 1.6rem; }
  .sb-grid { grid-template-columns: 1fr; }
}

/* featured (top) recommendation */
.sb-featured { display: grid; grid-template-columns: 1.6fr 1fr; gap: 1.6rem; background: var(--sb-surface);
  border: 1px solid var(--sb-line); border-radius: 18px; padding: 1.5rem 1.6rem; margin-bottom: 14px;
  box-shadow: 0 10px 30px rgba(15,47,92,.07); position: relative; overflow: hidden; }
.sb-featured::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 5px; background: var(--sb-accent-2); }
.sb-badge { display: inline-flex; align-items: center; gap: 6px; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em;
  text-transform: uppercase; color: var(--sb-accent); background: var(--sb-accent-soft); border-radius: 999px; padding: 4px 10px; }
.sb-featured-title { font-size: 1.4rem; font-weight: 800; line-height: 1.25; color: var(--sb-ink) !important;
  text-decoration: none !important; display: block; margin: 0.6rem 0 0.35rem; }
.sb-featured-title:hover { color: var(--sb-accent) !important; }
.sb-desc { color: var(--sb-ink-2); font-size: 0.9rem; line-height: 1.55; display: -webkit-box; -webkit-line-clamp: 3;
  -webkit-box-orient: vertical; overflow: hidden; margin: 0.4rem 0 0.2rem; }
.sb-desc-short { -webkit-line-clamp: 2; font-size: 0.84rem; margin: 0; }
.sb-featured-side { border-left: 1px solid var(--sb-line); padding-left: 1.4rem; display: flex; flex-direction: column; gap: 0.6rem; }
.sb-cta { display: inline-block; margin-top: auto; background: var(--sb-accent); color: #fff !important; font-weight: 600;
  font-size: 0.88rem; padding: 0.55rem 1rem; border-radius: 10px; text-decoration: none !important; text-align: center; }
.sb-cta:hover { background: #154a8c; }

/* profile summary strip */
.sb-strip { display: flex; flex-wrap: wrap; align-items: center; gap: 10px 22px; background: var(--sb-surface);
  border: 1px solid var(--sb-line); border-radius: 14px; padding: 0.8rem 1.1rem; margin-bottom: 1rem; }
.sb-strip-item { display: flex; flex-direction: column; }
.sb-strip-item small { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: var(--sb-muted); }
.sb-strip-item span { font-weight: 650; color: var(--sb-ink); font-size: 0.95rem; }

/* home stepper */
.sb-steps { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
.sb-stepcard { background: var(--sb-surface); border: 1px solid var(--sb-line); border-radius: 16px; padding: 1.3rem; }
.sb-stepcard em { font-style: normal; font-size: 0.8rem; font-weight: 800; color: var(--sb-accent-2); letter-spacing: 0.05em; }
.sb-stepcard b { display: block; font-size: 1.05rem; color: var(--sb-ink); margin: 0.35rem 0 0.3rem; }
.sb-stepcard span { color: var(--sb-ink-2); font-size: 0.9rem; line-height: 1.5; }

/* readiness ring */
.sb-ring-wrap { display: flex; align-items: center; gap: 1.2rem; background: var(--sb-surface); border: 1px solid var(--sb-line);
  border-radius: 16px; padding: 1.2rem 1.4rem; }
.sb-ring { width: 104px; height: 104px; border-radius: 50%; flex: none; display: flex; align-items: center; justify-content: center; }
.sb-ring-inner { width: 78px; height: 78px; border-radius: 50%; background: var(--sb-surface); display: flex; align-items: center;
  justify-content: center; font-size: 1.45rem; font-weight: 800; color: var(--sb-ink); }
.sb-ring-text b { display: block; font-size: 1.05rem; color: var(--sb-ink); }
.sb-ring-text span { color: var(--sb-ink-2); font-size: 0.88rem; }

/* learning-path timeline */
.sb-timeline { position: relative; padding-left: 0; }
.sb-timeline .sb-step { position: relative; }
.sb-timeline .sb-step:not(:last-child)::after { content: ""; position: absolute; left: calc(1.2rem + 16px); top: 52px;
  bottom: -12px; width: 2px; background: #d9e6f7; }
.sb-step-num { position: relative; z-index: 1; }
.sb-results-meta { color: var(--sb-ink-2); font-size: 0.88rem; margin: 0.2rem 0 0.8rem; }

@media (max-width: 760px) {
  .sb-featured { grid-template-columns: 1fr; }
  .sb-featured-side { border-left: none; padding-left: 0; border-top: 1px solid var(--sb-line); padding-top: 1rem; }
  .sb-steps { grid-template-columns: 1fr; }
}

/* journey stepper */
.sb-journey { display: flex; align-items: center; gap: 10px; margin: 0 0 1.4rem; flex-wrap: wrap; }
.sb-journey i { flex: 1; min-width: 18px; height: 2px; background: var(--sb-line); }
.sb-jstep { display: flex; align-items: center; gap: 8px; font-size: 0.85rem; font-weight: 600; color: var(--sb-muted); }
.sb-jstep span { width: 26px; height: 26px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
  font-size: 0.78rem; font-weight: 700; border: 2px solid var(--sb-line); background: var(--sb-surface); color: var(--sb-muted); }
.sb-jstep-now { color: var(--sb-ink); }
.sb-jstep-now span { background: var(--sb-accent); border-color: var(--sb-accent); color: #fff; }
.sb-jstep-done { color: var(--sb-good); }
.sb-jstep-done span { background: var(--sb-good-soft); border-color: #cbe8d5; color: var(--sb-good); }

/* technique tag */
.sb-tech { display: flex; align-items: center; gap: 10px; font-size: 0.86rem; color: var(--sb-ink-2); margin: 0.3rem 0 1rem; }
.sb-tech span { flex: none; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;
  color: #5b21b6; background: #f3effd; border: 1px solid #e2d8fb; border-radius: 6px; padding: 3px 8px; }

/* white surfaces for forms, bordered containers and expanders */
[data-testid="stForm"], [data-testid="stVerticalBlockBorderWrapper"] { background: var(--sb-surface); border-radius: 16px; }
[data-testid="stExpander"] details { background: var(--sb-surface); border-radius: 12px; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, subtitle: str = "") -> None:
    html = f'<div class="sb-eyebrow">{escape(eyebrow)}</div><div class="sb-title">{escape(title)}</div>'
    if subtitle:
        html += f'<div class="sb-subtitle">{escape(subtitle)}</div>'
    st.markdown(html, unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(f'<div class="sb-section">{escape(title)}</div>', unsafe_allow_html=True)


def stat_tiles(items: list[tuple[str, str]]) -> None:
    tiles = "".join(
        f'<div class="sb-tile"><div class="sb-tile-value">{escape(value)}</div>'
        f'<div class="sb-tile-label">{escape(label)}</div></div>'
        for label, value in items
    )
    st.markdown(f'<div class="sb-tiles">{tiles}</div>', unsafe_allow_html=True)


def chips(values: list[str], kind: str = "") -> str:
    css = f"sb-chip sb-chip-{kind}" if kind else "sb-chip"
    return "".join(f'<span class="{css}">{escape(str(v))}</span>' for v in values)


def format_rating(rating: float) -> str:
    return f"★ {rating:.1f}" if pd.notna(rating) else "Unrated"


def course_meta(row: pd.Series) -> str:
    return " · ".join([escape(str(row["provider"])), escape(str(row["difficulty"])), format_rating(row["rating"])])


def short_reasons(reasons: list[str], limit: int = 2) -> list[str]:
    """The most useful reasons for a card. Rating and taught skills are already shown as pills/chips."""
    useful = [r for r in reasons if not r.startswith(("Rated ", "No rating", "Teaches ", "Directly teaches "))]
    return useful[:limit]


def _match_html(score, score_label: str) -> str:
    if pd.isna(score):
        return ""
    width = max(0.0, min(1.0, float(score))) * 100
    tip = f"Relative {score_label} score {float(score):.2f} (not a probability)"
    return (f'<div class="sb-match" title="{escape(tip)}"><span class="sb-match-bar"><i style="width:{width:.0f}%">'
            f'</i></span>{escape(score_label)}</div>')


def _pills_html(row: pd.Series) -> str:
    return (f'<span class="sb-pill sb-pill-level">{escape(str(row["difficulty"]))}</span>'
            f'<span class="sb-pill">{escape(format_rating(row["rating"]))}</span>')


def _reason_items(reasons: list[str]) -> str:
    """Reasons as list items; cautions (e.g. a missing prerequisite) get a note marker instead of a tick."""
    return "".join(f'<li class="sb-note">{escape(r)}</li>' if r.startswith(CAVEAT_PREFIX) else f"<li>{escape(r)}</li>"
                   for r in reasons)


def _list_or_empty(value) -> list:
    return value if isinstance(value, list) else []


def _card_html(row: pd.Series, score_label: str, description: str = "") -> str:
    matched = _list_or_empty(row.get("matched_skills"))
    reasons = short_reasons(_list_or_empty(row.get("explanation")))
    url = escape(str(row["url"]))
    return (
        f'<div class="sb-card"><div class="sb-card-head"><div class="sb-provider">{escape(str(row["provider"]))}</div>'
        f'{_match_html(row.get("score", float("nan")), score_label)}</div>'
        f'<a class="sb-course-title" href="{url}" target="_blank" rel="noopener">{escape(str(row["title"]))}</a>'
        f'<div class="sb-pills">{_pills_html(row)}</div>'
        + (f'<p class="sb-desc sb-desc-short">{escape(description)}</p>' if description else "")
        + (f'<div class="sb-chips">{chips(matched[:3], "accent")}</div>' if matched else "")
        + (f'<ul class="sb-why">{_reason_items(reasons)}</ul>' if reasons else "")
        + f'<a class="sb-link" href="{url}" target="_blank" rel="noopener">View on Coursera ↗</a></div>'
    )


def course_grid(table: pd.DataFrame, score_label: str = "match", descriptions: dict[str, str] | None = None) -> None:
    """Course cards in a responsive grid. Scores are relative ranking scores, shown as a bar."""
    descriptions = descriptions or {}
    cards = "".join(_card_html(row, score_label, descriptions.get(row["course_id"], "")) for _, row in table.iterrows())
    st.markdown(f'<div class="sb-grid">{cards}</div>', unsafe_allow_html=True)


def featured_card(row: pd.Series, description: str) -> None:
    """The top recommendation, shown larger with its description and every reason."""
    matched = _list_or_empty(row.get("matched_skills"))
    reasons = short_reasons(_list_or_empty(row.get("explanation")), limit=3)
    url = escape(str(row["url"]))
    side = (f'<div class="sb-label">Skills it builds</div><div class="sb-chips">{chips(matched[:5], "accent")}</div>'
            if matched else "")
    side += (f'<div class="sb-label">Why it fits you</div><ul class="sb-why">{_reason_items(reasons)}</ul>'
             if reasons else "")
    side += f'<a class="sb-cta" href="{url}" target="_blank" rel="noopener">View course on Coursera ↗</a>'
    st.markdown(
        f'<div class="sb-featured"><div><span class="sb-badge">★ Top pick for you</span>'
        f'<a class="sb-featured-title" href="{url}" target="_blank" rel="noopener">{escape(str(row["title"]))}</a>'
        f'<div class="sb-meta">{course_meta(row)}</div><p class="sb-desc">{escape(description)}</p></div>'
        f'<div class="sb-featured-side">{side}</div></div>',
        unsafe_allow_html=True,
    )


def readiness_ring(percent: float, have: int, total: int) -> str:
    angle = max(0.0, min(100.0, percent)) * 3.6
    return (f'<div class="sb-ring-wrap"><div class="sb-ring" style="background: conic-gradient(#2a78d6 {angle:.0f}deg, '
            f'#eeede8 0deg)"><div class="sb-ring-inner">{percent:.0f}%</div></div><div class="sb-ring-text">'
            f'<b>Career readiness</b><span>You have {have} of the {total} skills this goal needs.</span></div></div>')


def summary_strip(items: list[tuple[str, str]]) -> None:
    html = "".join(f'<div class="sb-strip-item"><small>{escape(k)}</small><span>{escape(v)}</span></div>' for k, v in items)
    st.markdown(f'<div class="sb-strip">{html}</div>', unsafe_allow_html=True)


def course_rows_html(table: pd.DataFrame) -> str:
    """Compact list of courses (title link + meta line)."""
    return "".join(
        f'<div class="sb-row"><div><a class="sb-row-title" href="{escape(str(r["url"]))}" target="_blank" rel="noopener">'
        f'{escape(str(r["title"]))}</a><div class="sb-meta">{course_meta(r)}</div></div></div>'
        for _, r in table.iterrows()
    )



def snippet(text: str, limit: int = 240) -> str:
    """First sentences of a description, cut at a word boundary."""
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(",;:")
    return cut + "…"


# ------------------------------------------------------------ journey & technique tags
JOURNEY = [("profile", "Your profile"), ("skill-gap", "Skill gaps"), ("recommendations", "Courses for you"),
           ("learning-path", "Learning path")]


def journey(current: str) -> None:
    """The four-step flow (profile -> gaps -> courses -> path) with the current step highlighted."""
    keys = [key for key, _ in JOURNEY]
    position = keys.index(current)
    steps = []
    for i, (key, label) in enumerate(JOURNEY):
        state = "done" if i < position else "now" if i == position else "next"
        mark = "✓" if state == "done" else str(i + 1)
        steps.append(f'<div class="sb-jstep sb-jstep-{state}"><span>{mark}</span>{escape(label)}</div>')
    st.markdown(f'<div class="sb-journey">{"<i></i>".join(steps)}</div>', unsafe_allow_html=True)


def technique(name: str, description: str) -> None:
    """A small tag naming the recommendation technique behind what is shown, with a plain-language line."""
    st.markdown(f'<div class="sb-tech"><span>{escape(name)}</span>{escape(description)}</div>', unsafe_allow_html=True)
