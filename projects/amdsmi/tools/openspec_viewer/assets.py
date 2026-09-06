# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Inlined CSS and JS for the generated page.

Kept in one module so the visual language lives in a single place. The page is
self-contained and offline: nothing here may reference a CDN, a webfont or any
other external URL.

The language is an instrument panel: monospace for anything with an identity or
a measurement, sans for normative prose, serif for the arguing prose of a
change proposal, and a graduated scale as the recurring motif -- the rule under
the masthead, the sparkline in the capability index, and the task tape that
measures a change's progress are all the same object at different sizes.

Five hues, one meaning each, and no sixth:

===========  =================================================================
violet       the spec surface: a capability, a requirement, a cross reference
blue         a condition: the WHEN half of a scenario, an outgoing reference
green        present and asserted: THEN, an ADDED requirement, a done task
amber        under revision: a MODIFIED requirement
red          negated or withdrawn: SHALL NOT, a REMOVED requirement
===========  =================================================================
"""

# --------------------------------------------------------------------------
# render: css
# --------------------------------------------------------------------------

_LIGHT = """
  --bg:#f1f4f9; --panel:#ffffff; --raise:#e8edf5; --line:#d3dce8; --line-2:#e4eaf2;
  --fg:#121a25; --dim:#54637a; --faint:#8494a8; --tick:#c2cddc;
  --cap:#5b3fd4; --when:#0b5fa8; --ok:#166f3f; --mod:#9a5a05; --no:#b3271a;
  --cap-bg:rgba(91,63,212,.09); --when-bg:rgba(11,95,168,.09);
  --ok-bg:rgba(22,111,63,.10); --mod-bg:rgba(154,90,5,.10); --no-bg:rgba(179,39,26,.09);
  --on:#ffffff;
  --mark:#ffe08a; --mark-fg:#3a2c00; --shadow:0 1px 2px rgba(16,24,40,.06);
"""
_DARK = """
  --bg:#0c1016; --panel:#131a23; --raise:#18212d; --line:#26313f; --line-2:#1c2531;
  --fg:#d8dfe9; --dim:#8b98ab; --faint:#5c6a7d; --tick:#33404f;
  --cap:#a294ff; --when:#5cb0f7; --ok:#5ace8b; --mod:#efa845; --no:#f2705f;
  --cap-bg:rgba(162,148,255,.13); --when-bg:rgba(92,176,247,.12);
  --ok-bg:rgba(90,206,139,.12); --mod-bg:rgba(239,168,69,.12); --no-bg:rgba(242,112,95,.13);
  --on:#0c1016;
  --mark:#6a5312; --mark-fg:#ffe9a8; --shadow:none;
"""

# ---------------------------------------------------------------------------
# the mark
# ---------------------------------------------------------------------------

# A bracket pair enclosing a requirement and its two scenarios.
#
# The brackets are the corpus's own syntax -- a cross reference is written
# [capability-id] -- and the three rules inside are the structure every spec
# page has: a requirement, then WHEN, then THEN, narrowing as they nest. The
# lower two carry the same blue and green those steps carry in the body, so the
# mark is the page in miniature rather than decoration. Drawn on a 24 grid, so
# it still reads at 16px, which is the only size a favicon gets.

_LOGO = (
    '<svg class="mark" viewBox="0 0 24 24" aria-hidden="true" width="{w}" height="{w}">'
    '<path class="mk-b" d="M8 3.5H4.6v17H8"/>'
    '<path class="mk-b" d="M16 3.5h3.4v17H16"/>'
    '<path class="mk-r" d="M8.6 8.4h7"/>'
    '<path class="mk-w" d="M8.6 12h5.2"/>'
    '<path class="mk-t" d="M8.6 15.6h3.4"/>'
    "</svg>"
)


def logo(size: int = 22) -> str:
    """The mark at a given pixel size."""
    return _LOGO.format(w=size)


def favicon() -> str:
    """The same mark as a data-URI favicon: no file, no request."""
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' "
        "fill='none' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>"
        "<path stroke='%235b3fd4' d='M8 3.5H4.6v17H8'/>"
        "<path stroke='%235b3fd4' d='M16 3.5h3.4v17H16'/>"
        "<path stroke='%2364748b' d='M8.6 8.4h7'/>"
        "<path stroke='%230b5fa8' d='M8.6 12h5.2'/>"
        "<path stroke='%23166f3f' d='M8.6 15.6h3.4'/>"
        "</svg>"
    )
    return "data:image/svg+xml," + (
        svg.replace("<", "%3C").replace(">", "%3E").replace('"', "%22").replace("#", "%23")
    )


CSS = (
    """
:root{
"""
    + _LIGHT
    + """
  --mono:ui-monospace,"SFMono-Regular","SF Mono","Cascadia Mono","Roboto Mono",
    Menlo,Consolas,"DejaVu Sans Mono",monospace;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",
    "DejaVu Sans",Arial,sans-serif;
  --serif:"Iowan Old Style","Palatino Linotype",Palatino,"Book Antiqua",
    Georgia,"DejaVu Serif",serif;
}
@media (prefers-color-scheme:dark){ :root{ color-scheme:dark;
"""
    + _DARK
    + """ }}
:root[data-theme=light]{ color-scheme:light;
"""
    + _LIGHT
    + """ }
:root[data-theme=dark]{ color-scheme:dark;
"""
    + _DARK
    + """ }

*{box-sizing:border-box}
[hidden]{display:none!important}
html{scroll-behavior:smooth; scroll-padding-top:24px}
body{margin:0;background:var(--bg);color:var(--fg);font-family:var(--sans);
  font-size:15px;line-height:1.62;-webkit-text-size-adjust:100%}
a{color:inherit}
::selection{background:var(--cap-bg)}
:focus-visible{outline:2px solid var(--when);outline-offset:2px;border-radius:3px}
@media (prefers-reduced-motion:reduce){html{scroll-behavior:auto}
  *{transition:none!important;animation:none!important}}

/* ---- shell ---- */
.shell{display:grid;grid-template-columns:274px minmax(0,1fr);
  max-width:1580px;margin:0 auto;gap:0}
.rail{position:sticky;top:0;align-self:start;height:100vh;overflow:auto;
  border-right:1px solid var(--line);padding:22px 18px 28px;background:var(--panel)}
main{min-width:0;padding:34px 40px 120px}
section{scroll-margin-top:18px}

/* ---- rail ---- */
.brand{font-family:var(--mono);font-size:13px;letter-spacing:.02em;
  color:var(--fg);text-decoration:none;display:block}
.brand b{font-weight:600}
.brand span{color:var(--faint);display:block;font-size:10.5px;letter-spacing:.18em;
  text-transform:uppercase;margin-top:3px}

/* ---- the mark ---- */
.mark{flex:0 0 auto;fill:none;stroke-width:2;stroke-linecap:round;
  stroke-linejoin:round;display:block}
.mark .mk-b{stroke:var(--cap)}
.mark .mk-r{stroke:var(--fg)}
.mark .mk-w{stroke:var(--when)}
.mark .mk-t{stroke:var(--ok)}
.brand{display:flex;align-items:center;gap:9px}
.brand .txt{display:block}

/* ---- project switcher: obvious, not loud ---- */
.switch{display:flex;flex-direction:column;gap:2px;margin:16px 0 4px;
  border:1px solid var(--line);border-radius:8px;padding:3px;background:var(--bg)}
.switch button{display:flex;flex-direction:column;gap:1px;text-align:left;cursor:pointer;
  background:none;border:0;border-radius:5px;padding:6px 9px;color:var(--dim);font:inherit}
.switch button:hover{background:var(--raise);color:var(--fg)}
.switch .pn{font-family:var(--mono);font-size:12.4px;letter-spacing:-.01em}
.switch .pc{font-family:var(--mono);font-size:9.5px;letter-spacing:.12em;
  text-transform:uppercase;color:var(--faint)}
.switch button.on{background:var(--panel);color:var(--fg);
  box-shadow:inset 0 0 0 1px var(--line),inset 3px 0 0 var(--cap)}
.switch button.on .pc{color:var(--dim)}

.search{margin:18px 0 14px;position:relative}
.search input{width:100%;padding:9px 10px 9px 30px;border-radius:7px;
  border:1px solid var(--line);background:var(--bg);color:var(--fg);
  font-family:var(--mono);font-size:12.5px}
.search input::placeholder{color:var(--faint)}
.search svg{position:absolute;left:9px;top:50%;transform:translateY(-50%);
  width:13px;height:13px;stroke:var(--faint);fill:none;stroke-width:2}
.search kbd{position:absolute;right:8px;top:50%;transform:translateY(-50%);
  font-family:var(--mono);font-size:10px;color:var(--faint);
  border:1px solid var(--line);border-radius:4px;padding:1px 4px;background:var(--panel)}
.hits{font-family:var(--mono);font-size:10.5px;color:var(--dim);
  letter-spacing:.06em;text-transform:uppercase;margin:-6px 0 12px;min-height:14px}
.rail h2{font-family:var(--mono);font-size:10px;letter-spacing:.2em;
  text-transform:uppercase;color:var(--faint);font-weight:600;
  margin:20px 0 8px;padding-bottom:6px;border-bottom:1px solid var(--line-2)}
.rail nav a{display:flex;gap:8px;align-items:baseline;text-decoration:none;
  padding:4px 7px;margin:0 -7px;border-radius:6px;font-size:12.7px;color:var(--dim)}
.rail nav a:hover{background:var(--raise);color:var(--fg)}
.rail nav a.on{color:var(--fg);background:var(--cap-bg);box-shadow:inset 2px 0 0 var(--cap)}
.rail nav a .nm{font-family:var(--mono);font-size:11.8px;flex:1;min-width:0;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.rail nav a .ct{font-family:var(--mono);font-size:10px;color:var(--faint);
  font-variant-numeric:tabular-nums}
.rail nav a.dim{opacity:.28}
.rail nav.cnav a{flex-wrap:wrap;row-gap:2px}
.rail nav.cnav .nm{flex:1 0 100%}
.rail nav.cnav .ledger{margin-left:0}
.rail nav.cnav .ct{margin-left:auto}
.railfoot{margin:18px 0 4px;padding-top:12px;border-top:1px solid var(--line-2);
  font-family:var(--mono);font-size:10px;line-height:1.7;color:var(--faint);
  word-break:break-all}
.railtail{margin-top:6px}
.tbtn{font-family:var(--mono);font-size:10px;letter-spacing:.12em;
  text-transform:uppercase;background:none;border:1px solid var(--line);
  color:var(--dim);border-radius:5px;padding:4px 8px;cursor:pointer}
.tbtn:hover{color:var(--fg);border-color:var(--dim)}

/* ---- masthead ---- */
.mast{border-bottom:1px solid var(--line);padding-bottom:20px;margin-bottom:30px;
  scroll-margin-top:18px}
.mast h1{font-family:var(--mono);font-weight:500;font-size:clamp(26px,4.2vw,46px);
  letter-spacing:-.025em;margin:0;line-height:1.02}
.mast h1 i{font-style:normal;color:var(--faint)}
.mast .sub{font-family:var(--mono);font-size:11px;letter-spacing:.2em;
  text-transform:uppercase;color:var(--faint);margin:0 0 12px}
.scale{display:flex;align-items:flex-end;gap:0;height:12px;margin:18px 0 16px}
.scale i{display:block;width:1px;background:var(--line);height:4px;margin-right:7px}
.scale i:nth-child(5n+1){height:11px;background:var(--dim)}
.stats{display:flex;flex-wrap:wrap;gap:10px 34px;font-family:var(--mono);
  align-items:flex-start}
.stats>div{padding:2px 0;min-width:0}
.stats b{display:block;font-size:25px;font-weight:500;letter-spacing:-.02em;
  font-variant-numeric:tabular-nums;line-height:1.2}
.stats span{font-size:10px;letter-spacing:.18em;text-transform:uppercase;color:var(--faint);
  display:block;white-space:nowrap}
.stats .c1 b{color:var(--cap)}
/* anything that needs width lives here, under the numbers, never inside one */
.gauges{display:flex;flex-wrap:wrap;gap:10px 40px;align-items:center;
  margin-top:18px;padding-top:14px;border-top:1px dotted var(--line)}
.gauge{display:flex;align-items:center;gap:12px;min-width:0}
.gauge:last-child{flex:1 1 320px}
.glab{font-family:var(--mono);font-size:10px;letter-spacing:.18em;text-transform:uppercase;
  color:var(--faint);white-space:nowrap}

h2.sec{font-family:var(--mono);font-size:11px;letter-spacing:.2em;
  text-transform:uppercase;color:var(--dim);font-weight:600;
  margin:0 0 14px;display:flex;align-items:center;gap:12px}
h2.sec em{font-style:normal;color:var(--faint);font-variant-numeric:tabular-nums}
h2.sec::after{content:"";flex:1;height:1px;background:var(--line);order:5}
h2.sec .key{order:9;display:flex;gap:5px}
.block{margin:0 0 46px}
.chg-lead{margin-bottom:10px}

/* ---- context ---- */
.ctx{background:var(--panel);border:1px solid var(--line);border-radius:10px;
  padding:20px 24px;box-shadow:var(--shadow)}
.ctx pre{margin:0;font-family:var(--mono);font-size:12.4px;line-height:1.66;
  white-space:pre-wrap;color:var(--dim)}
.ctx pre b{color:var(--fg);font-weight:600}
.empty{border:1px dashed var(--line);border-radius:10px;padding:20px 24px;
  color:var(--dim);font-size:13.5px;max-width:80ch;background:
  repeating-linear-gradient(135deg,transparent 0 9px,var(--line-2) 9px 10px)}
.empty code{font-family:var(--mono);font-size:12.4px;color:var(--fg)}

/* ---- diagrams: natural size, scrolled, never scaled down ---- */
.mapwrap,.flowwrap,.dlvwrap{background:var(--panel);border:1px solid var(--line);
  border-radius:10px;padding:6px 0 0;box-shadow:var(--shadow);overflow-x:auto}
.flowwrap{margin:0 0 18px;padding:10px 0 0}
.dlvwrap{padding:10px 0 0}
.map,.flow,.dlv{display:block;margin:0 auto;font-family:var(--mono)}
.map .nlabel,.flow .nlabel,.dlv .nlabel{font-size:12px;fill:var(--fg);letter-spacing:-.01em}
.map .nbox,.flow .nbox,.dlv .nbox{fill:var(--raise);stroke:var(--line);stroke-width:1;rx:5}
.map a:hover .nbox,.map a:focus .nbox,
.flow a:hover .nbox,.flow a:focus .nbox,
.dlv a:hover .nbox,.dlv a:focus .nbox{fill:var(--cap-bg);stroke:var(--cap)}
.map a:focus,.flow a:focus,.dlv a:focus{outline:none}
.map a:focus-visible .nbox,.flow a:focus-visible .nbox,
.dlv a:focus-visible .nbox{stroke:var(--cap);stroke-width:2}
.map .ntab,.flow .ntab,.dlv .ntab{fill:var(--cap);opacity:.55}
.map a:hover .ntab,.flow a:hover .ntab,.dlv a:hover .ntab{opacity:1}
.map .edge,.flow .edge,.dlv .edge{color:var(--faint)}
.map .wire,.flow .wire,.dlv .wire{stroke:currentColor;fill:none;stroke-width:1.4;
  stroke-linejoin:round}
.map .head,.flow .head,.dlv .head{fill:currentColor;stroke:none}
.map.act .edge,.flow.act .edge,.dlv.act .edge{opacity:.1}
.map.act .edge.out,.flow.act .edge.out,.dlv.act .edge.out{opacity:1;color:var(--when)}
.map.act .edge.in,.flow.act .edge.in,.dlv.act .edge.in{opacity:1;color:var(--cap)}
.map.act .edge.out .wire,.map.act .edge.in .wire,
.flow.act .edge.out .wire,.flow.act .edge.in .wire,
.dlv.act .edge.out .wire,.dlv.act .edge.in .wire{stroke-width:2}
.map.act a,.flow.act a,.dlv.act a{opacity:.34}
.map.act a.lit,.map.act a.self,.flow.act a.lit,.flow.act a.self,
.dlv.act a.lit,.dlv.act a.self{opacity:1}
.map.act a.self .nbox,.flow.act a.self .nbox,
.dlv.act a.self .nbox{stroke:var(--cap);stroke-width:1.6;fill:var(--cap-bg)}
/* held focus: a click parks the highlight, so the rest goes further down than
   it does on hover and the held node keeps a ring while you read elsewhere */
.map.held a,.flow.held a,.dlv.held a{opacity:.16}
.map.held .edge,.flow.held .edge,.dlv.held .edge{opacity:.06}
.map.held a.lit,.flow.held a.lit,.dlv.held a.lit{opacity:1}
.map.held a.self .nbox,.flow.held a.self .nbox,.dlv.held a.self .nbox{stroke-width:2.4}
.map.held a.self .ntab,.flow.held a.self .ntab,.dlv.held a.self .ntab{opacity:1}
.map.held a.held .nbox,.flow.held a.held .nbox,.dlv.held a.held .nbox{stroke:var(--cap);
  stroke-width:2.4;fill:var(--cap-bg)}
.map.held a.held,.flow.held a.held,.dlv.held a.held{opacity:1}
.maplegend,.flowlegend{display:flex;flex-wrap:wrap;gap:6px 20px;padding:12px 18px 14px;
  font-family:var(--mono);font-size:10.5px;color:var(--faint);
  letter-spacing:.04em;border-top:1px solid var(--line-2);margin-top:4px}
.maplegend span,.flowlegend span{display:inline-flex;align-items:center;gap:6px}
.maplegend i,.flowlegend i{width:15px;height:2px;display:inline-block;background:currentColor}
.maplegend .o{color:var(--when)} .maplegend .i{color:var(--cap)}
.maplegend .now{color:var(--cap)}
/* ---- delivery: stage bands behind a left-to-right flow ---- */
.dlv .stage{fill:var(--raise);stroke:none}
.dlv .stagelab{font-size:9px;fill:var(--faint);letter-spacing:.16em;text-transform:uppercase}
.dlv .nnote{font-size:9.5px;fill:var(--faint);letter-spacing:0}
.dlv .elabel{font-size:9.5px;fill:var(--dim);letter-spacing:0}

/* ---- change flow: proposals on top, the capabilities they land on below ----
   node kinds carry the same vocabulary as the requirement rails, so a dashed
   outline means the same thing in the diagram as it does in the page. */
.flow{font-family:var(--mono)}
.flow .flabel{font-size:12px;fill:var(--fg);letter-spacing:-.01em}
.flow .fcaption{font-size:9px;fill:var(--faint);letter-spacing:.16em;
  text-transform:uppercase}
.flow .fbox{fill:var(--panel);stroke:var(--line);stroke-width:1}
.flow .frail{fill:var(--line)}
.flow a{cursor:pointer}
.flow a:focus{outline:none}
.flow a:hover .fbox,.flow a:focus-visible .fbox{stroke-width:2}
/* a proposal */
.flow .k-change .fbox{fill:var(--raise);stroke:var(--fg)}
.flow .k-change .frail{fill:var(--fg)}
.flow .k-change .flabel{font-weight:600}
/* a capability that already exists in the baseline */
.flow .k-spec .fbox{stroke:var(--cap)}
.flow .k-spec .frail{fill:var(--cap)}
/* a capability this proposal would create: dashed, because it is not there yet */
.flow .k-new .fbox{fill:none;stroke:var(--ok)}
.flow .k-new .frail{fill:var(--ok)}
/* owned by another project on this page */
.flow .k-ghost .fbox{fill:none;stroke:var(--faint)}
.flow .k-ghost .frail{fill:var(--faint)}
.flow .k-ghost .flabel{fill:var(--dim);font-style:italic}
/* several proposals converge here */
.flow .fconv{fill:var(--mod);stroke:var(--panel);stroke-width:1.5}
.flow .fconvn{fill:var(--on);font-size:9.5px;font-weight:700}
/* edges, coloured by what the change does */
.flow .fedge{color:var(--faint)}
.flow .fwire{stroke:currentColor;fill:none;stroke-width:1.4;stroke-linejoin:round}
.flow .fhead{fill:currentColor;stroke:none}
.flow .fbreak{stroke:currentColor;stroke-width:1.4}
.flow .fglyphbg{fill:var(--panel);stroke:currentColor;stroke-width:1}
.flow .fglyph{fill:currentColor;font-size:11px;font-weight:700}
.flow .e-added{color:var(--ok)}
.flow .e-modified{color:var(--mod)}
.flow .e-removed{color:var(--no)}
.flow .e-renamed{color:var(--cap)}
.flow .e-feeds{color:var(--when)}
.flow.act .fedge{opacity:.12}
.flow.act .fedge.in,.flow.act .fedge.out{opacity:1}
.flow.act .fedge.out .fwire{stroke-width:2.4}
.flow.act .fedge.in .fwire{stroke-width:2.4;stroke-dasharray:none}
.flow.act a,.flow.act>g{opacity:.3}
.flow.act a.lit,.flow.act a.self,.flow.act g.lit,.flow.act g.self{opacity:1}
.flow.act .self .fbox{stroke-width:2.4}
.flowlegend .k-change i,.flowlegend .k-spec i,.flowlegend .k-new i,
.flowlegend .k-ghost i{height:11px;width:16px;border-radius:2px;background:none;
  border:1.5px solid currentColor}
.flowlegend .k-change{color:var(--fg)}
.flowlegend .k-change i{background:var(--raise)}
.flowlegend .k-spec{color:var(--cap)}
.flowlegend .k-spec i{background:var(--panel)}
.flowlegend .k-new{color:var(--ok)}
.flowlegend .k-new i{border-style:dashed}
.flowlegend .k-ghost{color:var(--faint)}
.flowlegend .k-ghost i{border-style:dashed}
.flowlegend .e-added{color:var(--ok)}
.flowlegend .e-modified{color:var(--mod)}
.flowlegend .e-modified i{background:repeating-linear-gradient(90deg,
  currentColor 0 7px,transparent 7px 10px)}
.flowlegend .e-removed{color:var(--no)}
.flowlegend .e-removed i{background:repeating-linear-gradient(90deg,
  currentColor 0 2px,transparent 2px 5px)}
.flowlegend .e-feeds{color:var(--when)}
.flowlegend .e-feeds i{background:repeating-linear-gradient(90deg,
  currentColor 0 9px,transparent 9px 12px,currentColor 12px 14px,transparent 14px 17px)}

/* ---- capability index ---- */
.idx,.cidx{border:1px solid var(--line);border-radius:10px;overflow:hidden;
  background:var(--panel);box-shadow:var(--shadow)}
.idx a{display:grid;grid-template-columns:minmax(190px,270px) minmax(0,1fr) 128px 92px;
  gap:18px;align-items:center;padding:13px 20px;text-decoration:none;
  border-top:1px solid var(--line-2)}
.idx a:first-child,.cidx a:first-child{border-top:none}
.idx a:hover,.cidx a:hover{background:var(--raise)}
.idx .nm,.cidx .nm{font-family:var(--mono);font-size:12.8px;color:var(--cap);font-weight:500}
.idx .ds,.cidx .ds{color:var(--dim);font-size:13px;overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap}
.idx .ct{font-family:var(--mono);font-size:10.5px;color:var(--faint);
  letter-spacing:.04em;text-align:right;font-variant-numeric:tabular-nums}
.idx .ct b{color:var(--fg);font-weight:500}
.strip{display:block;height:20px;width:128px}
.strip rect{fill:var(--cap);opacity:.42}
.idx a:hover .strip rect{opacity:.85}

/* ---- change index ---- */
.cidx a{display:grid;grid-template-columns:minmax(180px,250px) minmax(0,1fr) auto 160px;
  gap:18px;align-items:center;padding:13px 20px;text-decoration:none;
  border-top:1px solid var(--line-2)}
.cidx .nm{color:var(--fg)}

/* ---- capability ---- */
.cap{margin:0 0 12px;padding-top:14px}
.cap-head{display:flex;align-items:baseline;gap:8px;flex-wrap:wrap;
  border-top:2px solid var(--cap);padding-top:14px}
.cap-head h2{font-family:var(--mono);font-size:clamp(18px,2.3vw,25px);font-weight:500;
  letter-spacing:-.02em;margin:0;color:var(--fg)}
.cap-head .meta{font-family:var(--mono);font-size:10.5px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--faint);margin-left:auto}
.purpose{margin:14px 0 26px;max-width:76ch;color:var(--dim)}
.purpose p{margin:0 0 12px}
.reflist{margin:0 0 26px}
.rrow{display:grid;grid-template-columns:112px minmax(0,1fr);gap:6px 14px;
  align-items:baseline;padding:3px 0}
.rrow>span{font-family:var(--mono);font-size:10px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--faint)}
.rchips{display:flex;flex-wrap:wrap;gap:6px}
.reflist a{font-family:var(--mono);font-size:11.5px;text-decoration:none;
  color:var(--cap);background:var(--cap-bg);border-radius:20px;padding:2px 10px}
.reflist a:hover{filter:brightness(1.12);text-decoration:underline}
@media (max-width:1080px){.rrow{grid-template-columns:minmax(0,1fr);gap:2px}}

/* ---- requirement ---- */
.req{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--cap);
  border-radius:8px;margin:0 0 16px;box-shadow:var(--shadow)}
.req-head{display:flex;gap:12px;align-items:baseline;padding:15px 20px 0;flex-wrap:wrap}
.req-head .ord{font-family:var(--mono);font-size:11px;font-weight:600;color:var(--cap);
  background:var(--cap-bg);border-radius:4px;padding:2px 7px;letter-spacing:.04em;
  white-space:nowrap;position:relative;top:-1px}
.req-head h3{margin:0;font-size:17px;font-weight:600;letter-spacing:-.01em;flex:1;
  min-width:min(100%,18ch)}
.anchor{font-family:var(--mono);font-size:12px;color:var(--faint);text-decoration:none;
  opacity:0;padding:0 4px}
.req:hover .anchor,.scn:hover .anchor,.cap:hover>.cap-head>.anchor,
.chg:hover .chg-id .anchor,.dspec:hover>.dspec-head>.anchor,.anchor:focus{opacity:1}
.anchor:hover{color:var(--cap)}
.prose{padding:8px 20px 4px;max-width:82ch}
.prose p{margin:0 0 11px}
.prose ul,.prose ol{margin:0 0 11px;padding-left:22px}
.prose li{margin:0 0 4px}
.prose code,.step code,.purpose code,.idx code,.tt code{font-family:var(--mono);
  font-size:.855em;background:var(--raise);border:1px solid var(--line-2);
  border-radius:4px;padding:.06em .34em;word-break:break-word}
.prose pre{background:var(--raise);border:1px solid var(--line-2);border-radius:6px;
  padding:12px 14px;overflow-x:auto;font-size:12.4px;line-height:1.55}
.prose pre code{background:none;border:none;padding:0;font-size:1em}
/* normative verbs are emphasis, not a colour -- except a prohibition */
.kw{font-family:var(--mono);font-size:.86em;font-weight:600;color:var(--fg);
  letter-spacing:.05em;border-bottom:1.5px solid var(--line);padding-bottom:.05em}
.kw.no{color:var(--no);border-bottom-color:var(--no)}
.xref{font-family:var(--mono);font-size:.87em;color:var(--cap);text-decoration:none;
  border-bottom:1px dashed currentColor}
.xref:hover{background:var(--cap-bg)}
.tw{overflow-x:auto;margin:0 0 13px;max-width:100%}
table{border-collapse:collapse;font-size:13px;width:100%;min-width:420px}
th,td{border:1px solid var(--line);padding:7px 11px;text-align:left;vertical-align:top}
th{background:var(--raise);font-family:var(--mono);font-size:10.5px;
  letter-spacing:.12em;text-transform:uppercase;color:var(--dim);font-weight:600;
  white-space:nowrap}
tbody tr:nth-child(even){background:var(--raise)}

/* ---- scenario ---- */
.scns{padding:4px 14px 14px 14px}
.scn{border:1px solid var(--line-2);border-radius:6px;background:var(--bg);
  margin:0 0 8px;padding:10px 14px 11px}
.scn:last-child{margin-bottom:0}
.scn-head{display:flex;gap:10px;align-items:baseline}
.scn-head .ord{font-family:var(--mono);font-size:10.5px;color:var(--faint);
  font-variant-numeric:tabular-nums;white-space:nowrap}
.scn-head h4{margin:0;font-size:13.6px;font-weight:600;color:var(--fg);flex:1;
  letter-spacing:-.005em}
.steps{list-style:none;margin:8px 0 0;padding:0}
.step{display:grid;grid-template-columns:56px minmax(0,1fr);gap:12px;
  padding:4px 0;align-items:baseline;font-size:13.6px;color:var(--dim)}
.step .k{font-family:var(--mono);font-size:9.5px;letter-spacing:.14em;font-weight:600;
  text-align:center;border-radius:3px;padding:2px 0;text-transform:uppercase}
.step.when .k{color:var(--when);border:1px solid var(--when);background:none}
.step.then .k{color:var(--on);background:var(--ok);border:1px solid var(--ok)}
.step.other .k{color:var(--faint);border:1px dashed var(--line);background:none}
.step.plain .k{visibility:hidden}
.step.when .t{color:var(--dim)}
.step.then .t{color:var(--fg)}
.step .t{min-width:0}

/* ---- the delta vocabulary: colour AND sigil AND border, everywhere ---- */
.ledger{display:inline-flex;gap:5px;flex-wrap:wrap;vertical-align:middle}
.dl,.delta{font-family:var(--mono);font-variant-numeric:tabular-nums;
  white-space:nowrap;border-radius:3px;font-weight:600}
.dl{font-size:11px;padding:1px 6px 1px 4px;display:inline-flex;gap:3px;align-items:baseline;
  border:1px solid currentColor;letter-spacing:.02em}
.dl b{font-weight:700;font-size:12px}
.delta{font-size:9.5px;letter-spacing:.14em;padding:2px 7px;text-transform:uppercase;
  display:inline-flex;gap:4px;align-items:baseline}
.delta b{font-size:11px;font-weight:700}
.dl.added,.dl.renamed:not(.x){color:var(--ok)}
.dl.added{background:var(--ok-bg)}
.dl.modified{color:var(--mod);background:var(--mod-bg)}
.dl.removed{color:var(--no);background:var(--no-bg)}
.dl.renamed{color:var(--cap);background:var(--cap-bg)}
.delta.added{color:var(--on);background:var(--ok)}
.delta.modified{color:var(--on);background:var(--mod)}
.delta.removed{color:var(--on);background:var(--no)}
.delta.renamed{color:var(--on);background:var(--cap)}
.req.d-added{border-left:3px solid var(--ok)}
.req.d-modified{border-left:5px double var(--mod)}
.req.d-removed{border-left:3px dashed var(--no)}
.req.d-renamed{border-left:3px dotted var(--cap)}
.req.d-added .ord{color:var(--ok);background:var(--ok-bg)}
.req.d-modified .ord{color:var(--mod);background:var(--mod-bg)}
.req.d-removed .ord{color:var(--no);background:var(--no-bg)}
.req.d-removed .req-head h3{text-decoration:line-through;text-decoration-thickness:1px;
  text-decoration-color:var(--no)}

/* ---- the task tape: the masthead ruler, used to measure ---- */
.prog{display:inline-flex;align-items:center;gap:9px;min-width:120px;flex:1}
.tape{display:flex;gap:5px;flex:1;height:10px;min-width:48px}
.tape .seg{display:grid;grid-auto-flow:column;grid-auto-columns:minmax(0,1fr);gap:1.5px}
.tape i{display:block;border:1px solid var(--tick);border-radius:1px;min-width:1px}
.tape i.on{background:var(--ok);border-color:var(--ok)}
.pnum{font-family:var(--mono);font-size:10.5px;color:var(--faint);
  font-variant-numeric:tabular-nums;white-space:nowrap}
.pnum b{color:var(--fg);font-weight:600}
.prog.sm{gap:7px;min-width:96px;flex:0 0 auto;width:150px}
.prog.sm .tape{height:7px;gap:3px}
.prog.sm .pnum{font-size:10px}
.prog.wide{min-width:220px}
.prog.wide .tape{height:13px;gap:7px}
.prog.wide .pnum{font-size:12px}
/* too many tasks to draw one countable cell each: one proportional segment per
   group instead, filled by that group's progress. Segment widths are the same
   flex ratios the cells use, so the green still covers done/total of the
   track and the two drawings agree. The outline is an inset shadow rather than
   a border, so it costs no layout width: a fill of 100% then reaches both
   edges exactly as a filled cell does, and the green covers done/total of the
   track to the pixel. */
.tape.bar .seg{display:block;overflow:hidden;min-width:2px;border-radius:1px;
  box-shadow:inset 0 0 0 1px var(--tick)}
.tape.bar .seg i{display:block;border:0;border-radius:1px 0 0 1px;height:100%;
  min-width:0}

/* ---- change ---- */
.chg{margin:0 0 40px;padding-top:16px}
.drule{display:flex;gap:2px;height:2px;margin-bottom:14px;border-radius:2px;overflow:hidden}
.drule span{display:block}
.drule .added{background:var(--ok)}
.drule .modified{background:var(--mod)}
.drule .removed{background:var(--no)}
.drule .renamed{background:var(--cap)}
.drule .none{flex:1;background:var(--line)}
.chg-id{display:flex;align-items:baseline;gap:6px}
.chg-head h2{font-family:var(--mono);font-size:clamp(18px,2.3vw,25px);font-weight:500;
  letter-spacing:-.02em;margin:0;color:var(--fg)}
.chg-sub{margin:4px 0 0;font-size:15px;color:var(--dim);max-width:70ch}
.chg-bar{display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin:14px 0 20px;
  padding:9px 14px;border:1px solid var(--line);border-radius:8px;background:var(--panel)}
.cmeta{font-family:var(--mono);font-size:10.5px;letter-spacing:.12em;
  text-transform:uppercase;color:var(--faint)}
.chg-bar .prog{min-width:180px}

/* change documents: label in the gutter, argument in serif */
.doc{border-top:1px solid var(--line-2);margin:0 0 4px}
.doc>summary{list-style:none;cursor:pointer;display:flex;align-items:baseline;gap:12px;
  padding:11px 0}
.doc>summary::-webkit-details-marker{display:none}
.doc>summary::before{content:"\\25B8";color:var(--faint);font-size:11px;width:9px;
  transition:transform .12s ease}
.doc[open]>summary::before{transform:rotate(90deg)}
.doc .dn{font-family:var(--mono);font-size:10px;letter-spacing:.2em;text-transform:uppercase;
  color:var(--dim);font-weight:600}
.doc .dsum{font-size:12px;color:var(--faint);overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap;flex:1;min-width:0}
.doc[open] .dsum{opacity:0}
.dbody{padding:2px 0 18px}
.dsec{display:grid;grid-template-columns:132px minmax(0,1fr);gap:4px 24px;
  padding:10px 0;border-top:1px dotted var(--line-2)}
.dsec:first-child{border-top:none;padding-top:0}
.dlab{font-family:var(--mono);font-size:10px;letter-spacing:.16em;text-transform:uppercase;
  color:var(--cap);font-weight:600;padding-top:5px}
.dbody .prose{padding:0;max-width:74ch;font-family:var(--serif);font-size:15.5px;
  line-height:1.66;color:var(--dim)}
.dbody .prose strong{color:var(--fg)}
.dbody .prose code{font-family:var(--mono);font-size:.79em;padding:.02em .26em}
.dbody .prose pre{font-family:var(--mono)}
.chg-refs{margin:10px 0 0}
.chg-refs span{color:var(--faint)}
.dbody .prose>p:first-child{margin-top:0}
h5.dh{font-family:var(--sans);font-size:12.5px;font-weight:700;letter-spacing:.02em;
  color:var(--fg);margin:16px 0 7px;text-transform:none}
h5.dh:first-child{margin-top:0}

/* tasks */
.blab{font-family:var(--mono);font-size:10px;letter-spacing:.2em;text-transform:uppercase;
  color:var(--dim);font-weight:600;margin:18px 0 8px;display:flex;align-items:center;gap:10px}
.blab em{font-style:normal;color:var(--faint);font-variant-numeric:tabular-nums}
.blab::after{content:"";flex:1;height:1px;background:var(--line-2);order:5}
.tall{order:9;font-size:10px;letter-spacing:.1em;color:var(--faint);text-decoration:none;
  border-bottom:1px dotted var(--faint)}
.tall:hover{color:var(--fg)}
.tasks-block{margin:0 0 8px}
.phase{border:1px solid var(--line);border-radius:8px;background:var(--panel);
  margin:0 0 6px;box-shadow:var(--shadow)}
.phase>summary{list-style:none;cursor:pointer;display:flex;align-items:center;gap:14px;
  padding:9px 14px}
.phase>summary::-webkit-details-marker{display:none}
.phase>summary::before{content:"\\25B8";color:var(--faint);font-size:11px;width:9px;
  flex:0 0 auto}
.phase[open]>summary::before{transform:rotate(90deg)}
.phase>summary:hover{background:var(--raise)}
.phase .pn{font-family:var(--mono);font-size:12.4px;color:var(--fg);flex:1;min-width:0;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.tasks{list-style:none;margin:0;padding:2px 14px 12px 14px}
.task{display:grid;grid-template-columns:16px minmax(0,1fr);gap:9px;align-items:baseline;
  font-size:13.2px;color:var(--dim);padding:3px 0 3px calc(var(--d,0) * 18px);
  border-top:1px solid var(--line-2)}
.task:first-child{border-top:none}
.task .box{width:11px;height:11px;border:1px solid var(--tick);border-radius:2px;
  display:block;position:relative;top:2px}
.task.done .box{background:var(--ok);border-color:var(--ok)}
.task.done .box::after{content:"";position:absolute;left:3px;top:0;width:3px;height:7px;
  border:solid var(--on);border-width:0 1.6px 1.6px 0;transform:rotate(42deg)}
.task.done .tt{color:var(--faint)}
.task .tt{min-width:0}
.task .tt code{font-size:.83em}

/* delta specs */
.delta-block{margin-top:22px}
.dspec{margin:0 0 20px}
.dspec-head{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;
  border-top:1px solid var(--line);padding-top:10px;margin-bottom:12px}
.dspec-head h4{font-family:var(--mono);font-size:15px;font-weight:600;margin:0;
  letter-spacing:-.01em}
.dspec .purpose{margin:0 0 14px;max-width:76ch;font-size:14px}
.owner{font-family:var(--mono);font-size:10.5px;color:var(--cap);text-decoration:none;
  background:var(--cap-bg);border-radius:20px;padding:2px 9px}
.owner:hover{text-decoration:underline}
.dspec-head .anchor{margin-left:auto}

/* ---- provenance: a line, not a banner ---- */
.prov{display:flex;flex-wrap:wrap;align-items:baseline;gap:7px 30px;
  margin:54px 0 0;padding-top:14px;border-top:1px solid var(--line-2);
  font-family:var(--mono);font-size:10.5px;color:var(--faint)}
.prov .pi{display:flex;align-items:baseline;gap:9px;min-width:0}
.prov .pl{font-size:9.5px;letter-spacing:.18em;text-transform:uppercase;
  color:var(--faint);white-space:nowrap}
.prov .pv{color:var(--dim);min-width:0;overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap}
.prov a.pv{text-decoration:none;border-bottom:1px solid var(--line)}
.prov a.pv:hover{color:var(--fg);border-bottom-color:var(--dim)}

/* ---- search ---- */
mark.hl{background:var(--mark);color:var(--mark-fg);border-radius:2px;padding:0 1px}
.hide{display:none!important}
.nores{display:none;padding:26px 2px;color:var(--dim);font-family:var(--mono);font-size:13px}
body.searching .nores.on{display:block}
body.searching .oview,body.searching .chg-lead,body.searching .drule{display:none}
.flash{animation:flash 1.3s ease-out}
@keyframes flash{from{box-shadow:0 0 0 3px var(--cap)}to{box-shadow:0 0 0 3px transparent}}

/* ---- narrow ---- */
@media (max-width:1080px){
  .dsec{grid-template-columns:minmax(0,1fr);gap:2px}
  .dlab{padding-top:0}
  .cidx a{grid-template-columns:minmax(0,1fr) auto;gap:6px 14px;padding:12px 14px}
  .cidx .ds{grid-column:1/-1;white-space:normal;font-size:12.6px}
  .cidx .prog.sm{grid-column:1/-1;width:auto}
}
@media (max-width:1000px){
  .shell{grid-template-columns:1fr}
  .rail{position:sticky;top:0;height:auto;max-height:none;z-index:9;
    border-right:none;border-bottom:1px solid var(--line);padding:12px 18px}
  .rail .brand{display:inline-block}
  .rail .search{margin:10px 0 8px}
  .rail nav,.rail h2,.railfoot{display:none}
  .rail details nav,.rail details h2{display:block}
  .rail details{margin-top:4px}
  .switch{flex-direction:row;flex-wrap:wrap;margin:10px 0 0}
  .switch button{flex:1 1 auto}
  .rail summary{font-family:var(--mono);font-size:10.5px;letter-spacing:.16em;
    text-transform:uppercase;color:var(--dim);cursor:pointer;padding:4px 0}
  .rail details[open] nav{max-height:44vh;overflow:auto}
  main{padding:22px 18px 90px}
  .idx a{grid-template-columns:minmax(0,1fr) 84px;gap:4px 14px;padding:12px 14px}
  .idx .ds{grid-column:1/-1;white-space:normal;font-size:12.6px}
  .idx .strip{display:none}
  .step{grid-template-columns:50px minmax(0,1fr);gap:9px}
  .req-head,.prose{padding-left:14px;padding-right:14px}
  .scns{padding:4px 8px 10px}
  .chg-bar{gap:10px 14px}
  .stats{gap:10px 24px}
  .stats b{font-size:21px}
}
@media (min-width:1001px){.rail summary{display:none}.rail details{display:contents}}

/* ---- print ---- */
@media print{
  :root{--bg:#fff;--panel:#fff;--raise:#f4f4f4;--line:#bbb;--line-2:#ddd;--tick:#999;
    --fg:#000;--dim:#333;--faint:#666;--on:#fff;--shadow:none}
  body{font-size:10.5pt}
  .rail,.anchor,.maplegend,.tbtn,.nores,.tall,.switch{display:none!important}
  .shell{display:block;max-width:none}
  main{padding:0}
  .proj[hidden]{display:block!important}
  .req,.scn,.ctx,.idx,.cidx,.mapwrap,.flowwrap,.phase,.dsec{break-inside:avoid;box-shadow:none}
  .cap,.chg{break-before:page}
  .cap:first-of-type{break-before:auto}
  .req-head h3{font-size:13pt}
  a{text-decoration:none}
  .map .edge,.flow .edge{color:#666}
  .doc>summary::before,.phase>summary::before{content:""}
  .doc .dsum{display:none}
  .tape i{border-color:#666}
  .tape i.on{background:#000;border-color:#000}
  /* a shadow may not print; the bar's track needs a real border on paper */
  .tape.bar .seg{box-shadow:none;border:1px solid #666}
  .prov{color:#333}
  .prov a.pv{border-bottom:none}
  .dl,.delta{border:1px solid #000!important;background:none!important;color:#000!important}
}
"""
)

# --------------------------------------------------------------------------
# render: js
# --------------------------------------------------------------------------

JS = r"""
(function(){
var doc=document, body=doc.body;
var qa=function(s,r){return Array.prototype.slice.call((r||doc).querySelectorAll(s));};
var input=doc.getElementById('q'), hits=doc.getElementById('hits');

/* ---------- project panes ---------- */
var panes=qa('.proj'), rails=qa('.railproj'), tabs=qa('.switch button');
var multi=panes.length>1, cur=null;

function paneOf(el){
  while(el&&el!==doc){
    if(el.nodeType===1&&el.classList&&el.classList.contains('proj'))return el;
    el=el.parentNode;
  }
  return null;
}
function activate(slug){
  if(!multi||!slug||slug===cur)return;
  cur=slug;
  panes.forEach(function(p){p.hidden=p.dataset.p!==slug;});
  rails.forEach(function(p){p.hidden=p.dataset.p!==slug;});
  tabs.forEach(function(b){
    var on=b.dataset.go===slug;
    b.classList.toggle('on',on); b.setAttribute('aria-selected',on?'true':'false');
  });
}
function openAncestors(el){
  var n=el;
  while(n&&n!==body){ if(n.nodeName==='DETAILS')n.open=true; n=n.parentNode; }
}
function goTo(id,push){
  if(!id)return false;
  var el=doc.getElementById(id);
  if(!el)return false;
  var p=paneOf(el);
  if(p)activate(p.dataset.p);
  openAncestors(el);
  if(el.classList.contains('hide'))el.classList.remove('hide');
  var jump=function(){
    el.scrollIntoView({block:'start'});
    el.classList.add('flash');
    setTimeout(function(){el.classList.remove('flash');},1400);
  };
  if(window.requestAnimationFrame)requestAnimationFrame(jump); else jump();
  if(push&&location.hash!=='#'+id)history.pushState(null,'','#'+id);
  return true;
}

tabs.forEach(function(b){
  b.addEventListener('click',function(){
    var slug=b.dataset.go;
    activate(slug);
    var mast=doc.querySelector('.proj[data-p="'+slug+'"] .mast');
    if(mast){history.replaceState(null,'','#'+mast.id);}
    window.scrollTo(0,0);
  });
});

/* every in-page link, including the SVG diagrams, has to be able to cross
   into a project pane that is currently off screen */
doc.addEventListener('click',function(e){
  var n=e.target;
  while(n&&n!==doc){
    if(n.nodeType===1&&n.nodeName.toLowerCase()==='a'){
      if(n.hasAttribute('data-all')){e.preventDefault();expandAll(n);return;}
      var h=n.getAttribute('href')||n.getAttribute('xlink:href')||'';
      if(h.charAt(0)==='#'&&h.length>1&&goTo(decodeURIComponent(h.slice(1)),true))
        e.preventDefault();
      return;
    }
    n=n.parentNode;
  }
});
window.addEventListener('hashchange',function(){
  goTo(decodeURIComponent(location.hash.slice(1)),false);
});

function expandAll(a){
  var blk=a.parentNode; while(blk&&!blk.classList.contains('tasks-block'))blk=blk.parentNode;
  if(!blk)return;
  var ds=qa('details.phase',blk), closed=false;
  ds.forEach(function(d){if(!d.open)closed=true;});
  ds.forEach(function(d){d.open=closed;});
  a.textContent=closed?'collapse all':'expand all';
}

/* ---------- search index, built once from the rendered DOM ---------- */
function ownText(el){
  var out=[];
  (function walk(n){
    for(var c=n.firstChild;c;c=c.nextSibling){
      if(c.nodeType===3){out.push(c.nodeValue);}
      else if(c.nodeType===1&&!c.hasAttribute('data-leaf')&&!c.hasAttribute('data-item')
              &&!c.hasAttribute('data-box')){walk(c);}
    }
  })(el);
  return out.join(' ').toLowerCase();
}

var units=qa('[data-unit]').map(function(el){
  var boxes=qa('[data-box]',el).map(function(b){return {el:b,key:ownText(b),vis:false};});
  var items=qa('[data-item]',el).map(function(it){
    var box=null;
    for(var i=0;i<boxes.length;i++){if(boxes[i].el.contains(it)){box=boxes[i];break;}}
    return {el:it,key:ownText(it),box:box,kind:it.className.split(' ')[0],
      leaves:qa('[data-leaf]',it).map(function(lf){
        return {el:lf,key:lf.textContent.toLowerCase(),kind:lf.className.split(' ')[0]};
      })};
  });
  return {el:el,boxes:boxes,items:items,
    key:ownText(el),
    nav:doc.querySelector('.railproj nav a[href="#'+el.id+'"]')};
});

function unmark(){
  qa('mark.hl').forEach(function(m){
    var p=m.parentNode;
    if(!p)return;
    p.replaceChild(doc.createTextNode(m.textContent),m); p.normalize();
  });
}
function mark(root,term){
  var w=doc.createTreeWalker(root,NodeFilter.SHOW_TEXT,{acceptNode:function(n){
    if(!n.nodeValue||n.nodeValue.toLowerCase().indexOf(term)<0)return 2;
    var t=n.parentNode.nodeName;
    return (t==='SCRIPT'||t==='STYLE'||t==='MARK')?2:1;}});
  var found=[],n; while((n=w.nextNode()))found.push(n);
  found.forEach(function(node){
    var s=node.nodeValue, low=s.toLowerCase(), i=0, frag=doc.createDocumentFragment(), j;
    while((j=low.indexOf(term,i))>=0){
      if(j>i)frag.appendChild(doc.createTextNode(s.slice(i,j)));
      var m=doc.createElement('mark'); m.className='hl';
      m.textContent=s.slice(j,j+term.length); frag.appendChild(m); i=j+term.length;
    }
    frag.appendChild(doc.createTextNode(s.slice(i)));
    node.parentNode.replaceChild(frag,node);
  });
}
function show(el,on){ el.classList.toggle('hide',!on); }

/* a collapsed phase or design doc must not hide a hit */
var detailsWere=null;
function forceOpen(on){
  var ds=qa('details');
  if(on){
    if(detailsWere===null)detailsWere=ds.map(function(d){return d.open;});
    ds.forEach(function(d){d.open=true;});
  }else if(detailsWere!==null){
    ds.forEach(function(d,i){d.open=detailsWere[i];});
    detailsWere=null;
  }
}

function clear(){
  body.classList.remove('searching'); hits.textContent='';
  forceOpen(false);
  units.forEach(function(u){
    show(u.el,true); if(u.nav)u.nav.classList.remove('dim');
    u.boxes.forEach(function(b){show(b.el,true);});
    u.items.forEach(function(it){
      show(it.el,true);
      it.leaves.forEach(function(lf){show(lf.el,true);});
    });
  });
  qa('.nores').forEach(function(n){n.classList.remove('on');});
}

function run(){
  var term=input.value.trim().toLowerCase();
  unmark();
  if(!term){clear();return;}
  body.classList.add('searching');
  forceOpen(true);
  var tally={};
  var bump=function(k){tally[k]=(tally[k]||0)+1;};
  var perPane={};
  units.forEach(function(u){
    var uHit=u.key.indexOf(term)>=0, any=uHit;
    u.boxes.forEach(function(b){b.vis=false;});
    u.items.forEach(function(it){
      var bHit=it.box?it.box.key.indexOf(term)>=0:false;
      var iHit=it.key.indexOf(term)>=0, lAny=false;
      it.leaves.forEach(function(lf){
        var lHit=lf.key.indexOf(term)>=0;
        show(lf.el,uHit||bHit||iHit||lHit);
        if(lHit){lAny=true;bump(lf.kind);}
      });
      var keep=uHit||bHit||iHit||lAny;
      show(it.el,keep);
      if(iHit)bump(it.kind);
      if(keep){any=true; if(it.box)it.box.vis=true;}
    });
    u.boxes.forEach(function(b){show(b.el,b.vis||uHit);});
    show(u.el,any);
    if(u.nav)u.nav.classList.toggle('dim',!any);
    if(any){
      bump(u.el.classList.contains('chg')?'chg':'cap');
      mark(u.el,term);
      var p=paneOf(u.el); if(p)perPane[p.dataset.p]=1;
    }
  });
  var label={cap:'cap',chg:'chg',req:'req',scn:'scn',task:'task',doc:'doc',phase:'phase'};
  var parts=[];
  ['cap','chg','req','scn','task','doc'].forEach(function(k){
    if(tally[k])parts.push(tally[k]+' '+label[k]);
  });
  hits.textContent=parts.length?parts.join(' \u00b7 '):'no matches';
  /* a hit in another project should be findable, not silently filtered away */
  if(multi&&!perPane[cur]){
    var other=Object.keys(perPane)[0];
    if(other){
      var name=doc.querySelector('.switch button[data-go="'+other+'"] .pn');
      hits.textContent+=' \u2192 '+(name?name.textContent:other);
    }
  }
  qa('.nores').forEach(function(n){
    var p=paneOf(n); n.classList.toggle('on',!p||!perPane[p.dataset.p]);
  });
}
var t; input.addEventListener('input',function(){clearTimeout(t);t=setTimeout(run,90);});

/* every diagram that can park a highlight registers a release here */
var release=[];
doc.addEventListener('keydown',function(e){
  if(e.key==='/'&&doc.activeElement!==input){e.preventDefault();input.focus();input.select();}
  else if(e.key==='Escape'){
    release.forEach(function(f){f();});
    if(input.value){input.value='';run();}
    input.blur();
  }
});

/* ---------- diagrams: light up a node's incoming and outgoing traces ----------
   Hover previews; on the reference diagram a click parks that preview so you
   can read the page with one capability's wires still lit. */
qa('svg.map, svg.flow, svg.dlv').forEach(function(map){
  var edges=qa('.edge, .fedge',map), nodes=qa('[data-node],[data-cap]',map);
  /* the flow diagram keys edges by node key (cap:<id>), the others by id */
  var keyOf=function(n){return n.dataset.node||n.dataset.cap;};
  var holds=map.classList.contains('map');
  var held=null;
  var setF=function(id){
    map.classList.toggle('held',held!==null);
    nodes.forEach(function(n){n.classList.toggle('held',held!==null&&keyOf(n)===held);});
    if(!id){
      map.classList.remove('act');
      edges.forEach(function(e){e.classList.remove('in','out');});
      nodes.forEach(function(n){n.classList.remove('lit','self');});
      return;
    }
    map.classList.add('act');
    var lit={}; lit[id]=1;
    edges.forEach(function(e){
      var f=e.dataset.from, t2=e.dataset.to;
      var o=f===id||(e.dataset.bi&&t2===id), i=t2===id||(e.dataset.bi&&f===id);
      e.classList.toggle('out',!!o&&!i); e.classList.toggle('in',!!i);
      if(o||i){lit[f]=1;lit[t2]=1;}
    });
    nodes.forEach(function(n){
      n.classList.toggle('lit',!!lit[keyOf(n)]);
      n.classList.toggle('self',keyOf(n)===id);
    });
  };
  if(holds)release.push(function(){ if(held!==null){held=null;setF(null);} });
  nodes.forEach(function(n){
    n.addEventListener('mouseenter',function(){setF(keyOf(n));});
    n.addEventListener('focus',function(){setF(keyOf(n));});
    n.addEventListener('mouseleave',function(){setF(held);});
    n.addEventListener('blur',function(){setF(held);});
    if(!holds)return;
    n.addEventListener('click',function(e){
      /* keyboard Enter (detail 0) and a double click both follow the link */
      if(e.detail===0||e.detail>1){held=null;setF(null);return;}
      e.preventDefault(); e.stopPropagation();
      var k=keyOf(n);
      held=held===k?null:k;
      setF(k);
    });
  });
});

/* ---------- keep the hash pointing at what you are reading ---------- */
var quiet=0;
doc.addEventListener('click',function(){quiet=Date.now()+900;},true);
if('IntersectionObserver' in window){
  var vis={};
  var io=new IntersectionObserver(function(es){
    es.forEach(function(e){
      if(e.isIntersecting)vis[e.target.id]=e.boundingClientRect.top;
      else delete vis[e.target.id];
    });
    var best=null,bt=1e9;
    Object.keys(vis).forEach(function(k){if(vis[k]<bt){bt=vis[k];best=k;}});
    qa('.railproj nav a').forEach(function(a){
      a.classList.toggle('on',a.getAttribute('href')==='#'+best);
    });
    if(best&&Date.now()>quiet&&location.hash!=='#'+best)
      history.replaceState(null,'','#'+best);
  },{rootMargin:'-8% 0px -70% 0px',threshold:0});
  qa('.cap, .chg, .proj > section[id], .proj > header[id]').forEach(function(s){io.observe(s);});
}

/* ---------- print: nothing collapsed on paper ---------- */
if(window.matchMedia){
  var mq=window.matchMedia('print');
  var onprint=function(m){ if(m.matches)forceOpen(true); else if(!input.value)forceOpen(false); };
  if(mq.addEventListener)mq.addEventListener('change',onprint);
  else if(mq.addListener)mq.addListener(onprint);
}
window.addEventListener('beforeprint',function(){forceOpen(true);});
window.addEventListener('afterprint',function(){if(!input.value)forceOpen(false);});

/* ---------- theme ---------- */
var tb=doc.getElementById('theme');
if(tb)tb.addEventListener('click',function(){
  var el=doc.documentElement, curT=el.getAttribute('data-theme');
  if(!curT)curT=matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light';
  el.setAttribute('data-theme',curT==='dark'?'light':'dark');
});

/* ---------- boot ---------- */
/* on a phone the rail is a banner across the top: do not spend the first
   screen on a table of contents. Wide again and it must come back, so follow
   the query rather than sampling it once at load. */
var railD=doc.querySelector('.rail>details');
if(railD&&window.matchMedia){
  var narrow=matchMedia('(max-width:1000px)');
  var syncRail=function(m){ railD.open=!m.matches; };
  syncRail(narrow);
  if(narrow.addEventListener)narrow.addEventListener('change',syncRail);
  else if(narrow.addListener)narrow.addListener(syncRail);
}
if(multi){
  cur=null;
  activate(panes[0].dataset.p);
}
if(location.hash.length>1)goTo(decodeURIComponent(location.hash.slice(1)),false);
})();
"""
