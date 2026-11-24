"""Passage backdrop: tech-poster arrows on a light field, drawn with QPainter.

Everything is generated in code (no image assets). The composition runs along one
45-degree axis: arrows point and travel from bottom-left to top-right. A static
layer (bands, ghost shapes, sphere, halftone, lettering) is cached per window size;
a set of arrow sprites is drawn on top. While idle the sprites sit still at their
poster positions; while translating they stream up-right along their lanes.
Front-end only: nothing here touches translation, files or task state.
"""
import math, random
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import (QColor, QPainter, QPen, QBrush, QPixmap, QPolygonF, QPainterPath, QTransform,
                           QLinearGradient, QRadialGradient, QFont)

PAPER = QColor('#f3f3f6')
INDIGO = QColor('#2f33c6')
INDIGO_DEEP = QColor('#1f2396')
LIME = QColor('#c4c82a')
LIME_DARK = QColor('#9a9d17')
INK = QColor('#121216')
GRAY = QColor('#9e9ea8')
GRAY_LIGHT = QColor('#c6c6ce')
GHOST = QColor('#e2e2e8')
LAVENDER = QColor('#7c78e6')
FONT = 'Noto Sans SC'

def rgba(c, a):
    c = QColor(c); c.setAlphaF(max(0., min(1., a))); return c

# ------------------------------------------------------------------ geometry

class Frame:
    """Local frame: u runs up-right along the arrows, v runs down-right across lanes."""
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.s = (w + h) / 2300.                       # scale relative to a 1400 x 900 window
        self.W = 118 * self.s                          # main band width
        self.C = QPointF(.10 * w + .5 * h, .5 * h)     # main line: left panel's foot -> top banner
        self.d = (math.sqrt(.5), -math.sqrt(.5)); self.n = (math.sqrt(.5), math.sqrt(.5))
        us = [self.uv(x, y)[0] for x, y in ((0, 0), (w, 0), (0, h), (w, h))]
        vs = [self.uv(x, y)[1] for x, y in ((0, 0), (w, 0), (0, h), (w, h))]
        self.umin, self.umax, self.vmin, self.vmax = min(us), max(us), min(vs), max(vs)
    def uv(self, x, y):
        dx, dy = x - self.C.x(), y - self.C.y()
        return dx * self.d[0] + dy * self.d[1], dx * self.n[0] + dy * self.n[1]
    def xy(self, u, v):
        return QPointF(self.C.x() + u * self.d[0] + v * self.n[0], self.C.y() + u * self.d[1] + v * self.n[1])
    def at(self, fx, fy):
        return self.uv(fx * self.w, fy * self.h)
    def transform(self):
        t = QTransform(); t.translate(self.C.x(), self.C.y()); t.rotate(-45); return t

# ------------------------------------------------------------------ shapes (local frame, pointing +x, tip at x=0)

def block_arrow(length, width, head=None):
    """Shaft plus a square-cut 90-degree head, the poster's main arrow form."""
    head = head or width * 2.1; tip = head / 2
    return QPolygonF([QPointF(-length, -width / 2), QPointF(-tip, -width / 2), QPointF(-tip, -head / 2), QPointF(0, 0),
                      QPointF(-tip, head / 2), QPointF(-tip, width / 2), QPointF(-length, width / 2)])

def chevron(size, thick):
    a = size / 2
    return QPolygonF([QPointF(-a - thick, -a), QPointF(-a, -a), QPointF(0, 0), QPointF(-a, a), QPointF(-a - thick, a), QPointF(-thick, 0)])

def corner_arrow(size, thick):
    """Small icon: right-angle corner plus a diagonal stem (the ↗ glyph), in screen axes."""
    path = QPainterPath(); s, t = size, thick
    path.addPolygon(QPolygonF([QPointF(0, 0), QPointF(s, 0), QPointF(s, t), QPointF(t, t), QPointF(t, s), QPointF(0, s)]))
    stem = QPolygonF([QPointF(t * .6, t * 1.4), QPointF(t * 1.4, t * .6), QPointF(s * .95, s * .95 - t * .3), QPointF(s * .95 - t * .3, s * .95)])
    path.addPolygon(stem); path.setFillRule(Qt.FillRule.WindingFill)
    return path

def comb(length, bar, teeth, rng):
    """Pixel 'barcode' arrow: a thin bar with square teeth of varying lengths."""
    path = QPainterPath(); path.addRect(QRectF(-length, -bar / 2, length, bar))
    step = length / teeth
    for i in range(teeth):
        if rng.random() < .25: continue
        span = bar * rng.choice((1.5, 2.5, 3.5, 5))
        side = rng.choice((-1, 1, 0))
        top = -span / 2 if side == 0 else (-span if side < 0 else 0)
        path.addRect(QRectF(-length + i * step, top, step * .62, span))
    path.addPolygon(chevron(bar * 4, bar * 1.2).translated(bar * 2.2, 0)); path.setFillRule(Qt.FillRule.WindingFill)
    return path

def stripes(p, rect, color, gap, width=1.):
    """Vertical pinstripes in screen space (the poster's lime texture), clipped by the caller."""
    p.save(); p.resetTransform(); p.setPen(QPen(color, width))
    x = rect.left() - rect.left() % gap
    while x < rect.right():
        p.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom())); x += gap
    p.restore()

def halftone(p, rect, color, gap, radius_at):
    """Dot screen in screen space; radius_at(x, y) -> radius (0 skips)."""
    p.save(); p.resetTransform(); p.setPen(Qt.PenStyle.NoPen); p.setBrush(color)
    y = rect.top(); row = 0
    while y < rect.bottom():
        x = rect.left() + (gap / 2 if row % 2 else 0)
        while x < rect.right():
            r = radius_at(x, y)
            if r > .35: p.drawEllipse(QPointF(x, y), r, r)
            x += gap
        y += gap * .87; row += 1
    p.restore()

# ------------------------------------------------------------------ static layer

def _fill_fading(p, poly, color, a_tip, a_tail, length):
    g = QLinearGradient(QPointF(0, 0), QPointF(-length, 0))
    g.setColorAt(0, rgba(color, a_tip)); g.setColorAt(1, rgba(color, a_tail))
    p.setPen(Qt.PenStyle.NoPen); p.setBrush(QBrush(g)); p.drawPolygon(poly)

def _local(p, f, u, v):
    """Move the painter to (u, v) in the local frame."""
    p.setTransform(f.transform()); p.translate(u, v)

def render_static(w, h, dpr=1.):
    f = Frame(w, h); W = f.W; s = f.s
    pix = QPixmap(max(1, int(w * dpr)), max(1, int(h * dpr))); pix.setDevicePixelRatio(dpr); pix.fill(PAPER)
    p = QPainter(pix); p.setRenderHint(QPainter.RenderHint.Antialiasing)
    rng = random.Random(11)
    # Paper: faint light falling from the top-right.
    g = QLinearGradient(QPointF(w, 0), QPointF(0, h)); g.setColorAt(0, QColor('#fbfbfd')); g.setColorAt(1, QColor('#ececf1'))
    p.fillRect(QRectF(0, 0, w, h), QBrush(g))

    # Ghost forms: huge pale arrows and thick rings, the poster's background type.
    p.setPen(QPen(rgba(GHOST, .9), 62 * s, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap))
    p.setBrush(Qt.BrushStyle.NoBrush)
    for fx, fy, r, start, span in ((.10, .22, .20, 40, 250), (.92, .86, .22, 200, 230), (.70, .05, .14, 160, 200)):
        rr = r * h; p.drawArc(QRectF(fx * w - rr, fy * h - rr, 2 * rr, 2 * rr), start * 16, span * 16)
    for fx, fy, width, length in ((.30, .30, 1.9, 9), (.95, .70, 1.5, 8), (.20, .98, 1.3, 7), (.62, .02, 1.1, 6)):
        u, v = f.at(fx, fy); _local(p, f, u, v)
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(rgba(GHOST, .75)); p.drawPolygon(block_arrow(length * W, width * W))
    p.resetTransform()
    # Thin frame lines, as on a layout sheet.
    p.setPen(QPen(QColor(20, 20, 30, 26), 1)); p.setBrush(Qt.BrushStyle.NoBrush)
    for fx in (.015, .30, .58, .87):
        p.drawRoundedRect(QRectF(fx * w, .02 * h, .12 * w, .96 * h), 10 * s, 10 * s)

    # Sphere behind the band.
    cx, cy, r = .60 * w, .50 * h, .29 * h
    rg = QRadialGradient(QPointF(cx - r * .35, cy - r * .4), r * 1.35)
    rg.setColorAt(0, QColor(250, 250, 252, 235)); rg.setColorAt(.55, QColor(170, 170, 180, 220)); rg.setColorAt(1, QColor(96, 96, 108, 225))
    p.setPen(Qt.PenStyle.NoPen); p.setBrush(QBrush(rg)); p.drawEllipse(QPointF(cx, cy), r, r)
    halftone(p, QRectF(cx - r, cy - r, 2 * r, 2 * r), QColor(255, 255, 255, 120), 7 * s,
             lambda x, y: (1.9 * s * max(0, 1 - math.hypot(x - cx, y - cy) / r) * max(0, (x - cx + y - cy) / r)) if math.hypot(x - cx, y - cy) < r else 0)

    # Lavender halftone arrows on the lower-right lanes.
    for fx, fy, width, length, a in ((.80, .52, 1.15, 7.5, .85), (.97, .80, .95, 6.0, .75), (.64, .86, .75, 5.0, .6), (.99, .30, .6, 4.5, .55)):
        u, v = f.at(fx, fy); _local(p, f, u, v)
        _fill_fading(p, block_arrow(length * W, width * W), LAVENDER, a, 0, length * W)
    p.resetTransform()
    corner = QPointF(w, h)
    halftone(p, QRectF(.55 * w, .45 * h, .45 * w, .55 * h), rgba(INDIGO, .55), 8 * s,
             lambda x, y: 3.0 * s * max(0, 1 - math.hypot(x - w, y - h) / (.62 * h)))

    # Gray and black arrows on the upper-left lanes.
    for fx, fy, width, length, color, a in ((.30, .36, .55, 6, GRAY_LIGHT, .95), (.67, .17, .7, 7, GRAY, .95), (.08, .60, .5, 5, GRAY, .7)):
        u, v = f.at(fx, fy); _local(p, f, u, v); _fill_fading(p, block_arrow(length * W, width * W), color, a, .05, length * W)
    for fx, fy, width, length in ((.47, .10, .78, 4.2), (.12, .97, .7, 3.6), (.985, .33, .7, 3.8)):
        u, v = f.at(fx, fy); _local(p, f, u, v)
        poly = block_arrow(length * W, width * W)
        _fill_fading(p, poly, INK, 1, .0, length * W)
        p.save(); p.setClipPath(_path(poly)); halftone(p, QRectF(0, 0, w, h), QColor(255, 255, 255, 40), 5 * s, lambda x, y: .9 * s); p.restore()

    # Main band: indigo shaft and head, lime pinstripe band beside it.
    hu = head_u(f)  # the head sits on the main line inside the top banner
    _local(p, f, hu, 0)
    band = block_arrow(hu - f.umin + 40, W, W * 2.25)
    g = QLinearGradient(QPointF(0, 0), QPointF(-(hu - f.umin), 0)); g.setColorAt(0, INDIGO); g.setColorAt(1, INDIGO_DEEP)
    p.setPen(Qt.PenStyle.NoPen); p.setBrush(QBrush(g)); p.drawPolygon(band)
    lime = QPolygonF([QPointF(-W * 1.45, -W * .5 - W * .66), QPointF(-(hu - f.umin) - 40, -W * .5 - W * .66),
                      QPointF(-(hu - f.umin) - 40, -W * .5 - W * .06), QPointF(-W * 1.45, -W * .5 - W * .06)])
    lg = QLinearGradient(QPointF(0, 0), QPointF(-(hu - f.umin) * .85, 0)); lg.setColorAt(0, LIME); lg.setColorAt(.75, rgba(LIME, .85)); lg.setColorAt(1, rgba(LIME, 0))
    p.setBrush(QBrush(lg)); p.drawPolygon(lime)
    p.save(); p.setClipPath(_path(lime)); stripes(p, QRectF(0, 0, w, h), rgba(LIME_DARK, .55), 5 * s, 1.2 * s); p.restore()
    # Notched lime blocks and indigo squares near the head.
    for k in range(5):
        x = -W * (1.7 + k * .9 + rng.random() * .3); sz = W * rng.choice((.28, .36, .44))
        p.setBrush(LIME if k % 2 else INDIGO); p.drawRect(QRectF(x, -W * .5 - W * .66 - sz * .8, sz, sz))
        p.setBrush(Qt.BrushStyle.NoBrush); p.setPen(QPen(INDIGO, 1.4 * s)); p.drawRect(QRectF(x - sz * .4, -W * .5 - W * .66 - sz * 1.9, sz * .7, sz * .7)); p.setPen(Qt.PenStyle.NoPen)
    # Tri-colour hairline and tiny system text on the band.
    for i, c in enumerate(('#e0322e', '#e8c21c', '#2fb2c8')):
        p.setBrush(QColor(c)); p.drawRect(QRectF(-W * 9.2 - i * W * .5, W * .5 + 2 * s, W * .5, 2.2 * s))
    font = QFont(FONT); font.setPixelSize(max(7, round(8 * s))); font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.5 * s); p.setFont(font)
    p.setPen(QColor(255, 255, 255, 215))
    for i, line in enumerate(('PASSAGE // LOCAL MANGA TRANSLATION SYSTEM', 'JA · EN · KO  →  ZH-HANS        OFFLINE', 'DETECT / OCR / TRANSLATE / CLEAN / LETTER')):
        p.drawText(QPointF(-W * 7.6, -W * .22 + i * 11 * s), line)

    # Big lettering along the band: white, shading into ink toward the bottom-left.
    font = QFont(FONT); font.setPixelSize(round(W * 1.55)); font.setWeight(QFont.Weight.Black); font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, -W * .06)
    path = QPainterPath(); path.addText(QPointF(0, 0), font, 'PASSAGE'); box = path.boundingRect()
    path.translate(-W * 4.0 - box.right(), W * .5 - box.bottom() + W * .08)
    tg = QLinearGradient(path.boundingRect().topRight(), path.boundingRect().bottomLeft())
    tg.setColorAt(0, QColor(255, 255, 255, 245)); tg.setColorAt(.6, QColor(236, 236, 242, 235)); tg.setColorAt(1, QColor(40, 40, 48, 220))
    p.setPen(QPen(QColor(255, 255, 255, 90), 1)); p.setBrush(QBrush(tg)); p.drawPath(path)
    p.resetTransform()

    # Pixel combs, corner icons, diamonds and plus-grids.
    for fx, fy, length in ((.20, .50, 3.6), (.55, .93, 4.4), (.88, .44, 3.2), (.40, .06, 2.6)):
        u, v = f.at(fx, fy); _local(p, f, u, v); p.setPen(Qt.PenStyle.NoPen); p.setBrush(rgba(GRAY, .75))
        p.drawPath(comb(length * W, W * .12, 11, rng))
    p.resetTransform()
    for fx, fy, size, color in ((.14, .70, 26, GRAY), (.42, .64, 22, GRAY), (.71, .29, 30, GRAY), (.90, .13, 26, GRAY), (.52, .31, 64, LAVENDER), (.76, .70, 40, rgba(LAVENDER, .8))):
        sz = size * s; p.save(); p.translate(fx * w, fy * h); p.rotate(90)  # corner pointing up-right
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(color); p.drawPath(corner_arrow(sz, sz * .32)); p.restore()
    for _ in range(14):
        u = rng.uniform(f.umin * .6, f.umax * .8); v = rng.uniform(-3.2 * W, 3.2 * W); c = f.xy(u, v); sz = rng.choice((8, 10, 13, 16)) * s
        p.save(); p.translate(c); p.rotate(45); p.setBrush(rng.choice((INDIGO, LAVENDER, rgba(INDIGO, .7), INK))); p.drawRect(QRectF(-sz / 2, -sz / 2, sz, sz)); p.restore()
    p.setPen(QPen(QColor(255, 255, 255, 230), 1.6 * s))
    for fx, fy in ((.66, .60), (.86, .87)):
        for i in range(3):
            for j in range(3):
                x, y = fx * w + i * 9 * s, fy * h + j * 9 * s
                p.drawLine(QPointF(x - 2.5 * s, y), QPointF(x + 2.5 * s, y)); p.drawLine(QPointF(x, y - 2.5 * s), QPointF(x, y + 2.5 * s))
    p.end()
    return pix, f

def head_u(f):
    return (.5 - .10) * f.h * math.sqrt(2)

def _path(poly):
    path = QPainterPath(); path.addPolygon(poly); path.closeSubpath(); return path

# ------------------------------------------------------------------ moving arrows

KINDS = ('block', 'block', 'hollow', 'chevron', 'chevron', 'stripe', 'comb', 'icon', 'triple', 'lavender')

def _sprite(kind, size, color, s, dpr, rng):
    """Pre-rotated pixmap; returns (pixmap, offset of the tip from the pixmap's top-left)."""
    box = int(size * 3.2) + 8
    pix = QPixmap(int(box * dpr), int(box * dpr)); pix.setDevicePixelRatio(dpr); pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix); p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.translate(box * .78, box * .22); p.rotate(-45)
    if kind == 'block':
        p.setPen(Qt.PenStyle.NoPen); _fill_fading(p, block_arrow(size * 2.6, size * .42, size * .95), color, 1, .0, size * 2.6)
    elif kind == 'lavender':
        p.setPen(Qt.PenStyle.NoPen); _fill_fading(p, block_arrow(size * 2.8, size * .5, size * 1.05), LAVENDER, .8, 0, size * 2.8)
    elif kind == 'hollow':
        p.setPen(QPen(color, max(1.2, 1.6 * s))); p.setBrush(QColor(255, 255, 255, 120)); p.drawPolygon(block_arrow(size * 2.2, size * .38, size * .9))
    elif kind == 'chevron':
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(color); p.drawPolygon(chevron(size, size * .26))
    elif kind == 'triple':
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(color)
        for k in range(3): p.drawPolygon(chevron(size * .5, size * .12).translated(-k * size * .32, 0))
    elif kind == 'stripe':
        poly = block_arrow(size * 2.4, size * .5, size * 1.0)
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(LIME); p.drawPolygon(poly)
        p.save(); p.setClipPath(_path(poly)); stripes(p, QRectF(0, 0, box, box), rgba(LIME_DARK, .7), 4 * s, 1.1 * s); p.restore()
    elif kind == 'comb':
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(rgba(color, .85)); p.drawPath(comb(size * 2.4, size * .1, 9, rng))
    elif kind == 'icon':
        p.resetTransform(); p.translate(box * .5, box * .5); p.rotate(90)
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(color); p.drawPath(corner_arrow(size * .7, size * .22))
    p.end()
    return pix

def make_sprites(f, dpr, seed=5):
    rng = random.Random(seed); W, s = f.W, f.s; out = []
    palette = {'block': (INK, GRAY, INDIGO, GRAY_LIGHT), 'hollow': (GRAY, INDIGO, INK), 'chevron': (INDIGO, LIME, GRAY, INK),
               'triple': (INDIGO, GRAY, INK, LIME), 'comb': (GRAY, INK), 'icon': (GRAY, INDIGO, LAVENDER), 'stripe': (LIME,), 'lavender': (LAVENDER,)}
    for i in range(38):
        kind = KINDS[i % len(KINDS)]
        size = W * rng.choice((.35, .5, .7, .9, 1.2)) * (.6 if kind in ('icon', 'triple') else 1)
        color = rng.choice(palette[kind])
        pix = _sprite(kind, size, color, s, dpr, rng)
        v = rng.uniform(f.vmin * .85, f.vmax * .85)
        out.append({'pix': pix, 'v': v, 'phase': rng.random(), 'speed': (240 - size / W * 110) * s * rng.uniform(.7, 1.3), 'size': size})
    return out

def draw_sprites(p, f, sprites, t):
    span = f.umax - f.umin + 4 * f.W
    for sp in sprites:
        u = f.umin - 2 * f.W + ((sp['phase'] * span + sp['speed'] * t) % span)
        c = f.xy(u, sp['v']); box = sp['pix'].width() / sp['pix'].devicePixelRatio()
        p.drawPixmap(QPointF(c.x() - box * .78 + box * .0, c.y() - box * .22), sp['pix'])

def draw_live(p, f, t, busy):
    """Status tag on the band; while busy a glint runs up the band and the tag blinks."""
    W, s = f.W, f.s
    hu = head_u(f)
    p.save(); p.setTransform(f.transform())
    if busy:
        head = f.umin + ((t * 520 * s) % (hu - f.umin + 6 * W))
        g = QLinearGradient(QPointF(head - 3 * W, 0), QPointF(head, 0))
        g.setColorAt(0, QColor(255, 255, 255, 0)); g.setColorAt(1, QColor(255, 255, 255, 120))
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(QBrush(g)); p.drawRect(QRectF(head - 3 * W, -W * .5, 3 * W, W))
    p.restore()
    # Status tag in the top banner, just left of the band's arrowhead (screen axes).
    tip = f.xy(hu, 0); font = QFont(FONT); font.setPixelSize(max(9, round(11 * s))); font.setBold(True)
    font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 3 * s); p.setFont(font)
    text = '[  翻 译 中  ]' if busy else '[  STANDBY  ]'; tw = p.fontMetrics().horizontalAdvance(text)
    box = QRectF(tip.x() - W * 1.9 - tw - 24 * s, max(10 * s, tip.y() - 52 * s), tw + 24 * s, 24 * s)
    on = (not busy) or (int(t * 2.2) % 2 == 0)
    p.setPen(Qt.PenStyle.NoPen); p.setBrush(INDIGO if busy else INK); p.drawRect(box)
    p.setBrush(LIME); p.drawRect(QRectF(box.right(), box.top(), 5 * s, box.height()))
    p.setPen(QColor(255, 255, 255, 240 if on else 60)); p.drawText(box, Qt.AlignmentFlag.AlignCenter, text)

def wordmark(height, dpr=2.):
    """PASSAGE in heavy black type over a spaced subtitle between hairlines."""
    w = int(height * 4.3); pix = QPixmap(int(w * dpr), int(height * dpr)); pix.setDevicePixelRatio(dpr); pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix); p.setRenderHint(QPainter.RenderHint.Antialiasing)
    font = QFont(FONT); font.setPixelSize(round(height * .66)); font.setWeight(QFont.Weight.Black); font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, -height * .02)
    path = QPainterPath(); path.addText(QPointF(0, 0), font, 'PASSAGE'); b = path.boundingRect()
    path.translate(-b.left(), height * .62 - b.bottom()); p.setPen(Qt.PenStyle.NoPen); p.setBrush(INK); p.drawPath(path)
    right = path.boundingRect().right()
    p.setBrush(LIME); p.drawRect(QRectF(right + height * .08, height * .62 - height * .2, height * .12, height * .2))
    small = QFont(FONT); small.setPixelSize(max(8, round(height * .17))); small.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, height * .14); p.setFont(small)
    text = '漫画翻译'; tw = p.fontMetrics().horizontalAdvance(text); y = height * .9
    mid = right / 2; p.setPen(INK); p.drawText(QPointF(mid - tw / 2, y), text)
    p.setPen(QPen(INK, max(1., height * .025)))
    p.drawLine(QPointF(0, y - height * .06), QPointF(mid - tw / 2 - height * .14, y - height * .06))
    p.drawLine(QPointF(mid + tw / 2 + height * .04, y - height * .06), QPointF(right, y - height * .06))
    p.end(); return pix
