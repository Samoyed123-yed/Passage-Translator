"""Passage desktop look: tech-poster arrows on a light field, ink text, indigo and lime.

Front-end only. Widgets here never touch translation, files or task state; app.py
wires them to the unchanged AutoBatch workflow. The arrow art is drawn by
arrow_art.py: still while idle, streaming up-right while a book is translated.
"""
import time
from pathlib import Path
from PySide6.QtCore import Qt, QSize, QRectF, QPointF, QRect, Signal, QTimer
from PySide6.QtGui import (QColor, QPainter, QPen, QPixmap, QIcon, QPolygonF, QFont, QFontDatabase, QRegion)
from PySide6.QtWidgets import (QWidget, QToolButton, QPushButton, QHBoxLayout, QButtonGroup, QListWidget,
                               QListView, QStyledItemDelegate, QStyle, QListWidgetItem, QAbstractItemView)
import arrow_art

ROOT = Path(__file__).resolve().parent
BG, PANEL, STAGE, LINE = '#f3f3f6', 'rgba(255,255,255,0.58)', 'rgba(255,255,255,0.72)', 'rgba(18,18,22,0.16)'
TEXT, MUTED, FAINT = '#121216', '#5f5f6b', '#b4b4be'
VIOLET, BLUE, LIME = '#2f33c6', '#2f33c6', '#c4c82a'
UI_FONT = 'Noto Sans SC'

STYLE = f'''
QWidget {{color:{TEXT};font-family:"{UI_FONT}","Microsoft YaHei UI";font-size:13px;background:transparent;}}
QMainWindow, QDialog {{background:{BG};}}
QFrame#panel {{background:{PANEL};border:1px solid {LINE};}}
QFrame#stage {{background:{STAGE};border:1px solid {LINE};}}
QLabel#heading {{font-size:11px;font-weight:700;color:{TEXT};letter-spacing:3px;border-left:3px solid {VIOLET};padding-left:7px;}}
QLabel#muted {{color:{MUTED};font-size:12px;}}
QLabel#tag {{color:{MUTED};font-size:10px;letter-spacing:4px;padding:0 0 4px 0;}}
QFrame#plate {{background:rgba(255,255,255,0.55);border:0;}}
QLabel#subtitle {{color:{TEXT};font-size:10px;letter-spacing:6px;}}
QLabel#chip {{color:#ffffff;font-size:11px;letter-spacing:2px;background:{TEXT};border-left:4px solid {LIME};padding:5px 12px;}}
QLabel#count {{color:{MUTED};font-size:12px;}}
QLabel#status {{color:{MUTED};font-size:12px;padding:2px 0;}}
QLineEdit, QPlainTextEdit, QComboBox {{background:rgba(255,255,255,0.88);border:1px solid {LINE};padding:8px;
    selection-background-color:{VIOLET};selection-color:white;}}
QLineEdit:focus, QPlainTextEdit:focus {{border:1px solid {VIOLET};}}
QPushButton {{background:rgba(255,255,255,0.78);border:1px solid rgba(18,18,22,0.30);padding:10px 12px;color:{TEXT};min-height:18px;}}
QPushButton:hover {{border-color:{VIOLET};color:{VIOLET};background:rgba(255,255,255,0.96);}}
QPushButton:disabled {{color:{FAINT};border-color:rgba(18,18,22,0.10);}}
QPushButton#primary {{background:{VIOLET};color:#ffffff;border:0;border-right:10px solid {LIME};font-weight:700;font-size:15px;
    padding:15px;letter-spacing:6px;min-height:20px;}}
QPushButton#primary:hover {{background:#2428a8;color:#ffffff;}}
QPushButton#primary:disabled {{background:#d6d6de;color:#9a9aa6;border-right:10px solid #c8c8d0;}}
QPushButton#stop {{background:{TEXT};color:#ffffff;border:0;border-right:10px solid {LIME};font-weight:700;padding:15px;letter-spacing:6px;}}
QPushButton#stop:hover {{background:#2a2a33;color:#ffffff;}}
QPushButton#seg {{border:1px solid rgba(18,18,22,0.20);padding:8px 0;color:{MUTED};background:rgba(255,255,255,0.72);}}
QPushButton#seg:checked {{background:{TEXT};color:#ffffff;border-color:{TEXT};font-weight:700;border-bottom:3px solid {LIME};}}
QPushButton#seg:disabled {{color:{FAINT};}}
QToolButton {{background:transparent;border:1px solid transparent;padding:6px;}}
QToolButton:hover {{border:1px solid {LINE};background:rgba(255,255,255,0.9);}}
QToolButton:checked {{border:1px solid {VIOLET};background:rgba(47,51,198,0.08);}}
QGraphicsView {{background:rgba(255,255,255,0.55);border:1px solid {LINE};}}
QListWidget {{background:transparent;border:0;outline:0;}}
QSplitter::handle {{background:transparent;width:10px;}}
QTableWidget {{background:#ffffff;gridline-color:{LINE};selection-background-color:#e4e5fb;selection-color:{TEXT};}}
QHeaderView::section {{background:#f1f1f5;color:{MUTED};padding:8px;border:0;border-bottom:1px solid {LINE};}}
QComboBox QAbstractItemView {{background:#ffffff;selection-background-color:#e4e5fb;selection-color:{TEXT};}}
QScrollBar:vertical {{background:transparent;width:8px;}}
QScrollBar::handle:vertical {{background:rgba(18,18,22,0.25);min-height:30px;}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{height:0;}}
QScrollBar:horizontal {{background:transparent;height:8px;}}
QScrollBar::handle:horizontal {{background:rgba(18,18,22,0.25);min-width:30px;}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{width:0;}}
QToolTip {{background:#ffffff;color:{TEXT};border:1px solid {LINE};padding:5px;}}
QMessageBox QLabel {{min-width:260px;}}
QDialogButtonBox QPushButton {{min-width:80px;}}
'''

def load_fonts():
    for name in ('NotoSansSC-Regular.otf', 'NotoSansSC-Bold.otf', 'NotoSansSC-Black.otf'):
        path = ROOT / 'fonts' / name
        if path.is_file(): QFontDatabase.addApplicationFont(str(path))

def wordmark(height=46):
    return arrow_art.wordmark(height, 2.)

# ---------------------------------------------------------------- arrow backdrop

class ArrowBackdrop(QWidget):
    """Arrow poster behind the panels. Still while idle; while translating the arrows
    stream from bottom-left to top-right and the band shows a blinking status tag."""
    def __init__(self, header=132):
        super().__init__(); self.header = header; self.static = None; self.frame = None; self.sprites = []
        self.busy = False; self.t = 0.; self.t0 = time.monotonic(); self.skip = []
        self.debounce = QTimer(self); self.debounce.setSingleShot(True); self.debounce.setInterval(120); self.debounce.timeout.connect(self.render_art)
        self.tick = QTimer(self); self.tick.setInterval(33); self.tick.timeout.connect(self._animate)
    def set_panels(self, left, right, skip=()):
        # Image views sit under a heavy white veil; they are not repainted every frame.
        self.skip = list(skip)
    def resizeEvent(self, event):
        super().resizeEvent(event); self.debounce.start()
    def showEvent(self, event):
        super().showEvent(event); self.debounce.start()
    def render_art(self):
        if self.width() < 50: return
        dpr = self.devicePixelRatioF()
        self.static, self.frame = arrow_art.render_static(self.width(), self.height(), dpr)
        self.sprites = arrow_art.make_sprites(self.frame, dpr); self.update()
    def set_animating(self, on):
        if on and not self.busy: self.busy = True; self.t0 = time.monotonic() - self.t; self.tick.start()
        elif not on and self.busy: self.busy = False; self.tick.stop(); self.update()
    def _animate(self):
        self.t = time.monotonic() - self.t0
        region = QRegion(self.rect())
        for widget in self.skip:
            if widget.isVisible(): region -= QRegion(QRect(widget.mapTo(self, widget.rect().topLeft()), widget.size()))
        self.update(region)
    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        if self.static is None: p.fillRect(self.rect(), QColor(BG)); return
        p.drawPixmap(0, 0, self.static)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        arrow_art.draw_sprites(p, self.frame, self.sprites, self.t)
        arrow_art.draw_live(p, self.frame, self.t, self.busy)

# Kept for older imports.
PosterBackdrop = CollageBackdrop = ArrowBackdrop

# ---------------------------------------------------------------- icons and controls

def _icon(draw):
    pix = QPixmap(64, 64); pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix); p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(QPen(QColor(TEXT), 4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.SquareCap, Qt.PenJoinStyle.MiterJoin))
    p.setBrush(Qt.BrushStyle.NoBrush); draw(p); p.end()
    icon = QIcon(pix)
    dim = QPixmap(pix.size()); dim.fill(Qt.GlobalColor.transparent)
    q = QPainter(dim); q.setOpacity(.25); q.drawPixmap(0, 0, pix); q.end()
    icon.addPixmap(dim, QIcon.Mode.Disabled); return icon

def glyph(name):
    def book(p):
        p.drawPolygon(QPolygonF([QPointF(8, 14), QPointF(31, 18), QPointF(31, 54), QPointF(8, 50)]))
        p.drawPolygon(QPolygonF([QPointF(56, 14), QPointF(33, 18), QPointF(33, 54), QPointF(56, 50)]))
    def clock(p):
        p.drawEllipse(QPointF(32, 32), 22, 22); p.drawLine(32, 32, 32, 18); p.drawLine(32, 32, 42, 38)
    def log(p):
        p.drawLine(10, 20, 20, 28); p.drawLine(20, 28, 10, 36); p.drawLine(26, 38, 52, 38); p.drawLine(26, 48, 46, 48)
    def fit(p):
        for x, y, dx, dy in ((10, 10, 1, 1), (54, 10, -1, 1), (10, 54, 1, -1), (54, 54, -1, -1)):
            p.drawLine(x, y, x + 12 * dx, y); p.drawLine(x, y, x, y + 12 * dy)
    def pen(p):
        p.drawPolygon(QPolygonF([QPointF(14, 50), QPointF(16, 40), QPointF(42, 14), QPointF(50, 22), QPointF(24, 48)]))
        p.drawLine(10, 56, 54, 56)
    def redo(p):
        p.drawArc(QRectF(12, 12, 40, 40), 30 * 16, 280 * 16)
        p.drawPolygon(QPolygonF([QPointF(50, 8), QPointF(52, 24), QPointF(37, 21)]))
    return _icon({'book': book, 'clock': clock, 'log': log, 'fit': fit, 'pen': pen, 'redo': redo}[name])

def icon_button(name, tip, checkable=False):
    b = QToolButton(); b.setIcon(glyph(name)); b.setIconSize(QSize(20, 20)); b.setToolTip(tip)
    b.setCheckable(checkable); b.setFixedSize(38, 38); b.setCursor(Qt.CursorShape.PointingHandCursor); return b

class Segmented(QWidget):
    """日文 / 英文 / 韩文 switch with the QComboBox calls app.py already uses."""
    currentIndexChanged = Signal(int)
    def __init__(self, items):
        super().__init__(); row = QHBoxLayout(self); row.setContentsMargins(0, 0, 0, 0); row.setSpacing(0)
        self.group = QButtonGroup(self); self.group.setExclusive(True); self.values = []
        for i, (label, value) in enumerate(items):
            b = QPushButton(label); b.setObjectName('seg'); b.setCheckable(True); b.setCursor(Qt.CursorShape.PointingHandCursor)
            self.group.addButton(b, i); row.addWidget(b); self.values.append(value)
        self.group.button(0).setChecked(True)
        self.group.idClicked.connect(self.currentIndexChanged.emit)
    def currentData(self): return self.values[max(0, self.group.checkedId())]
    def currentIndex(self): return self.group.checkedId()
    def findData(self, value): return self.values.index(value) if value in self.values else -1
    def setCurrentIndex(self, index):
        if 0 <= index < len(self.values): self.group.button(index).setChecked(True)

# Page states drawn by the grid. Labels come from the unchanged workflow events.
STATE_DONE, STATE_PARTIAL, STATE_FAILED, STATE_ACTIVE, STATE_PENDING, STATE_COVER = range(6)
STATE_ROLE = Qt.ItemDataRole.UserRole + 1
STATE_NAMES = {STATE_DONE: '完成', STATE_PARTIAL: '有未译', STATE_FAILED: '失败', STATE_ACTIVE: '处理中',
               STATE_PENDING: '待处理', STATE_COVER: '封面保留'}

def state_for(label):
    label = label or ''
    if '失败' in label: return STATE_FAILED
    if '处理中' in label: return STATE_ACTIVE
    if '封面' in label: return STATE_COVER
    if '未译' in label or '需检查' in label: return STATE_PARTIAL
    if '完成' in label or '跳过' in label or '已保存' in label: return STATE_DONE
    return STATE_PENDING

def _paint_cell(p, r, state, selected=False, text=''):
    fill, ink, border = None, QColor(MUTED), QColor(22, 20, 40, 60)
    if state == STATE_DONE: fill, ink = QColor(TEXT), QColor('#ffffff')
    elif state == STATE_COVER: fill, ink = QColor('#8f8ea0'), QColor('#ffffff')
    elif state == STATE_PARTIAL: fill, ink = QColor('#eef0b8'), QColor(TEXT)
    elif state == STATE_ACTIVE: fill, ink = QColor(VIOLET), QColor('#ffffff')
    elif state == STATE_FAILED: fill, border, ink = QColor('#ffffff'), QColor(TEXT), QColor(TEXT)
    if fill is not None: p.fillRect(r, fill)
    else: p.fillRect(r, QColor(255, 255, 255, 150))
    if state == STATE_PARTIAL:  # output saved, some text left in Japanese
        p.save(); p.setClipRect(r); p.setPen(QPen(QColor(154, 157, 23, 150), 1))
        for k in range(-int(r.height()), int(r.width()), 4):
            p.drawLine(QPointF(r.left() + k, r.bottom()), QPointF(r.left() + k + r.height(), r.top()))
        p.restore()
    if state == STATE_FAILED:
        p.setPen(QPen(QColor(TEXT), 1)); p.drawLine(r.topLeft(), r.bottomRight())
    if state in (STATE_PENDING, STATE_FAILED, STATE_PARTIAL):
        p.setPen(QPen(border, 1)); p.drawRect(r.adjusted(.5, .5, -.5, -.5))
    if selected:
        p.setPen(QPen(QColor(VIOLET) if state != STATE_ACTIVE else QColor(TEXT), 2)); p.drawRect(r.adjusted(1, 1, -1, -1))
    if text:
        font = QFont(UI_FONT); font.setPixelSize(10); font.setBold(state in (STATE_DONE, STATE_ACTIVE, STATE_COVER)); p.setFont(font)
        p.setPen(ink); p.drawText(r, Qt.AlignmentFlag.AlignCenter, text)

class _Cell(QStyledItemDelegate):
    def paint(self, p, option, index):
        r = QRectF(option.rect).adjusted(2, 2, -2, -2)
        p.save(); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        _paint_cell(p, r, index.data(STATE_ROLE), bool(option.state & QStyle.StateFlag.State_Selected), index.data(Qt.ItemDataRole.DisplayRole))
        p.restore()
    def sizeHint(self, option, index): return QSize(30, 30)

class PageGrid(QListWidget):
    """Numbered squares: the page list and the only progress display."""
    def __init__(self):
        super().__init__(); self.setViewMode(QListView.ViewMode.IconMode); self.setFlow(QListView.Flow.LeftToRight)
        self.setWrapping(True); self.setResizeMode(QListView.ResizeMode.Adjust); self.setMovement(QListView.Movement.Static)
        self.setGridSize(QSize(31, 31)); self.setUniformItemSizes(True); self.setSpacing(0)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setItemDelegate(_Cell(self)); self.setMouseTracking(True)
    def add_page(self, number, path, rel, label):
        item = QListWidgetItem(str(number)); item.setData(Qt.ItemDataRole.UserRole, str(path))
        self.addItem(item); self.mark(item, rel, label); return item
    def mark(self, item, rel, label):
        state = state_for(label); item.setData(STATE_ROLE, state)
        item.setToolTip(f'{rel}\n{label or STATE_NAMES[state]}'); self.viewport().update()
    def counts(self):
        states = [self.item(i).data(STATE_ROLE) for i in range(self.count())]
        return {s: states.count(s) for s in STATE_NAMES}

def legend():
    """Tiny key under the grid."""
    pix = QPixmap(230 * 2, 14 * 2); pix.setDevicePixelRatio(2); pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix); p.setRenderHint(QPainter.RenderHint.Antialiasing); font = QFont(UI_FONT); font.setPixelSize(11); p.setFont(font)
    x = 0
    for state in (STATE_DONE, STATE_PARTIAL, STATE_FAILED, STATE_PENDING):
        _paint_cell(p, QRectF(x, 2, 10, 10), state)
        p.setPen(QColor(MUTED)); p.setFont(font); p.drawText(QPointF(x + 15, 11.5), STATE_NAMES[state])
        x += 15 + p.fontMetrics().horizontalAdvance(STATE_NAMES[state]) + 14
    p.end(); return pix

def title_bar(widget, dark=False):
    """Match the Windows caption to the light field."""
    try:
        import ctypes
        value = ctypes.c_int(1 if dark else 0)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(int(widget.winId()), 20, ctypes.byref(value), ctypes.sizeof(value))
    except (AttributeError, OSError):
        pass

dark_title_bar = title_bar  # older imports (history_ui); the window is light now
