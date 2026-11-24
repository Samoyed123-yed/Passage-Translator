"""Paper, ink and vermilion: a quiet desktop workspace for reading pages."""
STYLE = '''
QWidget {background:#f3f0e9;color:#282724;font-family:"Microsoft YaHei UI";font-size:13px;}
QMainWindow {background:#f3f0e9;}
QLabel#title {font-family:"Georgia";font-size:36px;font-weight:700;color:#292723;}
QLabel#sub {color:#716c64;font-size:12px;}
QLabel#tag {color:#70665c;background:#eae5dc;border:0;border-radius:3px;padding:7px 13px;}
QFrame#card {background:#faf8f3;border:1px solid #dcd6cc;border-radius:5px;}
QFrame#card QLabel {background:transparent;}
QLabel#section {font-size:13px;font-weight:600;color:#554e45;padding-top:10px;padding-bottom:3px;}
QLineEdit,QComboBox,QPlainTextEdit {background:#fffdf9;border:1px solid #d4cdc2;border-radius:3px;padding:9px;selection-background-color:#963d2c;selection-color:white;}
QLineEdit:focus,QComboBox:focus,QPlainTextEdit:focus {border:1px solid #a74b36;}
QLineEdit:read-only {color:#777168;background:#eeebe4;}
QComboBox QAbstractItemView {background:#fffdf9;color:#282724;selection-background-color:#e6d5ca;selection-color:#282724;}
QPushButton {background:#faf8f3;border:1px solid #d4cdc2;border-radius:3px;padding:9px 13px;color:#3e3932;}
QPushButton:hover {background:#eae3d8;border-color:#aaa093;}
QPushButton:focus {border:1px solid #9e3f2e;}
QPushButton:disabled {color:#9a958e;background:#eeebe5;border-color:#dfd9d0;}
QPushButton#primary {background:#a44733;color:#fffaf4;font-weight:600;border:0;padding:14px;}
QPushButton#primary:hover {background:#873827;}
QPushButton#primary:disabled {background:#d7b8ab;color:#fffaf4;}
QPushButton#disclosure {text-align:left;background:transparent;border:0;color:#71665a;padding:7px 0;}
QPushButton#disclosure:checked {color:#963d2c;}
QListWidget {background:#faf8f3;border:1px solid #dcd6cc;border-radius:3px;padding:4px;outline:0;}
QListWidget::item {padding:8px;border-bottom:1px solid #eee9e1;}
QListWidget::item:selected {background:#e9ddd0;color:#3e3129;border-left:3px solid #a44733;}
QGraphicsView {background:#e6e1d8;border:1px solid #d5cec2;border-radius:3px;}
QProgressBar {border:0;background:#e2dcd2;border-radius:2px;max-height:5px;text-align:center;}
QProgressBar::chunk {background:#a44733;border-radius:2px;}
QSplitter::handle {background:#f3f0e9;width:12px;}
QTableWidget {background:#fffdf9;gridline-color:#dfd8cc;selection-background-color:#e9ddd0;selection-color:#282724;}
QHeaderView::section {background:#eae4da;color:#51483d;padding:9px;border:0;}
QScrollBar:vertical {background:#eee9e1;width:9px;} QScrollBar::handle:vertical {background:#b7ada0;min-height:30px;border-radius:4px;}
QToolTip {background:#fffdf9;color:#282724;border:1px solid #b7ada0;padding:5px;}
'''
