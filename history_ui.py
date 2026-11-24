"""Internal OCR archive; text-only recovery does not write to comic folders."""
from pathlib import Path
from PySide6.QtCore import QThread,Signal,Qt
from PySide6.QtWidgets import QDialog,QVBoxLayout,QComboBox,QListWidget,QPushButton,QLabel,QFileDialog,QMessageBox
from recovery_catalog import catalog,retranslate

class TextWorker(QThread):
    result=Signal(str);failed=Signal(str)
    def __init__(self,job,dest):super().__init__();self.job=job;self.dest=dest
    def run(self):
        try:self.result.emit(str(retranslate(self.job,self.dest)))
        except Exception as e:self.failed.emit(str(e))

class HistoryDialog(QDialog):
    def __init__(self,parent=None):
        super().__init__(parent);self.setWindowTitle('历史文字记录');self.resize(760,520);self.worker=None
        from poster_ui import dark_title_bar;dark_title_bar(self)
        self.jobs=[j for j in catalog()['jobs'] if j['recoverable_pages']]
        layout=QVBoxLayout(self);layout.setContentsMargins(18,16,18,16);self.combo=QComboBox()
        for j in self.jobs:self.combo.addItem(f'{Path(j["source"]).name} · {len(j["recoverable_pages"])} 页')
        layout.addWidget(self.combo);self.pages=QListWidget();layout.addWidget(self.pages)
        self.note=QLabel('识别记录可能有误。重译只另存文字，不改漫画。');self.note.setWordWrap(True);self.note.setObjectName('muted');layout.addWidget(self.note)
        self.view=QPushButton('查看原文与旧译');self.view.clicked.connect(self.inspect);layout.addWidget(self.view)
        self.translate=QPushButton('用 Sakura 重译（另存）');self.translate.clicked.connect(self.start);layout.addWidget(self.translate)
        self.combo.currentIndexChanged.connect(self.refresh);self.refresh()
    def refresh(self):
        self.pages.clear();index=self.combo.currentIndex()
        if index<0:self.view.setEnabled(False);self.translate.setEnabled(False);return
        job=self.jobs[index];self.translate.setEnabled(job.get('language','ja')=='ja')
        for p in job['recoverable_pages']:self.pages.addItem(p['page'])
        if self.pages.count():self.pages.setCurrentRow(0)
    def inspect(self):
        i=self.combo.currentIndex();row=self.pages.currentRow()
        if i<0 or row<0:return
        from app import EditDialog
        EditDialog(self.jobs[i]['recoverable_pages'][row]['metadata'],self).exec()
    def start(self):
        job=self.jobs[self.combo.currentIndex()]
        target=QFileDialog.getExistingDirectory(self,'选择保存位置')
        if not target:return
        self.worker=TextWorker(job['job'],target);self.worker.result.connect(self.success);self.worker.failed.connect(self.failure)
        self.translate.setEnabled(False);self.combo.setEnabled(False);self.note.setText('正在重译…');self.worker.start()
    def success(self,path):self.note.setText('已保存：'+path);self.translate.setEnabled(True);self.combo.setEnabled(True)
    def failure(self,message):self.note.setText(message);self.translate.setEnabled(True);self.combo.setEnabled(True)
    def reject(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self,'正在重译','请等重译完成再关闭。');return
        super().reject()
