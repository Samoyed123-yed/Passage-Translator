"""Native Windows desktop interface for Passage."""
import json, os, sys, traceback
from pathlib import Path
from PySide6.QtCore import Qt,QThread,Signal,QTimer,QLockFile
from PySide6.QtGui import QPixmap,QFont,QIcon,QPainter,QColor
from PySide6.QtWidgets import (QApplication,QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,
    QLineEdit,QFileDialog,QComboBox,QPlainTextEdit,QSplitter,QProgressBar,QMessageBox,QFrame,QGraphicsView,
    QGraphicsScene,QDialog,QTableWidget,QTableWidgetItem,QHeaderView,QDialogButtonBox)
import engine
from automatic import AutoBatch,sibling_output,read_manifest,scan_images,job_directory
from poster_ui import (STYLE,MUTED,PosterBackdrop,PageGrid,Segmented,icon_button,wordmark,legend,load_fonts,
                       title_bar,STATE_DONE,STATE_PARTIAL,STATE_FAILED,STATE_COVER)

ROOT=Path(__file__).resolve().parent

class ImageView(QGraphicsView):
    def __init__(self,empty_text='原图'):
        super().__init__(); self.setScene(QGraphicsScene(self)); self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform); self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.path=None; self.auto_fit=True; self.empty_text=empty_text
    def load(self,path):
        self.path=path; self.scene().clear(); self.resetTransform()
        if path and Path(path).exists():
            pix=QPixmap(str(path))
            if not pix.isNull(): self.scene().addPixmap(pix); self.scene().setSceneRect(pix.rect()); self.fit()
        else:
            font=QFont('Noto Sans SC'); font.setPixelSize(15); font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing,3)
            item=self.scene().addText(self.empty_text if not path else '尚未生成',font)
            item.setDefaultTextColor(QColor('#b9b8c8')); self.scene().setSceneRect(item.boundingRect())
    def fit(self):
        self.auto_fit=True
        # Only pages are scaled to the view; the empty-state label stays at its own size.
        if any(item.type()==7 for item in self.scene().items()):
            self.fitInView(self.scene().sceneRect(),Qt.AspectRatioMode.KeepAspectRatio)
        else:self.resetTransform()
    def resizeEvent(self,event):
        super().resizeEvent(event)
        if self.path and self.auto_fit:self.fit()
    def wheelEvent(self,event):
        self.auto_fit=False
        factor=1.2 if event.angleDelta().y()>0 else 1/1.2; self.scale(factor,factor)

class Worker(QThread):
    message=Signal(str); update=Signal(tuple); result=Signal(dict); failed=Signal(str)
    def __init__(self,source,output,language,glossary,files,reprocess=False):
        super().__init__(); self.args=(source,output,language,glossary); self.files=files; self.stop_requested=False;self.reprocess=reprocess
    def run(self):
        try:
            batch=AutoBatch(*self.args,log=self.message.emit,event=self.on_event,cancel=self.is_cancelled,reprocess=self.reprocess)
            self.result.emit(batch.run(self.files))
        except Exception as e:
            (ROOT/'runtime/last-error.txt').write_text(traceback.format_exc(),encoding='utf-8'); self.failed.emit(str(e))
    def on_event(self,*args): self.update.emit(args)
    def is_cancelled(self): return self.stop_requested

class RenderWorker(QThread):
    result=Signal(dict); failed=Signal(str)
    def __init__(self,path,draft=None): super().__init__(); self.path=path; self.draft=draft
    def run(self):
        try: self.result.emit(engine.rerender(self.path,self.draft))
        except Exception as e: self.failed.emit(str(e))

class EditDialog(QDialog):
    def __init__(self,path,parent):
        super().__init__(parent); self.path=Path(path); self.data=json.loads(self.path.read_text(encoding='utf-8')); self.worker=None
        self.setWindowTitle('校对'); self.resize(1120,560); title_bar(self)
        layout=QVBoxLayout(self); layout.setContentsMargins(18,16,18,16)
        note=QLabel('改好译文后点“重新嵌字”。清空译文＝保留原字；不勾选＝不翻译。'); note.setObjectName('muted'); layout.addWidget(note)
        from text_rules import show_in_editor
        self.visible_regions=[r for r in self.data['regions'] if show_in_editor(r)]
        self.table=QTableWidget(len(self.visible_regions),5); self.table.setHorizontalHeaderLabels(['翻译','原文','译文','状态','方向'])
        self.table.horizontalHeader().setSectionResizeMode(1,QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2,QHeaderView.ResizeMode.Stretch)
        self.table.setColumnWidth(0,55); self.table.setColumnWidth(3,150); self.table.setColumnWidth(4,110)
        for row,r in enumerate(self.visible_regions):
            check=QTableWidgetItem(); check.setFlags(Qt.ItemFlag.ItemIsUserCheckable|Qt.ItemFlag.ItemIsEnabled); check.setCheckState(Qt.CheckState.Checked if r.get('translation') and r.get('kind')!='sfx' else Qt.CheckState.Unchecked)
            self.table.setItem(row,0,check)
            for col,key in [(1,'source'),(2,'translation'),(3,'status')]:
                item=QTableWidgetItem(r.get(key,''))
                if col!=2: item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row,col,item)
            self.table.setRowHeight(row,78)
            flow=QComboBox()
            detected={'vertical':'竖排','horizontal':'横排'}.get(r.get('orientation'),'')
            flow.addItem(('自动 · '+detected) if detected else '自动','auto'); flow.addItem('竖排','vertical'); flow.addItem('横排','horizontal')
            flow.setCurrentIndex(max(0,flow.findData(r.get('direction_override','auto')))); self.table.setCellWidget(row,4,flow)
        layout.addWidget(self.table); self.status=QLabel(''); self.status.setObjectName('muted'); layout.addWidget(self.status)
        row=QHBoxLayout(); export=QPushButton('导出文字');export.clicked.connect(self.export_text);row.addWidget(export); row.addStretch()
        self.save=QPushButton('重新嵌字'); self.save.setObjectName('primary'); self.save.setMinimumWidth(220); self.save.clicked.connect(self.apply); row.addWidget(self.save)
        layout.addLayout(row)
        source=self.data.get('source','')
        if not source or not Path(source).is_file():
            self.save.setEnabled(False);self.status.setText('原图已不在，只能查看和导出文字。')
    def export_text(self):
        target,_=QFileDialog.getSaveFileName(self,'导出文字','','JSON (*.json)')
        if target:
            rows=[{'source':r.get('source',''),'translation':self.table.item(i,2).text()} for i,r in enumerate(self.visible_regions)]
            engine.atomic_json(Path(target),rows)
    def apply(self):
        for i,r in enumerate(self.visible_regions):
            r['translation']=self.table.item(i,2).text().strip()
            r['kind']='dialogue' if self.table.item(i,0).checkState()==Qt.CheckState.Checked else 'sfx'
            r.pop('skip_review',None)
            r['status']='已翻译' if r['kind']=='dialogue' and r['translation'] else '跳过特效字'
            r['direction_override']=self.table.cellWidget(i,4).currentData()
        self.save.setEnabled(False); self.table.setEnabled(False); self.status.setText('正在重新嵌字…')
        self.worker=RenderWorker(self.path,self.data); self.worker.result.connect(self.success); self.worker.failed.connect(self.failure); self.worker.start()
    def success(self,data): self.data=data; self.accept()
    def failure(self,message): self.status.setText(message); self.save.setEnabled(True); self.table.setEnabled(True)
    def reject(self):
        if self.worker and self.worker.isRunning(): return
        super().reject()

class GlossaryDialog(QDialog):
    """Edits the same glossary text the workflow already receives."""
    def __init__(self,text,parent):
        super().__init__(parent); self.setWindowTitle('术语表'); self.resize(460,380); title_bar(self)
        layout=QVBoxLayout(self); layout.setContentsMargins(18,16,18,16)
        self.edit=QPlainTextEdit(text); self.edit.setPlaceholderText('每行一个\nユキ->由纪 #男性\nsenpai->前辈'); layout.addWidget(self.edit)
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Save|QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText('保存'); buttons.button(QDialogButtonBox.StandardButton.Cancel).setText('取消')
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject); layout.addWidget(buttons)

def panel():
    frame=QFrame(); frame.setObjectName('panel'); return frame

def heading(text):
    label=QLabel(text); label.setObjectName('heading'); return label

class Window(QMainWindow):
    def __init__(self):
        load_fonts()
        super().__init__(); self.setWindowTitle('Passage'); self.resize(1400,900); self.setMinimumSize(1080,720)
        self.worker=None; self.files=[]; self.items={}; self.metadata={}; self.progress_count=0
        if (ROOT/'passage.ico').exists(): self.setWindowIcon(QIcon(str(ROOT/'passage.ico')))
        self.backdrop=backdrop=PosterBackdrop(132); self.setCentralWidget(backdrop)
        outer=QVBoxLayout(backdrop); outer.setContentsMargins(24,0,24,40); outer.setSpacing(12)

        # The banner is the arrow poster itself; only the name and status sit on it.
        self.banner=banner=QWidget(); banner.setFixedHeight(132); header=QHBoxLayout(banner); header.setContentsMargins(10,0,4,0); header.setSpacing(12)
        # Name sits on a faint veil so it stays legible over the arrows.
        plate=QFrame(); plate.setObjectName('plate'); brand=QVBoxLayout(plate); brand.setContentsMargins(14,8,16,8); brand.setSpacing(4)
        mark=QLabel(); mark.setPixmap(wordmark()); brand.addWidget(mark)
        header.addWidget(plate,0,Qt.AlignmentFlag.AlignVCenter); header.addStretch()
        self.models=QLabel(); self.models.setObjectName('chip'); header.addWidget(self.models,0,Qt.AlignmentFlag.AlignTop|Qt.AlignmentFlag.AlignRight)
        header.setAlignment(self.models,Qt.AlignmentFlag.AlignTop); header.setContentsMargins(0,0,0,0)
        outer.addWidget(banner)

        body=QHBoxLayout(); body.setSpacing(16); outer.addLayout(body,1)
        left=panel(); left.setFixedWidth(268); ll=QVBoxLayout(left); ll.setContentsMargins(18,18,18,14); ll.setSpacing(10)
        ll.addWidget(heading('选择作品'))
        self.source=QLineEdit(); self.source.setPlaceholderText('原图文件夹'); ll.addWidget(self.source)
        pick=QPushButton('选择原图文件夹'); pick.clicked.connect(self.choose_source); ll.addWidget(pick)
        self.pick=pick
        # Kept for the existing workflow calls; the sibling “（中译）” folder is automatic.
        self.output=QLineEdit(left); self.output.setReadOnly(True); self.output.hide()
        self.source.textChanged.connect(self.update_output)
        self.scan_button=QPushButton('刷新',left); self.scan_button.clicked.connect(self.scan_files); self.scan_button.hide()
        ll.addSpacing(8); ll.addWidget(heading('原文语言'))
        self.language=Segmented([('日文','ja'),('英文','en')]); ll.addWidget(self.language)
        ll.addSpacing(4); ll.addWidget(heading('嵌字字体'))
        import fonts
        self.lettering=Segmented([(fonts.FAMILIES[k][0],k) for k in ('rounded','noto','yahei') if k in fonts.available()] or [('雅黑','yahei')]); ll.addWidget(self.lettering)
        self.lettering.currentIndexChanged.connect(lambda _:self.preferences())
        ll.addSpacing(12)
        self.start=QPushButton('开始翻译'); self.start.setObjectName('primary'); self.start.setCursor(Qt.CursorShape.PointingHandCursor); self.start.clicked.connect(self.start_batch); ll.addWidget(self.start)
        self.stop=QPushButton('停止'); self.stop.setObjectName('stop'); self.stop.setEnabled(False); self.stop.clicked.connect(self.stop_batch); ll.addWidget(self.stop); self.stop.hide()
        self.openout=QPushButton('打开中文版文件夹'); self.openout.clicked.connect(self.open_output); ll.addWidget(self.openout)
        ll.addSpacing(12)
        pages=QHBoxLayout(); pages.addWidget(heading('页面')); pages.addStretch()
        self.count=QLabel(''); self.count.setObjectName('count'); pages.addWidget(self.count); ll.addLayout(pages)
        self.queue=PageGrid(); self.queue.currentItemChanged.connect(self.select_page); ll.addWidget(self.queue,1)
        key=QLabel(); key.setPixmap(legend()); ll.addWidget(key)
        tools=QHBoxLayout(); tools.setSpacing(4)
        self.glossary_button=icon_button('book','术语表'); self.glossary_button.clicked.connect(self.open_glossary); tools.addWidget(self.glossary_button)
        self.archive=icon_button('clock','历史文字记录'); self.archive.clicked.connect(self.open_archive); tools.addWidget(self.archive)
        self.log_toggle=icon_button('log','处理日志',True); tools.addWidget(self.log_toggle); tools.addStretch()
        ll.addLayout(tools)
        self.glossary=QPlainTextEdit(left); self.glossary.hide()
        body.addWidget(left)

        right=panel(); right.setObjectName('stage'); rr=QVBoxLayout(right); rr.setContentsMargins(18,14,18,14); rr.setSpacing(10)
        backdrop.set_panels(left,right)
        bar=QHBoxLayout(); bar.setSpacing(4)
        self.page_name=QLabel(''); self.page_name.setObjectName('muted'); bar.addWidget(self.page_name); bar.addStretch()
        fit=icon_button('fit','适合窗口'); fit.clicked.connect(self.fit); bar.addWidget(fit)
        self.edit=icon_button('pen','校对此页'); self.edit.setEnabled(False);self.edit.clicked.connect(self.edit_page);bar.addWidget(self.edit)
        self.redo=icon_button("redo","重做此页");self.redo.setEnabled(False);self.redo.clicked.connect(self.redo_page);bar.addWidget(self.redo);rr.addLayout(bar)
        previews=QSplitter()
        for attr,label,empty in (('original_view','原图','原图'),('translated_view','中文版','中文版')):
            box=QWidget(); col=QVBoxLayout(box); col.setContentsMargins(0,0,0,0); col.setSpacing(0)
            tag=QLabel(label); tag.setObjectName('tag'); col.addWidget(tag)
            view=ImageView(empty); setattr(self,attr,view); col.addWidget(view); previews.addWidget(box)
        previews.setSizes([500,500]); rr.addWidget(previews,1)
        backdrop.set_panels(left,right,(self.original_view,self.translated_view))
        self.status=QLabel(''); self.status.setObjectName('status'); rr.addWidget(self.status)
        self.log=QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumBlockCount(300); self.log.setFixedHeight(130); rr.addWidget(self.log); self.log.hide()
        self.log_toggle.toggled.connect(self.log.setVisible)
        body.addWidget(right,1)
        # Progress is shown by the page grid; this bar keeps the existing counters.
        self.progress=QProgressBar(right); self.progress.hide()

        self.original_view.load(None); self.translated_view.load(None); self.refresh_models(); self.load_preferences()
        self.timer=QTimer(self); self.timer.timeout.connect(self.refresh_models); self.timer.start(5000)
        title_bar(self)
    def resizeEvent(self,event):
        super().resizeEvent(event)
        # Short windows get a lower banner so the controls never get squeezed.
        height=132 if self.height()>=820 else 112
        if self.banner.height()!=height:
            self.banner.setFixedHeight(height); self.backdrop.header=height; self.backdrop.debounce.start()
    def rel(self,source):
        try:return str(Path(source).relative_to(Path(self.source.text()).resolve()))
        except ValueError:
            try:return str(Path(source).relative_to(Path(self.source.text())))
            except ValueError:return Path(source).name
    def mark(self,source,label):
        item=self.items.get(source)
        if item:self.queue.mark(item,self.rel(source),label)
        self.update_count()
    def update_count(self):
        c=self.queue.counts();done=c[STATE_DONE]+c[STATE_PARTIAL]+c[STATE_COVER]
        self.count.setText(f'{done} / {self.queue.count()}' if self.queue.count() else '')
    def refresh_models(self):
        missing=engine.model_status()
        import sakura
        names='日文 Sakura 7B · 英文 Qwen3 4B' if sakura.available() else 'Qwen3 4B'
        self.models.setText(f'缺少 {len(missing)} 个模型文件' if missing else '模型就绪')
        self.models.setToolTip('\n'.join(missing) if missing else names)
        if not self.worker or not self.worker.isRunning(): self.start.setEnabled(not missing)
    def load_preferences(self):
        try:
            p=json.loads((ROOT/'preferences.json').read_text(encoding='utf-8'))
            self.source.setText(p.get('source','')); self.update_output(); self.language.setCurrentIndex(max(0,self.language.findData(p.get('language','ja')))); self.glossary.setPlainText(p.get('glossary',''))
            self.lettering.setCurrentIndex(max(0,self.lettering.findData(p.get('lettering_font','rounded'))))
        except (OSError,ValueError): pass
    def preferences(self):
        # lettering_font is read by fonts.current() when pages are lettered.
        engine.atomic_json(ROOT/'preferences.json',{'source':self.source.text(),'output':self.output.text(),'language':self.language.currentData(),
                                                    'glossary':self.glossary.toPlainText(),'lettering_font':self.lettering.currentData()})
    def choose_source(self):
        folder=QFileDialog.getExistingDirectory(self,'选择原图文件夹',self.source.text())
        if folder:
            self.source.setText(folder); self.update_output(); self.scan_files()
    def update_output(self):
        try:self.output.setText(str(sibling_output(self.source.text())) if self.source.text().strip() else '')
        except (ValueError,OSError):self.output.clear()
        self.openout.setToolTip(self.output.text())
    def choose_output(self):
        folder=QFileDialog.getExistingDirectory(self,'选择中文版文件夹',self.output.text())
        if folder: self.output.setText(folder)
    def scan_files(self):
        self.files=[]
        try:
            self.update_output()
            source,output=engine.validate_roots(self.source.text(),self.output.text()); self.files=scan_images(source)
            saved=read_manifest(source,output) or {};pages=saved.get('pages',{})
            self.queue.clear(); self.items={}; self.metadata={}
            for number,p in enumerate(self.files,1):
                rel=p.relative_to(source); label='待处理'
                if str(rel) in pages and (output/rel).exists():
                    record=pages[str(rel)]
                    meta=job_directory(source,output)/'working'/rel.parent/(rel.name+'.json')
                    if meta.is_file():self.metadata[str(p)]=str(meta)
                    label='封面原样保留' if record.get('note')=='封面海报原样保留' else ('有未译文字' if record.get('retained') else '完成')
                self.items[str(p)]=self.queue.add_page(number,p,rel,label)
            self.update_count()
            c=self.queue.counts()
            self.status.setText(f'{len(self.files)} 页' + (f' · 有未译 {c[STATE_PARTIAL]} 页' if c[STATE_PARTIAL] else ''))
            self.progress.setMaximum(max(1,len(self.files)));self.progress.setValue(c[STATE_DONE]+c[STATE_PARTIAL]+c[STATE_COVER])
            if self.files: self.queue.setCurrentRow(0)
            self.preferences()
        except Exception as e: QMessageBox.warning(self,'无法读取',str(e))
    def select_page(self,current,previous=None):
        if not current: return
        source=current.data(Qt.ItemDataRole.UserRole)
        try: rel=Path(source).relative_to(Path(self.source.text()).resolve())
        except ValueError: return
        output=Path(self.output.text())/rel
        self.original_view.load(source); self.translated_view.load(output)
        self.page_name.setText(str(rel))
        idle=not self.worker or not self.worker.isRunning()
        self.edit.setEnabled(idle and bool(self.metadata.get(source)))
        self.redo.setEnabled(idle and Path(source).is_file())
    def fit(self): self.original_view.fit(); self.translated_view.fit()
    def set_busy(self,busy):
        self.stop.setVisible(busy);self.start.setVisible(not busy);self.redo.setEnabled(not busy);self.archive.setEnabled(not busy)
        self.start.setEnabled(not busy); self.stop.setEnabled(busy); self.scan_button.setEnabled(not busy)
        self.source.setEnabled(not busy); self.output.setEnabled(not busy); self.language.setEnabled(not busy); self.lettering.setEnabled(not busy); self.glossary.setEnabled(not busy); self.edit.setEnabled(False);self.backdrop.set_animating(busy)
        self.glossary_button.setEnabled(not busy); self.pick.setEnabled(not busy)
    def start_batch(self,checked=False,selected=None):
        self.scan_files()
        if selected:self.files=[Path(selected)]
        if not self.files: return
        try: engine.validate_roots(self.source.text(),self.output.text())
        except Exception as e: QMessageBox.warning(self,'无法开始',str(e)); return
        self.preferences(); self.set_busy(True); self.progress_count=0; self.progress.setRange(0,len(self.files)); self.progress.setValue(0)
        self.worker=Worker(self.source.text(),self.output.text(),self.language.currentData(),self.glossary.toPlainText(),self.files.copy(),reprocess=bool(selected))
        self.worker.message.connect(self.on_log); self.worker.update.connect(self.on_update); self.worker.result.connect(self.on_result); self.worker.failed.connect(self.on_failure); self.worker.finished.connect(self.on_finished); self.worker.start()
    def redo_page(self):
        item=self.queue.currentItem()
        if not item:return
        source=item.data(Qt.ItemDataRole.UserRole)
        if QMessageBox.question(self,'重做此页','重新翻译这一页？其他页不变。')!=QMessageBox.StandardButton.Yes:return
        self.start_batch(selected=source)
    def open_archive(self):
        from history_ui import HistoryDialog
        dialog=HistoryDialog(self);dialog.exec()
    def open_glossary(self):
        dialog=GlossaryDialog(self.glossary.toPlainText(),self)
        if dialog.exec():self.glossary.setPlainText(dialog.edit.toPlainText());self.preferences()
    def on_log(self,text): self.status.setText(text); self.log.appendPlainText(text)
    def on_update(self,args):
        kind=args[0]; source=args[1]
        if kind=='page':
            self.mark(source,'处理中'); self.queue.setCurrentItem(self.items[source])
        elif kind in ('done','error'):
            self.progress_count+=1; self.progress.setValue(self.progress_count)
            if kind=='done': self.metadata[source]=args[4]; label=args[3]
            else: label='失败：'+args[2]
            self.mark(source,label)
            if self.queue.currentItem()==self.items[source]: self.select_page(self.items[source])
    def on_result(self,result):
        lead='已停止' if result.get('cancelled') else '完成'
        parts=[f'{lead} · 输出 {result["completed"]+result["resumed"]} 页']
        if result['failed']:parts.append(f'失败 {result["failed"]}')
        if result.get('retained_pages'):parts.append(f'有未译 {result["retained_pages"]}')
        self.on_log(' · '.join(parts))
    def on_failure(self,message): self.on_log('出错：'+message); QMessageBox.warning(self,'出错',message)
    def on_finished(self): self.set_busy(False); self.refresh_models(); self.select_page(self.queue.currentItem())
    def stop_batch(self):
        if self.worker: self.worker.stop_requested=True; self.stop.setEnabled(False); self.status.setText('正在停止…')
    def open_output(self):
        if Path(self.output.text()).is_dir(): os.startfile(self.output.text())
    def edit_page(self):
        item=self.queue.currentItem()
        if not item: return
        meta=self.metadata.get(item.data(Qt.ItemDataRole.UserRole))
        if meta:
            dialog=EditDialog(meta,self); dialog.exec(); dialog.deleteLater(); self.select_page(item)
            saved=json.loads(Path(meta).read_text(encoding='utf-8')); reviews=saved.get('review_count',0)
            self.mark(item.data(Qt.ItemDataRole.UserRole),f'需检查 {reviews} 处' if reviews else '完成')
    def closeEvent(self,event):
        if self.worker and self.worker.isRunning():
            self.stop_batch(); QMessageBox.information(self,'正在停止','请等当前页结束后再关闭。'); event.ignore(); return
        self.preferences(); event.accept()

def main():
    try:
        import ctypes; ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('Passage.Desktop')
    except (AttributeError,OSError): pass
    app=QApplication(sys.argv); app.setApplicationName('Passage'); app.setStyle('Fusion'); load_fonts(); app.setStyleSheet(STYLE)
    if (ROOT/'passage.ico').exists(): app.setWindowIcon(QIcon(str(ROOT/'passage.ico')))
    lock=QLockFile(str(ROOT/'runtime/desktop.lock')); lock.setStaleLockTime(0)
    if not lock.tryLock(0):
        QMessageBox.information(None,'Passage','Passage 已经打开。'); return
    window=Window(); window.show()
    if '--sample' in sys.argv:
        sample=ROOT/'examples/英文'; window.source.setText(str(sample)); window.output.setText(str(ROOT/'examples/英文_中文')); window.language.setCurrentIndex(1); window.scan_files()
    if '--resume' in sys.argv and window.source.text() and window.output.text():
        # Continue the remembered book after an update: finished pages are verified and kept.
        QTimer.singleShot(1500, window.start_batch)
    sys.exit(app.exec())

if __name__=='__main__': main()
