"""Local installation smoke check. No books, OCR corpus, or translation requests."""
from pathlib import Path
import argparse
import json
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--show-seconds', type=int, default=2)
    parser.add_argument('--load-models', action='store_true', help='Load ONNX and both local LLM backends without translating')
    args = parser.parse_args()
    import engine, sakura
    engine.offline_guard()
    for operation in [lambda: socket.getaddrinfo('example.com', 443),
                      lambda: socket.create_connection(('203.0.113.1', 443), timeout=1)]:
        try:
            operation()
        except OSError as error:
            assert 'Passage' in str(error), error
        else:
            raise AssertionError('External connection was not blocked')
    assert not engine.model_status(), engine.model_status()
    assert sakura.available()
    if args.load_models:
        for path in sorted((ROOT / 'models').rglob('*.onnx')):
            session = engine.session(path)
            assert session.get_inputs() and session.get_outputs()
            del session
            print('ONNX loaded:', path.relative_to(ROOT), flush=True)
        vision = engine.Vision()
        assert vision.ocr('ja').vocab
        assert vision.ocr('en').charset and vision.ocr('ko').charset
        from english_coverage import EnglishCoverage
        coverage = EnglishCoverage(ROOT)
        print('OCR adapters and English coverage initialized offline.', flush=True)
        del coverage, vision
        for backend in ('sakura', 'qwen'):
            translator = engine.LocalTranslator(print)
            try:
                translator.start(backend)
                print('Local server started:', backend, flush=True)
            finally:
                translator.close()
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QTimer
    from PySide6.QtGui import QIcon
    import app
    application = QApplication([])
    application.setApplicationName('Passage installation check')
    application.setStyle('Fusion')
    app.load_fonts()
    application.setStyleSheet(app.STYLE)
    application.setWindowIcon(QIcon(str(ROOT / 'passage.ico')))
    window = app.Window()
    window.setWindowTitle('Passage - installation check')
    window.show()
    assert window.isVisible()
    QTimer.singleShot(max(1, args.show_seconds) * 1000, application.quit)
    assert application.exec() == 0
    print(json.dumps({'ui': 'opened', 'external_python_connections': 'blocked',
                      'required_assets': 'present', 'translated_pages': 0}))

if __name__ == '__main__':
    main()
