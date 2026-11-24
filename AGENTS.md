# Passage: instructions for AI assistants / 给 AI 助手

## Install and launch / 安装与启动

- Windows 10/11 x64, Python 3.13 x64. Open PowerShell in this repository root.
- Run `powershell -ExecutionPolicy Bypass -File install.ps1`. If Python is missing, follow the printed `winget install -e --id Python.Python.3.13` instruction, reopen the shell, and retry.
- The installer creates this clone's `.venv`, installs the locked dependencies, downloads and SHA-256 verifies all models and the Vulkan runtime, verifies bundled OFL fonts, and creates a desktop shortcut to `Launch.vbs`.
- When installation succeeds, run `wscript.exe .\Launch.vbs`. Use `Start-Passage.cmd` as an alternative. If startup fails, run `.\.venv\Scripts\python.exe app.py` to see errors.
- Do not change application source code merely to install it. Do not upload any user image, OCR text, translation, job record, or preferences. Installation downloads public dependencies only. Daily processing stays local.

## Boundaries / 操作边界

- 安装用本仓库独立目录，不修改另一个已存在的 Passage 安装。不要把 models、runtime、fonts 或 .venv 链接到用户正在使用的目录。
- 不运行用户的整本漫画作为安装测试，不重译或覆盖已有成品。未经用户要求，不改识别、翻译、清字或嵌字逻辑。
- 先读 README.md 和 THIRD_PARTY.txt。源码 MIT 不改变模型许可；Sakura 为 CC-BY-NC-SA-4.0，限制商业使用。
- 首次安装需要联网；启动时不下载。损坏或缺少的资源由用户明确重跑安装脚本处理。
- 桌面已有同名且指向其他安装的快捷方式时不得覆盖。验证安装使用 `-NoShortcut`，或用 `-ShortcutDirectory` 指向测试目录。
- 不将 `.venv/`、`models/`、`runtime/`、`exports/`、`evidence/`、`examples/`、`preferences.json`、`.passage`、真实漫画测试截图或本机专用维护脚本提交到 Git。
- 推送前展示实际文件清单并等待用户确认；仓库保持私有，公开需用户另行授权。

## Validation / 验证

```powershell
.\.venv\Scripts\python.exe setup_models.py --verify-only
.\.venv\Scripts\python.exe -m unittest discover -q
.\.venv\Scripts\python.exe scripts\smoke_install.py
```

`smoke_install.py` checks model inventory, external-network blocking, fonts and the UI on synthetic/empty input only. It does not translate a book. Private image regression fixtures are excluded; affected tests explicitly skip. Report skipped tests and any failed checks accurately. Never claim full-page quality verification from installer tests.

安装命令、路径和边界适用于 Claude Code、Codex、Gemini CLI、DeepSeek 等助手。CLAUDE.md 与本文件保持一致。
