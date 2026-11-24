# Passage — Offline Comic Translation Software

本地漫画翻译桌面应用，当前界面提供日文、英文到简体中文的翻译，包含文字识别、翻译、清字、嵌字和左右对照。后端保留韩文识别支持与资源，但当前界面没有韩文入口。界面使用 PySide6；模型在自己的电脑上运行。首次安装需要联网，完成后无需 API key，也不按页收费。

An offline Windows desktop application for translating Japanese and English comics into Simplified Chinese. It combines OCR, local translation, text removal, lettering and side-by-side review. Korean OCR support and assets remain in the backend, but the current UI has no Korean language selector. Setup downloads dependencies; everyday processing runs locally without an API key.

宣传视频 / Demo video：尚未提供链接 / Link not supplied yet.

## 安装 / Installation

需要 Windows 10/11 x64 和 **Python 3.13 x64**。建议 16 GB 内存、支持 Vulkan 的独立显卡和至少 6 GB 显存。无独显时可以尝试 CPU，但速度很慢；未对所有硬件组合保证兼容。资产和环境约占 10 GB，请至少预留 15 GB，安装期间还需要下载缓存空间。

Requires Windows 10/11 x64 and Python 3.13 x64. Recommended: 16 GB RAM, a Vulkan-capable discrete GPU with 6 GB or more VRAM, and 15 GB free disk space (approximately 10 GB installed, plus temporary downloads). CPU execution is much slower.

把 [Passage-Translator 仓库](https://github.com/Samoyed123-yed/Passage-Translator) 克隆或解压到一个**新的、可写的独立目录**。私有期间，需要先接受仓库拥有者发出的访问邀请，再登录 GitHub 下载；链接本身不会授予访问权限。公开后可直接点击 Code → Download ZIP，解压后使用，无需安装 Git。

Recipients can install from a downloaded source ZIP or a Git clone. While this repository is private, the owner must invite them first and they must sign in to download it. Once public, the source ZIP can be downloaded without Git. This is a Windows desktop installation, not a hosted web service.

在解压或克隆后的目录打开 PowerShell，执行：

```powershell
powershell -ExecutionPolicy Bypass -File install.ps1
```

If Python is missing / 如果提示缺少 Python：

```powershell
winget install -e --id Python.Python.3.13
```

重新打开 PowerShell，再运行安装命令。脚本兼容 Windows PowerShell 5.1，会创建本目录的 `.venv`、安装固定版本依赖、下载并校验模型和 llama.cpp Vulkan 运行时，然后建立指向 `Launch.vbs` 的桌面快捷方式。下载中断后可重跑，支持继续下载 `.part`；已有文件也必须通过大小和 SHA-256 校验，不能仅凭文件存在就跳过。

The installer checks Python, creates `.venv`, installs pinned dependencies, downloads and verifies assets, and creates a desktop shortcut. Rerun after an interrupted download. No administrator rights are needed for a normal writable installation. Do not place a clone inside an existing Passage installation.

验证副本或不想建立快捷方式时：

```powershell
powershell -ExecutionPolicy Bypass -File install.ps1 -NoShortcut
```

已有同名快捷方式指向其他安装时，脚本会保留它并提示；可以用 `-ShortcutDirectory '测试目录'` 在别处创建快捷方式。不要复制另一台电脑的 `.venv`，也不要把下载目录链接到正在使用的安装。

## 启动与使用 / Launch and use

双击桌面 Passage 快捷方式，或本目录的 `Launch.vbs` / `Start-Passage.cmd`。启动器使用相对位置，移动软件时需保持整个目录结构。

1. 选择原图文件夹，软件递归扫描其中的图片。
2. 选择主要语言和嵌字字体；混合语言页面仍需自行选主要语言。左下角的书本图标可打开术语表，填写人名和固定用语。
3. 点击开始翻译。输出默认是原图旁边的 `原文件夹名（中译）`，无需预先创建。
4. 完成后左右对照；滚轮缩放、拖动查看，点击“打开中文版文件夹”查阅成品。

Choose the source folder, primary language and lettering font, optionally provide names/terms using the book icon, then start. Results are written to a sibling Chinese-output folder. The originals are read-only; folder structure, image dimensions and supported file formats are preserved. PNG/WebP use lossless output, while JPEG is re-encoded at high quality. Animated/multipage inputs may be copied rather than translated.

普通非图片文件和空目录随结构保留；`.passage` 记录与链接跳转不复制。已有输出没有可验证的本次任务记录时不会随意覆盖。停止通常需要等当前模型调用结束；恢复处理会按来源、设置和摘要跳过已完成页面。改动语言、词表或原图可能触发重处理，重要原图应自行保留备份。

支持横排、竖排、对白和规则旁白，复杂背景、手写字和装饰字仍可能识别失败。自动筛选会保留部分拟声和艺术字；检查结果不等于人工译审。错误语义、漏字和修复痕迹都仍有可能发生。更改风格配置不会主动重做既有书籍。

`translation_style.json` 包含口语与成人语境词表，本仓库保留完整版本。当前界面没有独立的日常/成人切换，是否启用某个词条由现有程序根据原文和默认模式判断。配置不含个人漫画记录或用户图片。

## 离线与资源 / Offline operation and assets

- 日文翻译使用 Sakura-GalTransl-7B-v3.7 IQ4_XS；英文和韩文使用 Qwen3-4B Q4_K_M。
- 漫画检测、manga-ocr、PaddleOCR、LaMa 和英文覆盖模型全部随安装脚本下载到本地。韩文 Paddle 识别模型和备用 coverage 权重也纳入完整资源清单。
- 本机原来的 manga-ocr 安装只有两个 ONNX 和 vocab；当前适配器直接读取这些文件。分发安装另补齐上游配置和 tokenizer 文件，便于完整记录来源，未改变识别逻辑。
- 默认使用 Resource Han Rounded CN Medium/Bold/Heavy，字体及 OFL 许可随仓库提供；Noto Sans SC 三字重由安装脚本下载。
- `assets.lock.json` 固定下载地址、字节数和 SHA-256，也列出运行时压缩包内每个文件的校验值。模型、运行时与 Noto 字体二进制不提交到 Git。
- GPU 自动选择优先独显；`PASSAGE_GPU_DEVICE` 可以指定 llama.cpp 列出的设备。不要照抄另一台电脑的 `Vulkan1` 编号。
- 翻译进程只使用本机模型服务。`engine.offline_guard()` 拒绝 Python 非回环连接，llama-server 使用 `--offline` 并绑定 `127.0.0.1`。这不是系统级网络防火墙。

The app does not run the downloader at launch. Installation requires access to PyPI, GitHub, Hugging Face and the RapidAI ModelScope mirror. During processing, images and text stay local. Runtime jobs and error logs stay in `runtime/`; do not upload that folder. Completion cleanup and recovery depend on the existing application workflow.

## 检查与故障排查 / Checks and troubleshooting

```powershell
.\.venv\Scripts\python.exe setup_models.py --verify-only
.\.venv\Scripts\python.exe -m unittest discover -q
.\.venv\Scripts\python.exe scripts\smoke_install.py
```

校验命令不下载、不修改资产。下载资源缺失或损坏时重跑安装；如果报错的是仓库随附的 Resource Han Rounded 字体，请从源码 ZIP 或 Git 恢复该字体文件再安装。不能跳过校验或关闭 TLS 验证。SHA-256 不一致也可能意味着上游变更，需要维护者核实后更新资源锁；不要随意把新下载值当作正确值。

界面没有出现时，可运行 `.\.venv\Scripts\python.exe app.py` 查看错误。应用错误记录在 `runtime/last-error.txt`，模型服务日志在 `runtime/llama.log`。显存不足时先关闭占用显存的软件，避免同时开启多个翻译任务。记录可能包含个人路径或文本，分享前须删去这些内容。

单元测试使用合成数据；两项依赖真实漫画裁图的测试在分发版明确跳过，相关图片不分发。安装冒烟检查不运行整本漫画，不证明每页识别或排版质量。原 `使用说明.txt` 保留作历史详细参考，其中旧版本、旧模型及硬件描述不能代替当前代码和本 README。

## 许可 / Licensing

Passage 原创源码采用 [MIT](LICENSE)。第三方源码、字体、模型和运行时沿用各自许可，详见 [THIRD_PARTY.txt](THIRD_PARTY.txt)、[assets.lock.json](assets.lock.json) 以及其中的上游链接。MIT 不覆盖第三方资源的独立限制。

**Sakura 使用 CC-BY-NC-SA-4.0，限制商业使用。** 完整安装包含该模型，请在使用或再分发前阅读其[官方模型说明](https://huggingface.co/SakuraLLM/Sakura-GalTransl-7B-v3.7)。Qt/PySide6 以及其他依赖也有自己的许可。仓库只随附允许再分发的字体及其许可，不包含系统字体、用户原图或译图。

Original Passage code is MIT licensed. Dependencies retain their own licenses. In particular, the default Japanese model Sakura is CC-BY-NC-SA-4.0 and is not licensed for unrestricted commercial use. Bundled Resource Han Rounded fonts use OFL-1.1; the vendored adapter license is retained under `vendor/`.

## For AI assistants / 给 AI 助手

Read [AGENTS.md](AGENTS.md) or the identical [CLAUDE.md](CLAUDE.md). Run `install.ps1` in the repository root, then `Launch.vbs` after success. **Do not edit application code just to install it. Never upload user images or translation records.** Use a separate clone for validation, preserve existing installations and translated books, and report actual failures/skips rather than claiming success from file presence alone.

请按上述命令安装和启动；Python 缺失时先装指定版本。不自行“优化”翻译逻辑，不重译用户已有成品，不运行本机专用历史脚本。公开仓库和推送都要遵守用户的明确授权。
