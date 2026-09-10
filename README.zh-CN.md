<p align="center">
  <img src="studio-web/src/assets/smart-badminton-mark.png" alt="Smart Badminton 标志" width="200">
</p>

<h1 align="center">Smart Badminton</h1>

<p align="center">留下精彩回合，剪去等待时间，换个角度看你的比赛。</p>

<p align="center">
  <a href="README.md" lang="en">English</a> | <strong>简体中文</strong>
</p>

<p align="center">
  <a href="https://smart-badminton-tau.vercel.app/">网页版</a> ·
  <a href="#从一场比赛开始">快速开始</a> ·
  <a href="docs/studio-tutorial.zh-CN.md">使用指南</a> ·
  <a href="https://github.com/TempestShaw/Smart-Badminton/releases">版本下载</a>
</p>

<p align="center">
  <a href="https://github.com/TempestShaw/Smart-Badminton/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/TempestShaw/Smart-Badminton/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python 3.10 或更高版本" src="https://img.shields.io/badge/Python-3.10%2B-3776AB">
  <a href="LICENSE"><img alt="Apache-2.0 许可证" src="https://img.shields.io/badge/license-Apache--2.0-111827"></a>
</p>

Smart Badminton 把固定机位拍摄的羽毛球录像整理成回合精彩片段。先找到候选回合，再微调边界，最后导出想要重看的部分。浏览器剪辑和本机分析过程中，视频保留在你的设备上。

## 从一场比赛开始

[打开网页版](https://smart-badminton-tau.vercel.app/)，选择**浏览器版**：

1. 选择设备上的比赛视频。
2. 运行**快速分析**，寻找候选回合。
3. 检查片段，调整开始和结束时间。
4. 点击**输出 MP4**，保存剪辑结果。

浏览器版无需安装或注册账号，使用轻量画面活动分析和 FFmpeg WebAssembly。目前输出支持的源文件大小上限为 1.5 GB；更长的录像或完整姿态、球路和比分分析请使用原生精准版。

两个版本都支持中文和英文，首次打开会根据浏览器偏好自动选择支持的语言，并记住手动选择的语言与日夜模式。

## 为什么使用 Smart Badminton？

- **少花时间寻找比赛片段。** 从候选回合边界开始，而不是逐段拖过每次捡球和等待。
- **每一刀都由你确认。** 检查最后一拍、拖动边界、逐帧微调、分割片段，并随时撤销。自动结果是可以修正的初稿。
- **在自己的设备上处理录像。** 浏览器版在浏览器中处理视频；原生精准版在本机运行 Python 分析，并从原始录像渲染成片。

## 选择适合的版本

| | 浏览器快速版 | 原生精准版 |
| --- | --- | --- |
| 开始使用 | [打开网页版](https://smart-badminton-tau.vercel.app/) | 安装并运行本地 Studio |
| 回合识别 | 轻量画面活动分析 | 经过球场校准的多模态分析 |
| 剪辑 | 边界调整、分割／新增／删除、本机时间轴保存 | 详细时间轴、逐帧微调、撤销／重做、成片预览 |
| 分析 | 候选回合片段 | Hybrid 羽球追踪、人物姿态和基于证据的比分计算 |
| 输出 | 浏览器 WebAssembly 输出 MP4 | 本机 FFmpeg 从原始录像渲染 |
| 配置 | 无需安装 | Python；视觉分析另外需要模型和球场校准 |

原生精准版是本地配套应用。网站提供打开本地 Studio 的入口，Python 和 CUDA 分析不会在 Vercel 上运行。

## 在自己的电脑上使用原生精准版

需要 Python 3.10 或更高版本。通过仓库源码安装，可获得最新界面和双语支持：

```bash
git clone https://github.com/TempestShaw/Smart-Badminton.git
cd Smart-Badminton
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[studio]"
```

Windows 请将激活环境的命令替换为 `.venv\Scripts\activate`。

打开存有自己比赛录像的文件夹：

```bash
smart-badminton studio --library /path/to/matches
```

Studio 会打开 [127.0.0.1:8765](http://127.0.0.1:8765)。选择视频和输出位置，编辑时间轴，并在导出前保存。基础 Studio 安装支持手工剪片；自动视觉分析需要额外依赖、模型和球场校准。

需要完整视觉功能时，再安装：

```bash
python -m pip install -e ".[full]"
smart-badminton install-models --directory models
```

在 Studio 中框选有效比赛场地，生成并微调辅助区域，保存校准后再分析。CUDA 为可选加速；也支持 CPU 视觉分析，但速度较慢。详见[使用指南](docs/studio-tutorial.zh-CN.md)和[模型配置](models/README.md)。

[版本下载](https://github.com/TempestShaw/Smart-Badminton/releases)也提供 Python wheel 和独立的离线模型包。标签版本可能晚于 `main` 的最新更新。

## 直接在终端中使用

校对后的时间表是一个小型 CSV 文件。将以下示例保存为 `rallies.csv`，并使用时长足够覆盖这些区间的录像：

```csv
rally,start_seconds,end_seconds
1,2.600,8.750
2,11.770,25.800
```

从自己的录像中只渲染这些区间：

```bash
smart-badminton render --video match.mp4 --rallies rallies.csv --output highlights.mp4
```

CLI 也提供独立的特征提取、回合预测、羽球追踪、比分分析和模型训练命令。详见[命令参考](smart_badminton/README.md)和[进阶工作流](docs/advanced-workflows.md)。

## 探索与贡献

- [简体中文使用指南](docs/studio-tutorial.zh-CN.md)和[英文使用指南](docs/studio-tutorial.en.md)。
- [架构](docs/architecture.md)、[本地 Studio API](docs/studio-api.md)和[网页部署](docs/deployment.md)。
- [模型卡](models/MODEL_CARD.md)、[模型配置](models/README.md)和[进阶研究工作流](docs/advanced-workflows.md)。
- [合成示例项目](examples/privacy_safe_sample)：包含元数据和标注，不含真实人物、场馆录像或音频。
- [报告问题](https://github.com/TempestShaw/Smart-Badminton/issues/new)：请提供使用的版本、操作系统、复现步骤和错误信息，不要附上凭据或私人录像。

前端开发与检查：

```bash
cd studio-web
npm ci
npm run dev
```

打开 [localhost:3000/?engine=browser](http://localhost:3000/?engine=browser)。提交界面修改前运行 `npm run check`、`npm test` 和 `npm run build`。`npm run build:embedded` 会更新本地 Studio 使用的前端。CI 还会测试 Python 3.10、3.12，并构建 Python wheel。

修改安装方式或功能说明时，请同步更新中英文 README。

## 信任与许可证

项目仍处于早期研究阶段。在无人值守批量处理前，请检查回合结尾，并用具有代表性的录像验证效果。证据不足的结果应保持未知；球路、比分及探索性动作统计不保证等同真实标注。

正常浏览器剪辑和本机分析使用本地录像。可选的多模态比分标注会把选定的证据发送到你配置的模型服务，并需要单独设置凭据，详见[使用指南](docs/studio-tutorial.zh-CN.md)。

项目源码和已发布的回合状态模型采用 [Apache-2.0](LICENSE)。完整视觉模型包还包含 AGPL-3.0 和 MIT 组件，分别遵循其许可证。模型文件附有与校验和绑定的许可证说明，详见[第三方声明](THIRD_PARTY_NOTICES.md)。
