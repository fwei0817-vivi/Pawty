# Pawty 🐾

一个多模块的宠物互动体验平台，包含前端 Web 应用、AI 图像风格转换服务、RAG 聊天顾问以及配套的数据资产。

## 仓库结构

```
Pawty/
├── frontend/
│   └── pawty-web/               # 前端单页应用（index.html、静态资源）
├── backend/
│   └── image-style/             # Stable Diffusion 图像风格转换 FastAPI 服务
├── llm/
│   └── rag-service/             # RAG + LLM 顾问服务（FastAPI + LangChain）
├── data/
│   ├── knowledge/               # JSONL 知识库数据
│   └── vectorstores/            # 持久化的 Chroma 索引（运行后自动生成/更新）
└── README.md
```

### 模块说明

- **frontend/pawty-web**  
  原 `project-Pawty` 文件夹。直接打开 `index.html` 即可本地预览；若需要部署，可通过任意静态托管服务（Vercel、Netlify、Nginx 等）。

- **backend/image-style**  
  基于 Stable Diffusion Img2Img 的 AI 变身服务，提供 `POST /stylize` 接口，负责处理前端“变身”功能的图片上传与风格生成。

- **llm/rag-service**  
  RAG（Retrieval-Augmented Generation）顾问，负责“聊聊”模块。使用 Chroma 作为向量库，OpenAI 作为 LLM，暴露 `POST /ask` 接口。

- **data**  
  - `knowledge/`：存放 JSONL 原始知识文档（前缀可自由扩展）。  
  - `vectorstores/`：存放构建后的向量库，例如 `chroma_pawty/`。`rag_build.py --build` 会在此处写入/覆盖。

## 快速开始

### 1. 前端

```bash
# 直接双击或用任何静态服务器打开
open frontend/pawty-web/index.html
```

若部署到线上，记得在 `frontend/pawty-web/index.html` 中的 `API_ENDPOINTS` 调整后端域名。

### 2. 图像风格服务（变身）

```bash
cd backend/image-style
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python server.py --host 0.0.0.0 --port 8011   # 可选：--lazy（首次请求再加载模型）
```

可选环境变量：
- `PET_IMAGE_MODEL_ID`：自定义 Stable Diffusion 模型（默认 `runwayml/stable-diffusion-v1-5`）
- `HUGGINGFACE_TOKEN`：若模型需要 HuggingFace 授权

### 3. RAG 聊天顾问（聊聊）

```bash
cd llm/rag-service
python -m venv .venv
source .venv/bin/activate
pip install -r requirement.txt
OPENAI_API_KEY=sk-xxx python rag_build.py --build          # 生成向量库
OPENAI_API_KEY=sk-xxx python rag_build.py --serve --port 8001
```

服务启动后，前端会调用 `http://localhost:8001/ask` 获取宠物健康建议。

### 4. 数据与向量库

- 将新的 JSONL 资料放入 `data/knowledge/`，再次执行 `rag_build.py --build` 即可更新。
- 生成的 Chroma 数据位于 `data/vectorstores/chroma_pawty/`，可根据需要备份或清理。

## 常见工作流

1. 启动 RAG 服务与图像风格服务。
2. 打开前端页面进行联调。
3. 若更新知识数据，重复执行 `--build`。

## 授权

MIT License - 欢迎贡献与改进。🧡