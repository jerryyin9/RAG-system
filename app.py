import time
import logging
from datetime import timedelta
import pandas as pd
import streamlit as st
from langchain_core.messages import HumanMessage
from rag_core import RAGConfig, UniversalRAGEngine, RAGChatBot, SecretManager

logger_app = logging.getLogger("app")

@st.cache_resource
def _get_cached_embeddings(model_name: str, device: str):
    """Process-level cache for the embedding model.
    @st.cache_resource survives reruns, tab switches and session resets.
    The old module-level _embedding_singleton only survived within one import."""
    from langchain_huggingface import HuggingFaceEmbeddings
    logger_app.info("st.cache_resource: loading embedding model (truly once per process)...")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

# --- 页面配置 ---
st.set_page_config(page_title="RAG 知识库助手", layout="wide")

# --- CSS 优化日志显示, 自定义日志区样式 (白底深蓝字，自动滚动) ---
st.markdown("""
<style>
    /* A. 极致压缩顶部空间 - 解决标题过低问题 */
    .stApp { margin-top: -50px !important; }
    .block-container { 
        padding-top: 2rem !important; 
        padding-bottom: 12rem !important; /* 给底部输入框留出足够的垫片空间 */
        max-width: 90% !important; 
    }
    header { visibility: hidden; } /* 隐藏顶部装饰条 */

    /* B. 标题样式 */
    h1 { margin-top: -10px !important; padding-bottom: 15px; font-size: 2.2rem !important; }

    /* C. 按钮对齐 - 解决对齐不准问题 */
    div[data-testid="stButton"] button {
        margin-top: 28px !important;
        height: 45px;
        border-radius: 8px;
    }

    /* D. 仿 Gemini 输入框 - 自动避让侧边栏且样式轻盈 */
    /* 核心：不要用 position: fixed，利用原生容器进行样式穿透 */
    [data-testid="stChatInput"] {
        background-color: transparent !important;
        padding-bottom: 25px !important;
    }

    [data-testid="stChatInput"] textarea {
        /* 细边框与轻微阴影 */
        border: 1px solid #e0e4e8 !important;
        /*box-shadow: 0 1px 2px rgba(0,0,0,0.05), 0 8px 16px rgba(0,0,0,0.05) !important;*/
        border-radius: 12px !important;
        /* 默认两行高度 */
        min-height: 65px !important;
        background-color: #ffffff !important;
        font-size: 1rem !important;
        line-height: 1.5 !important;
    }

    /* 4. 消息气泡风格 */
    .stChatMessage {
        border-radius: 20px;
        margin-bottom: 15px;
        max-width: 85%;
    }

    /* F. Metric cards used in the ingestion dashboard */
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 10px 14px;
        text-align: center;
    }
    .metric-label { font-size: 0.75rem; color: #6c757d; }
    .metric-value { font-size: 1.4rem; font-weight: 600; color: #212529; }

    /* G. Log area */
    .log-container {
        background: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 10px;
        height: 300px;
        overflow-y: auto;
        font-family: monospace;
        font-size: 0.82rem;
        color: #1a2a4a;
    }
    .log-entry { padding: 2px 0; border-bottom: 1px solid #f1f3f5; }
</style>
""", unsafe_allow_html=True)

# --- 会话状态管理 (Session State) ---
# Session state bootstrap
_STATE_DEFAULTS = {
    "run_state":          "idle",   # idle | running | paused | stopped
    "logs":               [],
    "start_time_stamp":   None,
    "last_processed_idx": 0,
    "chat_messages":      [],
    "engine":             None,     # APP-FIX-1: persistent engine instance
    "current_coll_name":  None,
}
for key, default in _STATE_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


# --- 侧边栏：参数配置 ---
with st.sidebar:
    st.header("⚙️ 系统配置")
    
    st.subheader("🤖 模型配置（可选）")
    embedding_model_name = st.text_input(
        "Embedding 模型",
        value="BAAI/bge-m3",
        help="Embedding 模型建议固定不频繁切换；切换后建议重启 Streamlit 进程释放旧模型内存。",
    )
    llm_model_name = st.text_input(
        "对话LLM模型 ID",
        value="accounts/fireworks/models/llama-v3p3-70b-instruct",
        help="高级用法：填写 Fireworks 上对应的模型 ID，不懂可保持默认。"
    )
    # APP-FIX-2: LLM base URL is now configurable
    llm_base_url = st.text_input(
        "对话LLM模型 Base URL",
        value="https://api.fireworks.ai/inference/v1",
        help="切换到其他 OpenAI 兼容接口时修改此项，例如本地 Ollama。",
    )

    # 1. 首先尝试从本地加载已存储的密钥
    stored_f, stored_g = SecretManager.load_keys()
    # 2. 定义输入框，并将初始值设为加载到的密钥 (如果没有则为空)
    fireworks_key = st.text_input(
        "Fireworks API Key",
        value=stored_f or "",
        type="password",
    )
    google_key = st.text_input(
        "Google API Key",
        value=stored_g or "",
        type="password",
    )

    # 3. 密钥保存/清除操作区
    col_k1, col_k2 = st.columns(2)
    with col_k1:
        if st.button("💾 记忆密钥", use_container_width=True):
            if fireworks_key and google_key:
                SecretManager.save_keys(fireworks_key, google_key)
                st.success("已加密存储")
                st.rerun()
            else:
                st.error("请先输入 Key")
    with col_k2:
        if st.button("🗑️ 清除记忆", use_container_width=True):
            SecretManager.delete_keys()
            st.warning("已清除本地存储")
            st.rerun()
    
    st.divider()

    # ── Crawler settings ─────────────────────────────────────────────────────
    st.subheader("🕸️ 爬虫设置")
    
    firecrawl_url = st.text_input("本地 Firecrawl 地址", value="http://localhost:13002")
    sitemap_url = st.text_input("Sitemap URL", value="https://milvus.io")
    
    include_pattern = st.text_input("只爬取包含以下路径的网址 (逗号分隔)", value="/docs")
    exclude_pattern = st.text_input("排除爬取包含以下路径的网址 (逗号分隔)", value="")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
    # 爬取网址层级深度输入, 使用 value=None 允许清空输入框，placeholder 提示用户
        max_depth = st.number_input(
            "爬取目录深度", 
            min_value=0, 
            value=0, # 默认为0，不限制爬取深度
            help="URL 路径中 '/' 的最大数量，0 表示不限制深度"
        )
    with col_f2:
    # 同样处理最大爬取网页数量
        max_limit = st.number_input(
            "爬取网页数量", 
            min_value=0, 
            value=0, # 默认为0，不限制爬取数量
            help="限制单次爬取的 URL 总数量，0 表示不限制，如果高频抓取上万个页面可能会触发目标网站的防火墙导致封禁风险！"
        )

    # APP-FIX-5: pure 4-space indent (original line 143 had a mixed tab)
    start_index_ui = st.number_input(
        "起始索引（断点续爬）",
        value=0,
        help="上次任务结束时的网页索引，用于断点续爬。",
    )
    # st.subheader("🌐 语言偏好设置") #这个标题有点多余，不用显示了
    # 定义语言映射表：用户看到的名称 -> 匹配 URL 或检测代码的标识
    # 这里的 'zh' 将同时匹配 zh, zh-cn, zh-tw, zh-hk
    lang_options = {
        "中文(简/繁)": "zh",
        "英语":        "en",
        "日语":        "ja",
        "韩语":        "ko",
        "德语":        "de",
        "法语":        "fr",
        "西班牙语":    "es",
    }
    # 多选框：默认选中文和英文
    selected_langs = st.multiselect(
        "爬取以下语言的网页",
        options=list(lang_options.keys()),
        default=["中文(简/繁)", "英语"],
        help="留空则不进行语言过滤。",
    )
    # 转换为简码列表 [ 'zh', 'en' ]
    target_langs = [lang_options[name] for name in selected_langs]

    st.divider()
    st.subheader("⚡ 性能与数据库")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        batch_size = st.number_input(
            "批处理大小", 10, 500, value=200,
            help="每批提交到线程池的任务数，控制内存消耗。如果 25 并发线程运行稳定，可以把 BATCH_SIZE酌情提高。",
        )
        chunk_size = st.number_input(
            "切片大小", value=1000,
            help="块越大，Embedding 调用次数越少，速度越快。如很多网页被切出了 50 个以上的块，尝试调大到 1500 或 2000。",
        )
    with col_p2:
        max_threads = st.number_input(
            "并发线程数", min_value=1, max_value=30, value=10,
            help="数值越大速度越快，建议 5-15。",
        )
        chunk_overlap = st.number_input("切片重叠", value=200)

    collection_name = st.text_input("Milvus 集合名称", value="rag_docs")
    drop_old = st.checkbox(
        "启动时清空旧数据",
        value=False,
        help="仅在数据结构出错时使用，完成后务必取消勾选。",
    )

    # APP-FIX-2: Milvus connection is now configurable in the sidebar
    st.divider()
    st.subheader("🗄️ Milvus 连接")
    milvus_host = st.text_input("Milvus Host", value="127.0.0.1")
    milvus_port = st.text_input("Milvus Port", value="19530")

    # APP-FIX-2: Retrieval tuning params exposed to the user
    st.divider()
    st.subheader("🔍 检索参数")
    retrieval_k = st.number_input(
        "向量召回数量 (k)",
        min_value=1, max_value=50, value=10,
        help="向量库粗排阶段召回的文档数量。",
    )
    rerank_top_n = st.number_input(
        "精排保留数量 (top-n)",
        min_value=1, max_value=10, value=3,
        help="FlashRank 精排后保留并注入 prompt 的文档数，必须 ≤ 向量召回数量。",
    )


# --- 主界面 ---
st.title("📚 Universal RAG Engine")

if not fireworks_key or not google_key:
    st.warning("⚠️ 请先在侧边栏输入 API Key。")
    st.stop()

# 将 0 解释为“不限制”，传入 RAGConfig 时用 None 表示
# Build RAGConfig — APP-FIX-2: all new fields wired in
max_depth_cfg = None if max_depth == 0 else int(max_depth)
max_limit_cfg = None if max_limit == 0 else int(max_limit)

# Validate rerank_top_n ≤ retrieval_k before constructing config
if rerank_top_n > retrieval_k:
    st.sidebar.error(f"⚠️ 精排保留数量 ({rerank_top_n}) 不能大于向量召回数量 ({retrieval_k})，已自动修正。")
    rerank_top_n = int(retrieval_k)

# 初始化配置对象
config = RAGConfig(
    # Crawler
    sitemap_url=sitemap_url,
    include_patterns=[p.strip() for p in include_pattern.split(",") if p.strip()],
    exclude_patterns=[p.strip() for p in exclude_pattern.split(",") if p.strip()],
    firecrawl_url=firecrawl_url,
    max_depth=max_depth_cfg,      # 爬取网站目录层级深度
    max_limit=max_limit_cfg,      # 爬取网站的最大网页数
    target_languages=target_langs, # 待爬取网页的语言
    # Performance
    max_threads      = int(max_threads),
    chunk_size       = int(chunk_size),
    chunk_overlap    = int(chunk_overlap),
    batch_size       = int(batch_size),
    # Model
    embedding_model  = embedding_model_name,
    llm_base_url     = llm_base_url,            # APP-FIX-2
    # DB
    collection_name=collection_name,
    drop_old=drop_old,
    milvus_host      = milvus_host,             # APP-FIX-2
    milvus_port      = milvus_port,             # APP-FIX-2
    # Retrieval
    retrieval_k      = int(retrieval_k),        # APP-FIX-2
    rerank_top_n     = int(rerank_top_n),       # APP-FIX-2
    start_index      = int(start_index_ui),
)

api_keys = {"FIREWORKS_API_KEY": fireworks_key, "GOOGLE_API_KEY": google_key}

# --- 主界面 Tabs ---
# ---------------------------------------------------------------------------
# Helper: get-or-create engine (APP-FIX-1)
# The engine is stored in session_state so that the _cached_urls set by
# preview_sitemap() (FIX-12 in rag_core) is not thrown away when the user
# subsequently clicks "Start Build".
# ---------------------------------------------------------------------------
def _get_engine() -> UniversalRAGEngine:
    cfg_key = (
        str(sitemap_url), str(collection_name),
        str(milvus_host), str(milvus_port),
        int(chunk_size), int(chunk_overlap), bool(drop_old),
        str(include_pattern), str(exclude_pattern),   # ← ADD THESE
        tuple(sorted(target_langs)),                  # ← ADD THIS
    )
    if (
        st.session_state.engine is None
        or st.session_state.get("_engine_cfg_key") != cfg_key
    ):
        # Get the process-level cached embedding — never reloads
        emb = _get_cached_embeddings(embedding_model_name, "cpu")
        st.session_state.engine = UniversalRAGEngine(config, api_keys, embeddings=emb)
        st.session_state["_engine_cfg_key"] = cfg_key
    return st.session_state.engine

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2 = st.tabs(["🕷️ 知识库构建仪表盘", "💬 RAG 智能文档助理问答"])

# --- TAB 1: 知识库构建 ---
with tab1:
    # 仪表盘显示区
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    elapsed_ui = col_m1.empty()
    eta_ui = col_m2.empty()
    speed_ui = col_m3.empty()
    count_ui = col_m4.empty()
    
    main_bar = st.progress(0)
    
    # 控制按钮区
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        preview_click = st.button("👀 仅预览 URL", use_container_width=True)		
    with c2:
        start_click = st.button(
            "🚀 开始构建",
            use_container_width=True,
            disabled=(st.session_state.run_state == "running"),
        )
    with c3:
        pause_label = "▶️ 恢复" if st.session_state.run_state == "paused" else "⏸️ 暂停"
        pause_click = st.button(
            pause_label,
            use_container_width=True,
            disabled=(st.session_state.run_state == "idle"),
        )
    with c4:
        stop_click = st.button(
            "🛑 停止",
            use_container_width=True,
            disabled=(st.session_state.run_state == "idle"),
        )

    # 状态提示
    status_msg = st.empty()
    log_area = st.empty()

    # 处理按钮点击逻辑
    if start_click:
        st.session_state.run_state = "running"
        st.session_state.start_time_stamp = time.time()
        st.rerun()
    if pause_click:
        st.session_state.run_state = (
            "paused" if st.session_state.run_state == "running" else "running"
        )
        st.rerun()
    if stop_click:
        st.session_state.run_state = "stopped"
        st.rerun()

    # 1. 预览逻辑实现
    if preview_click:
        with st.spinner("正在解析 Sitemap…"):
            # APP-FIX-1: use session-scoped engine so the cache is preserved
            engine = _get_engine()
            urls, preview_logs = engine.preview_sitemap()
            for log_line in preview_logs:
                st.info(log_line)
            if urls:
                st.dataframe(pd.DataFrame(urls, columns=["待抓取列表"]), height=300)

    # 2. 构建核心循环
    if st.session_state.run_state == "running":
        # APP-FIX-1: reuse the same engine instance (preserves _cached_urls)
        engine = _get_engine()
        current_start = max(config.start_index, st.session_state.last_processed_idx)
        
        generator = engine.run_generator(start_index=current_start)
        
        build_failed = False  # 标记本次构建是否失败
        
        try:
            for event in generator:
                # 实时检查外部控制状态
                if st.session_state.run_state == "paused":
                    status_msg.warning("⏸️ 抓取已暂停")
                    st.stop()
                if st.session_state.run_state == "stopped":
                    status_msg.error("🛑 抓取已终止")
                    st.session_state.run_state = "idle"
                    st.session_state.last_processed_idx = 0 
                    st.stop()

                if event["type"] == "progress":
                    st.session_state.last_processed_idx = event["completed"]
                    
                    # 时间与速度计算
                    now = time.time()
                    elapsed = now - st.session_state.start_time_stamp
                    processed_count = event["completed"] - current_start
                    speed_min       = (processed_count / (elapsed / 60)) if elapsed > 0 else 0
                    remaining       = event["total"] - event["completed"]
                    eta_sec         = (
                        (remaining / (processed_count / elapsed))
                        if processed_count > 0
                        else 0
                    )

                    elapsed_ui.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-label">已用时间</div>'
                        f'<div class="metric-value">{int(elapsed)}s</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    eta_ui.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-label">预计剩余</div>'
                        f'<div class="metric-value">{str(timedelta(seconds=int(eta_sec)))}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    speed_ui.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-label">抓取速度</div>'
                        f'<div class="metric-value">{speed_min:.1f} P/m</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    count_ui.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-label">当前进度</div>'
                        f'<div class="metric-value">'
                        f'{event["completed"]}/{event["total"]}'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )

                    progress_val = min(event["completed"] / max(event["total"], 1), 1.0)
                    main_bar.progress(progress_val)

                    log_text = f"[{event['status']}] {event['url']} → {event['detail']}"
                    st.session_state.logs.append(log_text)
                
                elif event["type"] == "log":
                    st.session_state.logs.append(f"ℹ️ {event['message']}")

                # 渲染日志区 (限制显示最后100条以保证性能)
                display_logs = st.session_state.logs[-100:]
                log_html = "".join(
                    [f'<div class="log-entry">{line}</div>' for line in reversed(display_logs)]
                )
                log_area.markdown(
                    f'<div class="log-container">{log_html}</div>',
                    unsafe_allow_html=True,
                )

        except Exception as exc:
            build_failed = True
            st.session_state.run_state = "idle"
            st.error(f"构建过程中发生错误: {exc}")
        
        # 只有在未失败时才显示成功提示
        if not build_failed:
            # 抓取成功后，立刻将侧边栏填写的名字同步给“当前激活集合”
            st.session_state.current_coll_name = collection_name
            st.session_state["_coll_refresh_needed"] = True   # trigger collection list refresh in Tab 2
            st.balloons()
            st.success(f"✅ 知识库构建完成！集合 '{collection_name}' 已就绪。")
            st.info("💡 您现在可以切换到『智能问答』标签页开始对话了，系统已为您自动选中该库。")
            st.session_state.run_state = "idle"

# --- Tab 2: 聊天界面 ---
with tab2:
    #st.subheader("RAG 智能文档助理对话") #占web界面的空间，去掉
    
    # 1. 顶部增加集合选择区
    #st.markdown("### 🔍 知识库选择")
    col_select, col_refresh = st.columns([4, 1])
    # 获取集合信息
    # 使用 st.spinner 包裹，防止加载时界面卡死
    # only fetches when needed
    # Only query Milvus when: first load, user clicks refresh, or after a build completes
    _should_refresh = (
        "all_collections_cache" not in st.session_state
        or st.session_state.get("_coll_refresh_needed", False)
    )
    if _should_refresh:
        with st.spinner("正在同步知识库列表…"):
            st.session_state.all_collections_cache = RAGChatBot.get_milvus_collections_info(
                milvus_host=milvus_host,
                milvus_port=milvus_port,
            )
        st.session_state["_coll_refresh_needed"] = False

    all_collections = st.session_state.get("all_collections_cache", [])

    if not all_collections:
        st.warning("⚠️ 未检测到有效知识库，请先前往『知识库构建』页面抓取数据。")
        st.stop()

    # Build option list and name map
    options  = [f"{c['name']} ({c['count']} 条数据)" for c in all_collections]
    name_map = {f"{c['name']} ({c['count']} 条数据)": c["name"] for c in all_collections}

    # Auto-select logic: sidebar input > session state > first in list
    sidebar_input      = collection_name.strip()
    all_existing_names = [c["name"] for c in all_collections]

    if sidebar_input in all_existing_names:
        current_target = sidebar_input
    elif st.session_state.get("current_coll_name") in all_existing_names:
        current_target = st.session_state["current_coll_name"]
    else:
        current_target = all_existing_names[0]

    default_idx = next(
        (i for i, opt in enumerate(options) if name_map[opt] == current_target),
        0,
    )

    with col_select:
        selected_option = st.selectbox(
            "选择要对话的知识库：",
            options=options,
            index=default_idx,
            key="chat_coll_select_final",
            help="系统已自动选择您最近构建或使用的知识库。",
        )
    with col_refresh:
        if st.button("🔄 刷新列表", use_container_width=True):
            st.session_state["_coll_refresh_needed"] = True
            st.rerun()

    # 2. 核心逻辑：检测切换并初始化 ChatBot
    if selected_option and all_collections:
        target_coll_name = name_map[selected_option]
        
        # 如果当前没有 chatbot，或者用户手动切换了集合名
        if "chatbot" not in st.session_state or st.session_state.get("current_coll_name") != target_coll_name:
            if fireworks_key and google_key:
                with st.spinner(f"正在切换至知识库: {target_coll_name}…"):
                    # APP-FIX-4: pass all new params to RAGChatBot
                    st.session_state.chatbot = RAGChatBot(
                        fireworks_key=fireworks_key,
                        google_key=google_key,
                        collection_name=target_coll_name,
                        llm_model=llm_model_name,
                        llm_base_url=llm_base_url,
                        milvus_host=milvus_host,
                        milvus_port=milvus_port,
                        retrieval_k=int(retrieval_k),
                        rerank_top_n=int(rerank_top_n),
                        embeddings=_get_cached_embeddings(embedding_model_name, "cpu"),  # ← pass in
                    )
                    st.session_state.current_coll_name = target_coll_name
                    # 切换知识库时清空聊天记录，防止上下文污染
                    st.session_state.chat_messages = [] 
                    st.session_state.thread_id = f"session_{int(time.time())}"
                    st.toast(f"已连接到 {target_coll_name}", icon="✅")
            else:
                st.warning("请先在侧边栏配置 API Key。")

    # Chat history display
    chat_history_container = st.container()
    
    with chat_history_container:
        # 显示历史消息
        for m in st.session_state.chat_messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

    if "chatbot" not in st.session_state:
        st.info("请先在上方选择知识库并完成初始化。")
        st.stop()

    # 关键：将 chat_input 放在 Tab 的最外层，不要包裹在任何 col 或 sub-container 里
    # Streamlit 会自动将其锚定在浏览器底部
    if prompt := st.chat_input("基于当前选择的知识库，输入您的问题..."):
        
        # 1. 立即显示用户输入
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        # 刷新页面以显示新消息，chat_input 会自动留在底部
        with chat_history_container:
            with st.chat_message("user"):
                st.markdown(prompt)

        # 2. 生成 AI 回复
        with chat_history_container:
            with st.chat_message("assistant"):
                with st.spinner("正在检索文档并思考..."):
                    try:
                        # 构造图输入
                        inputs = {"messages": [HumanMessage(content=prompt)]}
                        cfg = {"configurable": {"thread_id": st.session_state.thread_id}}
                        
                        # 运行 LangGraph
                        final_state = st.session_state.chatbot.graph.invoke(inputs, config=cfg)
                        answer = final_state["messages"][-1].content
                        
                        # 显示回答
                        st.markdown(answer)
                        
                        # 存入历史
                        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
                        
                        # 只有在需要显示 expander 时才显示
                        if "user_profile" in final_state:
                            with st.expander("🔍 查看检索上下文 & 用户画像"):
                                st.write("**当前用户画像：**", final_state.get("user_profile", {}))
                                st.write("**对话摘要：**", final_state.get("summary", "尚无摘要"))

                                retrieval_results = final_state.get("retrieval_results", [])
                                if retrieval_results:
                                    st.markdown(
                                        f"**本轮检索命中的文档（Top {len(retrieval_results)}）**"
                                    )
                                    for idx, item in enumerate(retrieval_results, start=1):
                                        src = item.get("source", "未知来源")
                                        score = item.get("score", None)
                                        snippet = item.get("snippet", "")

                                        score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "N/A"
                                        st.markdown(f"- **{idx}. 来源**: {src}  \n  **相似度**: {score_str}")
                                        st.markdown(f"  **片段预览**: {snippet}")
                                else:
                                    st.info("本轮未检索到可用文档，或检索失败。")

                    except Exception as exc:
                        st.error(f"对话发生错误: {exc}")

        # 这一段会在对话完成后运行，强制浏览器滚动到页面底部
        # --- 增强版：强制自动置底的 JavaScript 补丁 ---
        st.components.v1.html(
            """
            <script>
                (function() {
                    const scroll = () => {
                        const main = window.parent.document.querySelector('section.main');
                        if (main) main.scrollTo({top: main.scrollHeight, behavior: 'smooth'});
                    };
                    setTimeout(scroll, 300); // 初始滚动
                    setTimeout(scroll, 1000); // 补丁：处理长文本渲染后的高度变化
                })();
            </script>
            """,
            height=0
        )
