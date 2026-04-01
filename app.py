import time
import logging
from datetime import timedelta, date
from urllib.parse import urlparse
import pandas as pd
import streamlit as st
from langchain_core.messages import HumanMessage
from pymilvus import connections, utility
import rag_settings
from rag_core import RAGConfig, UniversalRAGEngine, RAGChatBot, SecretManager

logger_app = logging.getLogger("app")

@st.cache_resource
def _get_cached_embeddings(model_name: str, google_key: str):
    """Process-level cache for the embedding model.
    @st.cache_resource survives reruns, tab switches and session resets.
    The old module-level _embedding_singleton only survived within one import."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    logger_app.info("st.cache_resource: loading Google embedding model (once per process)...")
    return GoogleGenerativeAIEmbeddings(
        model=model_name,
        google_api_key=google_key,
        task_type="retrieval_document",
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
        padding-bottom: 5rem !important; /* just enough to clear the sticky input bar */
        max-width: 90% !important;
    }
    header { visibility: hidden; } /* 隐藏顶部装饰条 */

    /* B. 标题样式 */
    h1 { margin-top: -10px !important; padding-bottom: 15px; font-size: 2.2rem !important; }

    /* C. 按钮对齐 */
    div[data-testid="stButton"] button {
        height: 38px;
        border-radius: 8px;
        margin-top: 0 !important;
    }
    /* Align selectbox (label-collapsed) with adjacent buttons */
    div[data-testid="stSelectbox"] {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }

    /* E. Tighten vertical gaps between all elements */
    [data-testid="stVerticalBlock"] {
        gap: 0.3rem !important;
    }
    [data-testid="stVerticalBlock"] > div:has([data-testid="stColumns"]) {
        margin-bottom: 0 !important;
    }
    .element-container {
        margin-bottom: 0 !important;
    }

    /* D. Chat input — stBottom positioning handled by JS in the chat section */
    [data-testid="stChatInput"] {
        background-color: transparent !important;
        padding-bottom: 0 !important;
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
        display: flex;
        flex-direction: column-reverse;
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
    "preview_logs":       [],    # ← ADD: persists preview output across reruns
    "preview_urls":       [],
    "last_progress":      0.0,
    "last_elapsed_s":     None,
    "last_eta_s":         None,
    "last_speed_pm":      None,
    "last_count_str":     None,
    "_build_done_coll":   None,  # set after build completes, cleared after display
    "active_section":     "build",  # "build" | "chat"
}
for key, default in _STATE_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Load persisted settings (falls back to defaults for any missing key)
if "rag_cfg" not in st.session_state:
    st.session_state.rag_cfg = rag_settings.load()

_S = st.session_state.rag_cfg   # short alias used below

def _auto_collection_name(sitemap_url: str, existing_colls: list) -> tuple:
    """Return (domain_prefix, suggested_new_name).
    Format: Domain@DD-Mon-YYYY  e.g. Milvus@30-Mar-2026"""
    hostname = urlparse(sitemap_url).hostname or "rag"
    domain = hostname.split(".")[0].capitalize()
    new_name = f"{domain}_{date.today().strftime('%d%b%Y')}"
    return domain, new_name


def _best_existing(domain_prefix: str, existing_colls: list) -> str | None:
    """Return the name of the latest existing collection whose name starts
    with '<domain_prefix>_', or None if no match."""
    from datetime import datetime
    prefix = f"{domain_prefix}_"
    matches = [c["name"] for c in existing_colls if c["name"].startswith(prefix)]
    if not matches:
        return None
    def _parse(name):
        try:
            return datetime.strptime(name[len(prefix):], "%d%b%Y")
        except ValueError:
            return datetime.min
    return max(matches, key=_parse)


# --- 侧边栏：参数配置 ---
with st.sidebar:
    st.header("⚙️ 系统配置")

    # Disable all sidebar controls while a build is running/paused OR while in chat section
    is_building = (
        st.session_state.run_state in ("running", "paused")
        or st.session_state.active_section == "chat"
    )
    if st.session_state.run_state in ("running", "paused"):
        st.info("⏳ 构建进行中，参数已锁定。")
    elif st.session_state.active_section == "chat":
        st.info("💬 问答模式中，参数已锁定。")

    st.subheader("🤖 模型配置（可选）")
    embedding_model_name = st.text_input(
        "Embedding 模型",
<<<<<<< HEAD
        value="models/gemini-embedding-001",
        help="Embedding 模型建议固定不频繁切换；切换后建议重启 Streamlit 进程释放旧模型内存, 且须勾选'清空旧数据'并重新入库。",
=======
        value=_S["embedding_model_name"],
        help="Embedding 模型建议固定不频繁切换；切换后建议重启 Streamlit 进程释放旧模型内存, 且须勾选'清空旧数据'并重新入库。",
        disabled=is_building,
>>>>>>> c871ba62308154c6755c3fe58b34b0adb98ed810
    )
    llm_model_name = st.text_input(
        "对话LLM模型 ID",
        value=_S["llm_model_name"],
        help="高级用法：填写 Fireworks 上对应的模型 ID，不懂可保持默认。",
        disabled=is_building,
    )
    llm_base_url = st.text_input(
        "对话LLM模型 Base URL",
        value=_S["llm_base_url"],
        help="切换到其他 OpenAI 兼容接口时修改此项，例如本地 Ollama。",
        disabled=is_building,
    )

    stored_f, stored_g = SecretManager.load_keys()
    fireworks_key = st.text_input(
        "Fireworks API Key",
        value=stored_f or "",
        type="password",
        disabled=is_building,
    )
    google_key = st.text_input(
        "Google API Key",
        value=stored_g or "",
        type="password",
        disabled=is_building,
    )

    col_k1, col_k2 = st.columns(2)
    with col_k1:
        if st.button("💾 记忆密钥", use_container_width=True, disabled=is_building):
            if fireworks_key and google_key:
                SecretManager.save_keys(fireworks_key, google_key)
                st.success("已加密存储")
                st.rerun()
            else:
                st.error("请先输入 Key")
    with col_k2:
        if st.button("🗑️ 清除记忆", use_container_width=True, disabled=is_building):
            SecretManager.delete_keys()
            st.warning("已清除本地存储")
            st.rerun()

    st.divider()

    # ── Crawler settings ──────────────────────────────────────────────────────
    st.subheader("🕸️ 爬虫设置")

    firecrawl_url = st.text_input(
        "Firecrawl 地址", value=_S["firecrawl_url"], disabled=is_building,
    )
    sitemap_url = st.text_input(
        "Sitemap URL", value=_S["sitemap_url"], disabled=is_building,
    )

    include_pattern = st.text_input(
        "只爬取包含以下路径的网址 (逗号分隔)", value=_S["include_pattern"], disabled=is_building,
    )
    exclude_pattern = st.text_input(
        "排除爬取包含以下路径的网址 (逗号分隔)", value=_S["exclude_pattern"], disabled=is_building,
    )

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        max_depth = st.number_input(
            "爬取目录深度",
            min_value=0, value=_S["max_depth"],
            help="URL 路径中 '/' 的最大数量，0 表示不限制深度",
            disabled=is_building,
        )
    with col_f2:
        max_limit = st.number_input(
            "爬取网页数量",
            min_value=0, value=_S["max_limit"],
            help="限制单次爬取的 URL 总数量，0 表示不限制，如果高频抓取上万个页面可能会触发目标网站的防火墙导致封禁风险！",
            disabled=is_building,
        )

    start_index_ui = st.number_input(
        "起始索引（断点续爬）",
        value=_S["start_index_ui"],
        help="上次任务结束时的网页索引，用于断点续爬。",
        disabled=is_building,
    )

    lang_options = {
        "中文(简/繁)": "zh",
        "英语":        "en",
        "日语":        "ja",
        "韩语":        "ko",
        "德语":        "de",
        "法语":        "fr",
        "西班牙语":    "es",
    }
    selected_langs = st.multiselect(
        "爬取以下语言的网页",
        options=list(lang_options.keys()),
        default=_S["selected_langs"],
        help="留空则不进行语言过滤。",
        disabled=is_building,
    )
    target_langs = [lang_options[name] for name in selected_langs]

    st.divider()

    # ── Milvus connection (must come before collection name so we can query DB) ─
    st.subheader("🗄️ Milvus 连接")
    milvus_host = st.text_input("Milvus Host", value=_S["milvus_host"], disabled=is_building)
    milvus_port = st.text_input("Milvus Port", value=_S["milvus_port"], disabled=is_building)

    # Populate collection cache now that we have host/port
    if "all_collections_cache" not in st.session_state:
        try:
            st.session_state.all_collections_cache = RAGChatBot.get_milvus_collections_info(
                milvus_host=milvus_host, milvus_port=milvus_port,
            )
        except Exception:
            st.session_state.all_collections_cache = []

    st.divider()

    # ── Performance & DB ──────────────────────────────────────────────────────
    st.subheader("⚡ 性能与数据库")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        batch_size = st.number_input(
            "批处理大小", 10, 500, value=_S["batch_size"],
            help="每批提交到线程池的任务数，控制内存消耗。如果 25 并发线程运行稳定，可以把 BATCH_SIZE酌情提高。",
            disabled=is_building,
        )
        chunk_size = st.number_input(
            "切片大小", value=_S["chunk_size"],
            help="块越大，Embedding 调用次数越少，速度越快。如很多网页被切出了 50 个以上的块，尝试调大到 1500 或 2000。",
            disabled=is_building,
        )
    with col_p2:
        max_threads = st.number_input(
            "并发线程数", min_value=1, max_value=30, value=_S["max_threads"],
            help="数值越大速度越快，建议 5-15。",
            disabled=is_building,
        )
        chunk_overlap = st.number_input("切片重叠", value=_S["chunk_overlap"], disabled=is_building)

    # ── Collection name: auto-suggest or pick existing ────────────────────────
    _existing_colls = st.session_state.get("all_collections_cache", [])
    _domain, _new_name = _auto_collection_name(sitemap_url, _existing_colls)
    _matched = _best_existing(_domain, _existing_colls)

    _NEW_OPTION = "🆕 创建新知识库"
    _existing_names = [c["name"] for c in _existing_colls]
    _dropdown_opts = _existing_names + ([_NEW_OPTION] if _NEW_OPTION not in _existing_names else [])

    # Default selection: matched existing (if any), else "创建新知识库"
    _default_sel = _matched if _matched else _NEW_OPTION
    _default_idx = _dropdown_opts.index(_default_sel) if _default_sel in _dropdown_opts else len(_dropdown_opts) - 1

    _selected = st.selectbox(
        "知识库选择",
        options=_dropdown_opts,
        index=_default_idx,
        help="自动匹配已有同域名知识库；选「🆕 创建新知识库」可新建。",
        disabled=is_building,
    )

    if _selected == _NEW_OPTION:
        collection_name = st.text_input(
            "新知识库名称（可修改）",
            value=_new_name,
            help="格式：域名@日期，仅限字母/数字/下划线，可在此修改。",
            disabled=is_building,
        )
    else:
        collection_name = _selected

    drop_old = st.checkbox(
        "启动时清空旧数据",
        value=_S["drop_old"],
        help="仅在数据结构出错时使用，完成后务必取消勾选。",
        disabled=is_building,
    )

    # ── Retrieval params ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("🔍 检索参数")
    retrieval_k = st.number_input(
        "向量数据库初查结果数量 (k)",
        min_value=1, max_value=50, value=_S["retrieval_k"],
        help="向量库粗排阶段召回的文档数量。",
        disabled=is_building,
    )
    rerank_top_n = st.number_input(
        "精确排序后保留数量 (top-n)",
        min_value=1, max_value=10, value=_S["rerank_top_n"],
        help="FlashRank 精确排序后保留并注入 prompt 的文档数，必须 ≤ 向量召回数量。",
        disabled=is_building,
    )



# Auto-save whenever sidebar values differ from stored.


# Auto-save settings when the user changes any sidebar value.
# Comparison is done by value so this never saves unnecessarily.
_current_sidebar_values = rag_settings.current_values_from_sidebar(
    embedding_model_name = embedding_model_name,
    llm_model_name       = llm_model_name,
    llm_base_url         = llm_base_url,
    firecrawl_url        = firecrawl_url,
    sitemap_url          = sitemap_url,
    include_pattern      = include_pattern,
    exclude_pattern      = exclude_pattern,
    max_depth            = max_depth,
    max_limit            = max_limit,
    start_index_ui       = start_index_ui,
    selected_langs       = selected_langs,
    milvus_host          = milvus_host,
    milvus_port          = milvus_port,
    batch_size           = batch_size,
    chunk_size           = chunk_size,
    max_threads          = max_threads,
    chunk_overlap        = chunk_overlap,
    drop_old             = drop_old,
    retrieval_k          = retrieval_k,
    rerank_top_n         = rerank_top_n,
)

if _current_sidebar_values != st.session_state.rag_cfg:
    # Values changed — persist immediately and update the in-session cache
    rag_settings.save(_current_sidebar_values)
    st.session_state.rag_cfg = _current_sidebar_values


# --- 主界面 ---
st.title("📚 Universal RAG Engine")

if not fireworks_key or not google_key:
    st.warning("⚠️ 请先在侧边栏输入 API Key。")
    st.stop()

# 将 0 解释为"不限制"，传入 RAGConfig 时用 None 表示
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
        str(include_pattern), str(exclude_pattern),   
        tuple(sorted(target_langs)),   
        str(embedding_model_name),    
    )
    if (
        st.session_state.engine is None
        or st.session_state.get("_engine_cfg_key") != cfg_key
    ):
        # Get the process-level cached embedding — never reloads
        emb = _get_cached_embeddings(embedding_model_name, google_key)
        st.session_state.engine = UniversalRAGEngine(config, api_keys, embeddings=emb)
        st.session_state["_engine_cfg_key"] = cfg_key
    return st.session_state.engine

# ---------------------------------------------------------------------------
# Navigation (replaces st.tabs so active section is always trackable)
# ---------------------------------------------------------------------------
_nav_col1, _nav_col2 = st.columns(2)
with _nav_col1:
    if st.button(
        "🕷️ 知识库构建仪表盘",
        use_container_width=True,
        type="primary" if st.session_state.active_section == "build" else "secondary",
    ):
        st.session_state.active_section = "build"
        st.rerun()
with _nav_col2:
    if st.button(
        "💬 RAG 智能文档助理问答",
        use_container_width=True,
        type="primary" if st.session_state.active_section == "chat" else "secondary",
    ):
        st.session_state.active_section = "chat"
        st.rerun()

st.markdown('<hr style="margin:4px 0 6px 0; border:none; border-top:1px solid #dee2e6;">', unsafe_allow_html=True)

# --- SECTION 1: 知识库构建 ---
if st.session_state.active_section == "build":
    # Show build-complete banner if we just finished (persisted across rerun)
    if st.session_state._build_done_coll:
        st.balloons()
        st.success(f"✅ 知识库构建完成！集合 '{st.session_state._build_done_coll}' 已就绪。")
        st.info("💡 您现在可以切换到『智能问答』标签页开始对话了，系统已为您自动选中该库。")
        st.session_state._build_done_coll = None

    # 仪表盘显示区
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    elapsed_ui = col_m1.empty()
    eta_ui = col_m2.empty()
    speed_ui = col_m3.empty()
    count_ui = col_m4.empty()
    
    main_bar = st.progress(st.session_state.last_progress)
    
    # 控制按钮区
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        preview_click = st.button(
            "👀 仅预览 URL", 
            use_container_width=True,
            disabled=(st.session_state.run_state in ("running", "paused")),
        )		
    with c2:
        start_click = st.button(
            "🚀 开始构建",
            use_container_width=True,
            disabled=(st.session_state.run_state in ("running", "paused")),
        )
    with c3:
        pause_label = "▶️ 恢复" if st.session_state.run_state == "paused" else "⏸️ 暂停"
        #only enabled when a build is actually active (running or paused):
        pause_click = st.button(
            pause_label,
            use_container_width=True,
            disabled=(st.session_state.run_state not in ("running", "paused")),
        )
    with c4:
        #only enabled when a build is actually active (running or paused):
        stop_click = st.button(
            "🛑 停止",
            use_container_width=True,
            disabled=(st.session_state.run_state not in ("running", "paused")),
        )

    # 状态提示
    #status_msg = st.empty()
    log_area   = st.empty()   # ← placeholder, can be overwritten mid-loop
    preview_table_area = st.empty()
    
    # Render any logs already in session_state (shows results after pause/stop/rerun)
    def _render_log():
        if st.session_state.logs:
            display_logs = st.session_state.logs[-100:]
            log_html = "".join(
                [f'<div class="log-entry">{line}</div>' for line in display_logs]
            )
            log_area.markdown(
                f'<div class="log-container">{log_html}</div>',
                unsafe_allow_html=True,
            )
        if st.session_state.get("preview_urls"):
            preview_table_area.dataframe(
                pd.DataFrame(st.session_state.preview_urls, columns=["待抓取列表"]),
                height=300,
            )

    _render_log()   # renders previous-run logs immediately on page load

    def _render_metrics():
        if st.session_state.last_elapsed_s is not None:
            elapsed_ui.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">已用时间</div>'
                f'<div class="metric-value">{str(timedelta(seconds=st.session_state.last_elapsed_s))}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        if st.session_state.last_eta_s is not None:
            eta_ui.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">预计剩余</div>'
                f'<div class="metric-value">{str(timedelta(seconds=st.session_state.last_eta_s))}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        if st.session_state.last_speed_pm is not None:
            speed_ui.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">抓取速度</div>'
                f'<div class="metric-value">{st.session_state.last_speed_pm:.1f} P/m</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        if st.session_state.last_count_str is not None:
            count_ui.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">完成进度</div>'
                f'<div class="metric-value">{st.session_state.last_count_str}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    _render_metrics()  # restore last values on every rerun (after pause/stop)

    # 处理按钮点击逻辑
    if start_click:
        st.session_state.run_state        = "running"
        st.session_state.start_time_stamp = time.time()
        st.session_state.last_progress    = 0.0
        st.session_state.last_elapsed_s   = None
        st.session_state.last_eta_s       = None
        st.session_state.last_speed_pm    = None
        st.session_state.last_count_str   = None
        st.rerun()
    if pause_click:
        st.session_state.run_state = (
            "paused" if st.session_state.run_state == "running" else "running"
        )
        st.rerun()
    if stop_click: # reset immediately so buttons disable on the very next rerun:
        st.session_state.run_state        = "stopped"   # loop break signal
        st.session_state.last_progress    = 0.0         # reset progress bar
        st.session_state.last_processed_idx = 0
        st.rerun()

    # 1. 预览逻辑实现
    if preview_click:
        st.session_state.logs = []           # clear old build logs
        st.session_state.preview_urls = []   # clear old preview table   
        with st.spinner("正在解析 Sitemap…"):
            # APP-FIX-1: use session-scoped engine so the cache is preserved
            engine = _get_engine()
            urls, preview_logs = engine.preview_sitemap()
            for log_line in preview_logs:
                st.session_state.logs.append(f"ℹ️ {log_line}")
            st.session_state.preview_urls = urls
        _render_log()   # ← render immediately in the SAME run, no second click needed
            #if urls:
            #    st.dataframe(pd.DataFrame(urls, columns=["待抓取列表"]), height=300)

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
                    st.session_state.logs.append("⏸️ 抓取已暂停，点击▶️恢复继续。")
                    break   # exit the for-event loop cleanly, rerun will show the log
                if st.session_state.run_state == "stopped":
                    st.session_state.logs.append("🛑 抓取已手动终止。")
                    st.session_state.run_state = "idle"
                    st.session_state.last_processed_idx = 0 
                    break   # exit the for-event loop cleanly

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

                    st.session_state.last_elapsed_s  = int(elapsed)
                    st.session_state.last_eta_s      = int(eta_sec)
                    st.session_state.last_speed_pm   = speed_min
                    st.session_state.last_count_str  = f'{event["completed"]}/{event["total"]}'
                    _render_metrics()

                    progress_val = min(event["completed"] / max(event["total"], 1), 1.0)
                    st.session_state.last_progress = progress_val
                    main_bar.progress(progress_val)

                    log_text = f"[{event['status']}] {event['url']} → {event['detail']}"
                    st.session_state.logs.append(log_text)
                
                elif event["type"] == "log":
                    st.session_state.logs.append(f"ℹ️ {event['message']}")

                # 渲染日志区 (限制显示最后100条以保证性能)
                display_logs = st.session_state.logs[-100:]
                log_html = "".join(
                    [f'<div class="log-entry">{line}</div>' for line in display_logs]
                )
                log_area.markdown(
                    f'<div class="log-container">{log_html}</div>',
                    unsafe_allow_html=True,
                )

        except Exception as exc:
            build_failed = True
            st.session_state.run_state = "idle"
            st.session_state.last_progress = 0.0 # After successful build, reset progress for next run:
            st.error(f"构建过程中发生错误: {exc}")
        
        # 只有在未失败时才显示成功提示
        if not build_failed:
            # 抓取成功后，立刻将侧边栏填写的名字同步给"当前激活集合"
            st.session_state.current_coll_name  = collection_name
            st.session_state["_coll_refresh_needed"] = True
            st.session_state._build_done_coll   = collection_name  # display banner on next rerun
            st.session_state.run_state          = "idle"
            st.rerun()  # re-render buttons to idle state, then banner is shown at top of tab1

# --- SECTION 2: 聊天界面 ---
elif st.session_state.active_section == "chat":
    # Pin the chat input to the true viewport bottom, aligned to the block-container
    st.components.v1.html("""
    <script>
    (function() {
        function pinChatInput() {
            var doc = window.parent.document;
            var stBottom = doc.querySelector('[data-testid="stBottom"]');
            var container = doc.querySelector('.block-container');
            if (!stBottom || !container) return;
            var rect = container.getBoundingClientRect();
            stBottom.style.position      = 'fixed';
            stBottom.style.bottom        = '0';
            stBottom.style.left          = rect.left + 'px';
            stBottom.style.width         = rect.width + 'px';
            stBottom.style.right         = '';
            stBottom.style.zIndex        = '1000';
            stBottom.style.background    = 'white';
            stBottom.style.paddingBottom = '0.75rem';
        }
        pinChatInput();
        setTimeout(pinChatInput, 300);
        window.parent.addEventListener('resize', pinChatInput);
    })();
    </script>
    """, height=0)

    # 1. 顶部增加集合选择区
    col_label, col_select, col_refresh, col_delete = st.columns([1.6, 4.4, 1.5, 1.5])
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

    _confirming = bool(st.session_state.get("_confirm_delete_coll"))

    with col_label:
        st.markdown(
            '<div style="padding-top:8px; font-size:0.95rem; white-space:nowrap;">'
            '选择要对话的知识库：</div>',
            unsafe_allow_html=True,
        )
    with col_select:
        selected_option = st.selectbox(
            "选择要对话的知识库：",
            options=options,
            index=default_idx,
            key="chat_coll_select_final",
            help="系统已自动选择您最近构建或使用的知识库。",
            label_visibility="collapsed",
            disabled=_confirming,
        )
    with col_refresh:
        if st.button("🔄 刷新列表", use_container_width=True, disabled=_confirming):
            st.session_state["_coll_refresh_needed"] = True
            st.rerun()
    with col_delete:
        if st.button("删除知识库", use_container_width=True, type="secondary", disabled=_confirming):
            st.session_state["_confirm_delete_coll"] = name_map[selected_option]
            st.rerun()

    # Confirm-delete banner (appears below the toolbar row)
    if st.session_state.get("_confirm_delete_coll"):
        _del_name = st.session_state["_confirm_delete_coll"]
        st.warning(f"⚠️ 确认删除知识库 **{_del_name}**？此操作不可撤销。")
        _dc1, _dc2, _ = st.columns([1.8, 1.8, 6.4])
        with _dc1:
            if st.button("✅ 确认删除", type="primary", use_container_width=True):
                try:
                    connections.connect(host=milvus_host, port=milvus_port)
                    utility.drop_collection(_del_name)
                    st.session_state["_coll_refresh_needed"] = True
                    st.session_state.pop("_confirm_delete_coll", None)
                    if st.session_state.get("current_coll_name") == _del_name:
                        st.session_state.pop("chatbot", None)
                        st.session_state["current_coll_name"] = None
                    st.success(f"✅ 知识库 '{_del_name}' 已删除。")
                    st.rerun()
                except Exception as _e:
                    st.error(f"删除失败: {_e}")
        with _dc2:
            if st.button("❌ 取消", use_container_width=True):
                st.session_state.pop("_confirm_delete_coll", None)
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
                        embeddings=_get_cached_embeddings(embedding_model_name, google_key),  # ← pass in
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
    if prompt := st.chat_input("基于当前选择的知识库，输入您的问题...", disabled=_confirming):
        
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
