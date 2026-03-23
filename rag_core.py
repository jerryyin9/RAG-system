import concurrent.futures
import logging
import hashlib
import io
import json
import os
import random
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse
from typing import Annotated, TypedDict, List, Optional

import bs4
import requests

from cryptography.fernet import Fernet

# LangChain 核心组件
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_milvus import Milvus
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, RemoveMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from pymilvus import connections, utility, Collection
from langdetect import detect_langs

# 直接导入 FlashRank 原始库（避开 problematic 的 langchain.retrievers）
from flashrank import Ranker, RerankRequest

# ---------------------------------------------------------------------------
# FIX-05: Structured logging (replaces all print() calls)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("langchain_milvus.vectorstores.milvus").setLevel(logging.ERROR) #Do not let AsyncMilvusClient WARNING floods the terminal window
logger = logging.getLogger(__name__)

#模拟网页浏览器，以免网站回复403错误，导致爬取失败
# HTTP helper
def get_headers(languages: List[str] = None) -> dict:
    """Build browser-like request headers, optionally locked to target languages."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/115.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }
    
    # 如果传入了语言列表，动态构建 Accept-Language 和 Cookie
    if languages:
        # 构造类似: "zh-CN,zh;q=0.9,en;q=0.8"
        lang_str = ",".join(
            [f"{lang};q={round(1.0 - i * 0.1, 1)}" for i, lang in enumerate(languages)]
        )
        headers["Accept-Language"] = lang_str
        # 针对 Milvus 等 Netlify 部署站点的语言锁定
        headers["Cookie"] = f"nf_lang={languages[0]}"
        
    return headers

#用于本地加密存储API key
# ---------------------------------------------------------------------------
# FIX-02: SecretManager – store key files under ~/.rag_secrets/
# ---------------------------------------------------------------------------
class SecretManager:
    # FIX-02: store keys next to the script file so the path is stable
    # across all launch directories and survives Streamlit hot-reloads.
    # Path.home() was removed because it caused a one-time migration break
    # and chmod(0o600) can raise on Windows before the key is returned.
    _BASE_DIR = Path(__file__).parent
    KEY_FILE  = _BASE_DIR / ".key.secret"
    DATA_FILE = _BASE_DIR / ".keys.enc"

    @classmethod
    def _ensure_dir(cls):
        # mode= is silently ignored on Windows; exist_ok avoids race
        cls._BASE_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_or_create_key(cls) -> bytes:
        cls._ensure_dir()
        if cls.KEY_FILE.exists():
            return cls.KEY_FILE.read_bytes()
        key = Fernet.generate_key()
        cls.KEY_FILE.write_bytes(key)
        try:   # chmod is a no-op / may raise on Windows – safe to ignore
            cls.KEY_FILE.chmod(0o600)
        except Exception:
            pass
        return key

    @classmethod
    def save_keys(cls, f_key: str, g_key: str):
        cls._ensure_dir()
        fernet = Fernet(cls.get_or_create_key())
        data = json.dumps({"f": f_key, "g": g_key}).encode()
        cls.DATA_FILE.write_bytes(fernet.encrypt(data))
        try:   # chmod may raise on Windows – safe to ignore
            cls.DATA_FILE.chmod(0o600)
        except Exception:
            pass

    @classmethod
    def load_keys(cls):
        if not cls.DATA_FILE.exists():
            return None, None
        # FIX-06: typed exception instead of bare except
        try:
            fernet = Fernet(cls.get_or_create_key())
            data = fernet.decrypt(cls.DATA_FILE.read_bytes())
            keys = json.loads(data.decode())
            return keys.get("f"), keys.get("g")
        except (Exception,) as exc:
            logger.warning("SecretManager: failed to load keys – %s", exc)
            return None, None

    @classmethod
    def delete_keys(cls):
        for path in [cls.KEY_FILE, cls.DATA_FILE]:
            if path.exists():
                path.unlink()


# --- 配置类 (不再包含硬编码默认值，完全由外部传入) ---
# ---------------------------------------------------------------------------
# FIX-03 & FIX-04: RAGConfig – validation + new Milvus/LLM/retrieval fields
# ---------------------------------------------------------------------------
class RAGConfig:
    def __init__(self, **kwargs):
        # 爬虫配置
        self.sitemap_url = kwargs.get("sitemap_url")
        self.include_patterns = kwargs.get("include_patterns", [])
        self.exclude_patterns = kwargs.get("exclude_patterns", [])
        self.firecrawl_url = kwargs.get("firecrawl_url", "http://localhost:13002")
        
        # 深度、数量及语言控制
        self.max_limit = kwargs.get("max_limit", None) # 对应 URL 过滤深度/数量限制
        self.max_depth = kwargs.get("max_depth", None) # 目录层级深度
        self.target_languages = kwargs.get("target_languages", []) # 抓取网页的语言
        # 性能切片配置
        self.max_threads = kwargs.get("max_threads", 10)
        self.chunk_size = kwargs.get("chunk_size", 1000)
        self.chunk_overlap = kwargs.get("chunk_overlap", 200)
        # FIX-04: Milvus connection now configurable (no more hardcoded 127.0.0.1)
        self.milvus_host       = kwargs.get("milvus_host", "127.0.0.1")
        self.milvus_port       = kwargs.get("milvus_port", "19530")
        
        # 数据库相关配置
        self.collection_name = kwargs.get("collection_name", "rag_docs")
        self.drop_old = kwargs.get("drop_old", False)

        # 新增：Embedding 模型名称（入库阶段用）        
        self.embedding_model   = kwargs.get("embedding_model", "models/text-embedding-004")
        # embedding_device is meaningless for a cloud API — remove or keep as dead field.
        # Safest: keep it but ignore it, so no downstream code breaks.
        self.embedding_device  = kwargs.get("embedding_device", "cpu")

        # FIX-04: LLM provider base URL now configurable
        self.llm_base_url      = kwargs.get("llm_base_url", "https://api.fireworks.ai/inference/v1")

        # FIX-04 / FIX-15: retrieval parameters configurable
        self.retrieval_k       = kwargs.get("retrieval_k", 10)
        self.rerank_top_n      = kwargs.get("rerank_top_n", 3)

        # Ingestion control执行控制
        self.start_index = kwargs.get("start_index", 0)
        self.batch_size = kwargs.get("batch_size", 200)

        # FIX-03: validate after all fields are set
        self._validate()

    def _validate(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be < chunk_size ({self.chunk_size})"
            )
        if self.max_threads < 1:
            raise ValueError("max_threads must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.retrieval_k < 1:
            raise ValueError("retrieval_k must be >= 1")
        if self.rerank_top_n < 1 or self.rerank_top_n > self.retrieval_k:
            raise ValueError(
                f"rerank_top_n ({self.rerank_top_n}) must be between 1 and retrieval_k ({self.retrieval_k})"
            )

    @property
    def milvus_connection_args(self) -> dict:
        return {"host": self.milvus_host, "port": self.milvus_port}


# --- 状态定义 ---
# LangGraph state
class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages] # 对话历史 (自动追加)
    summary: str          # 长对话摘要
    rewritten_query: str  # 改写后的查询词
    user_profile: dict    # 用户画像 (新增)
    retrieval_results: list  # 新增：结构化的检索结果
    retrieval_ok: bool       # 新增：是否检索到文档

# --- 核心引擎：爬虫与入库 ---
# UniversalRAGEngine
class UniversalRAGEngine:
    def __init__(self, config: RAGConfig, api_keys: dict, embeddings=None):
        self.config = config
        self._original_sitemap_url = config.sitemap_url
        self.google_key = api_keys.get("GOOGLE_API_KEY")
        # --- 还原底层的 urllib3 重试逻辑 ---
        # HTTP session with auto-retry
        from urllib3.util import Retry
        from requests.adapters import HTTPAdapter
        self.session = requests.Session()
        retries = Retry(
            total=3,                # 总重试次数
            backoff_factor=1,       # 等待间隔系数
            status_forcelist=[500, 502, 503, 504], # 遇到这些状态码重试
        )
        self.session.mount("http://",  HTTPAdapter(max_retries=retries))
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

        # FIX-10: lock that guards all read-modify-write operations on existing_urls_set
        self._url_set_lock = threading.Lock()

        # FIX-12: cache for the sitemap URL list so run_generator doesn't re-fetch
        self._cached_urls: Optional[List[str]] = None
        self._external_embeddings = embeddings  # store it before _setup_milvus
        self._setup_milvus()

    # -----------------------------------------------------------------------
    # Milvus setup
    # -----------------------------------------------------------------------
    def _setup_milvus(self):
        # FIX-08: always initialise existing_urls_set before the try block
        self.vector_store      = None
        self.existing_urls_set = set()

        try:
            connections.connect(
                alias="default",
                host=self.config.milvus_host,   # FIX-04
                port=self.config.milvus_port,
            )

            if self._external_embeddings is None:
                logger.error("Embeddings not provided. UniversalRAGEngine requires embeddings injection from app.py.")
                self.vector_store = None
                return
            self.embeddings = self._external_embeddings

            self.vector_store = Milvus(
                embedding_function=self.embeddings,
                connection_args=self.config.milvus_connection_args,  # FIX-04
                collection_name=self.config.collection_name,
                auto_id=False,
                drop_old=self.config.drop_old,
            )
            self.existing_urls_set = self._load_existing_urls()
        # FIX-06: named exception with logging
        except Exception as exc:
            logger.warning("Milvus connection failed (preview-only mode OK): %s", exc)

    # -----------------------------------------------------------------------
    # FIX-09: load existing URLs via query_iterator (no 16 K cap)
    # -----------------------------------------------------------------------
    def _load_existing_urls(self) -> set:
        existing: set = set()
        logger.info("Syncing existing URLs from Milvus for deduplication…")
        try:
            coll = Collection(self.config.collection_name, using="default")

            # query_iterator pages through ALL records without offset limit
            iterator = coll.query_iterator(
                expr='pk != ""',
                output_fields=["source"],
                batch_size=1000,
            )
            while True:
                batch = iterator.next()
                if not batch:
                    iterator.close()
                    break
                for item in batch:
                    src = item.get("source")
                    if src:
                        existing.add(src)

            logger.info("Loaded %d existing URLs for dedup.", len(existing))
        # FIX-06: named exception
        except Exception as exc:
            logger.warning(
                "Could not load existing URLs (first run or empty DB): %s", exc
            )
        return existing


    #严谨的通过 URL 判断语言种类进行过滤的逻辑函数：支持种子保护、路径前缀匹配、子域名匹配
    # URL language filter
    def _is_url_target_lang(self, url, target_langs, seed_urls=None):
        # 1. 种子链接保护：如果是用户直接输入的起始种子，100% 放行
        if seed_urls and url in seed_urls:
            return True, "unknown"

        # 如果没有指定目标语言，则全放行
        if not target_langs:
            return True, "unknown"

        try:
            parsed = urlparse(url)
            url_path = parsed.path.lower()
            # 移除开头和结尾的 / 并按层级切分
            path_segments = [s for s in url_path.split("/") if s]
        
            # 扩展标准语言代码库 (加上极容易混淆的技术文档语言和小语种)
            known_lang_codes = {
                "zh", "en", "ja", "ko", "fr", "de", "es",
                "zh-cn", "zh-tw", "zh-hant", "en-us", "en-gb",
                "ru", "it", "ar", "pt", "id", "vi", "nl", "pl",
            }

            # --- 跟下面的“--- D. 严格排除”的代码逻辑重复，故全部注释掉
            # A. 检测 URL 路径的第一段 ---
            # 例如 milvus.io/zh/docs，第一段是 'zh'
            #first_segment = path_segments[0] if path_segments else ""
            #if first_segment in known_lang_codes:
            #    is_match = any(first_segment.startswith(tl) for tl in target_langs)
            #    return is_match, first_segment # 明确指定了语言，直接返回匹配结果

            # --- B. 检测子域名 ---
            # 例如 zh.wikipedia.org，子域名是 'zh'
            netloc_parts = parsed.netloc.lower().split(".")
            if len(netloc_parts) > 2:
                subdomain = netloc_parts[0]
                if subdomain in known_lang_codes:
                    is_match = any(subdomain.startswith(tl) for tl in target_langs)
                    return is_match, subdomain

            # --- C. 检测查询参数 ---
            # 例如 example.com?lang=en
            query_params = parsed.query.lower()
            if "lang=" in query_params or "language=" in query_params:
                for tl in target_langs:
                    if f"lang={tl}" in query_params or f"language={tl}" in query_params:
                        return True, tl
                return False, "unknown" # 指明了语言但不匹配

            # --- D. 检测路径中第一个语言标识段（仅看前3段）---
            # Universal convention: language codes appear in the first 1-2 path segments.
            # Scanning beyond segment 3 risks false-positives on filenames/slugs.
            for idx, segment in enumerate(path_segments):
                if idx >= 3:
                    # Beyond 3rd segment — not a language selector, stop checking
                    break
                if segment in known_lang_codes:
                    is_match = any(segment.startswith(tl) for tl in target_langs)
                    return is_match, segment

            # # --- E. 严格排除：路径深处出现的已知语言标识 ---
            # # 例如 /docs/fr/xxx，如果 fr 不在目标语言中，坚决拦截
            # for segment in path_segments:
                # if segment in known_lang_codes:
                    # is_match = any(segment.startswith(tl) for tl in target_langs)
                    # return is_match, segment


            # --- F. 兜底放行, 无明确语言标识的 URL（如 /docs/intro），放行到第二层内容检测
            logger.debug("URL language 没有明确标记，默认放行: %s", url)
            return True, "unknown"
        except (ValueError, AttributeError) as exc:
            logger.debug("URL lang check failed for %s: %s", url, exc)
            return True, "unknown"

        # 保险：严格确保此函数始终返回 (bool, str)
        logger.debug("URL lang check意外执行路径，默认放行: %s", url)
        return True, "unknown"

    # -----------------------------------------------------------------------
    # Sitemap auto-discovery
    # -----------------------------------------------------------------------
    def _auto_discover_sitemap(self, input_url):
        #智能探测：
        #1. 从 robots.txt 找 Sitemap
        #2. 猜测常见路径 (sitemap.xml, sitemap_index.xml)
        #3. 返回 (真实的sitemap_url, 路径过滤器path_filter)
        parsed = urlparse(input_url)
        # 提取基础域名 (scheme + netloc)，例如 https://sohu.com
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        
        # 提取用户想要的路径过滤器，例如 /doc
        # 如果路径很短（只有 /），则没有过滤器
        path_filter = parsed.path if len(parsed.path) > 1 else None

        potential_sitemaps = []
        
        # A. 尝试读取 robots.txt
        try:
            robots_url = f"{base_url}/robots.txt"
            resp = self.session.get(
                robots_url,
                headers=get_headers(self.config.target_languages),
                timeout=5,
            )
            if resp.status_code == 200:
                for line in resp.text.splitlines():
                    if line.lower().startswith("sitemap:"):
                        potential_sitemaps.append(line.split(":", 1)[1].strip())
        # FIX-06: named exception
        except requests.RequestException as exc:
            logger.debug("robots.txt fetch failed: %s", exc)


        # B. 猜测常见路径 Common paths
        for p in ["/sitemap.xml", "/sitemap_index.xml", "/sitemap.php"]:
            potential_sitemaps.append(f"{base_url}{p}")

        # C. 验证哪个 Sitemap 是真的
        valid_sitemap = None
        current_headers=get_headers(self.config.target_languages)
        for sm_url in potential_sitemaps:
            try:
                # 只发 HEAD 请求探测，不下载内容，速度快
                r = self.session.head(sm_url, headers=current_headers, timeout=5)
                if r.status_code == 200:
                    valid_sitemap = sm_url
                    break
                # 有些服务器不支持 HEAD，用 GET 尝试前 10 个字节
                if r.status_code == 405: 
                    r = self.session.get(sm_url, headers=current_headers, stream=True, timeout=5)
                    if r.status_code == 200:
                        valid_sitemap = sm_url
                        break
            except requests.RequestException:
                continue
        
        # 如果找到了 Sitemap，返回它；如果没找到，但用户输入本身以 .xml 结尾，那它自己就是
        if not valid_sitemap and input_url.endswith(".xml"):
            valid_sitemap = input_url
            path_filter = None # 如果直接给了 xml，就不需要路径过滤了

        return valid_sitemap, path_filter

    # -----------------------------------------------------------------------
    # Doc ID
    # -----------------------------------------------------------------------
    def generate_doc_id(self, url: str) -> str:
        clean_url = url.split("#")[0].rstrip("/")
        return hashlib.md5(clean_url.encode()).hexdigest()[:8]

    # -----------------------------------------------------------------------
    # Sitemap preview
    # -----------------------------------------------------------------------
    def preview_sitemap(self):
        #仅获取并过滤 URL，不进行爬取。用于 UI 预览。支持 Sitemap Index 索引文件的深度解析
        log = []
        try:
            # 智能探测 Sitemap
            raw_url = self._original_sitemap_url
            log.append(f"正在智能分析目标: {raw_url} ...")
        
            real_sitemap_url, path_filter = self._auto_discover_sitemap(raw_url)
        
            if not real_sitemap_url:
                return [], [f"⚠️ 在 {raw_url} 未发现 Sitemap。将回退到【普通爬虫模式】，速度会较慢。建议手动寻找该网站的 sitemap.xml 链接。"]
        
            log.append(
                f"🎯 Sitemap: {real_sitemap_url}"
                + (f"  |  path filter: '{path_filter}'" if path_filter else "")
            )

            # 使用伪装头请求
            resp = self.session.get(
                real_sitemap_url,
                headers=get_headers(self.config.target_languages),
                timeout=20,
            )
            if resp.status_code != 200:
                return [], [f"❌ 请求 Sitemap 失败，状态码: {resp.status_code}"]

            # --- 修改部分：显式指定 'xml' 解析器 ---
            # 如果报错，请确保执行了 pip install lxml
            soup = bs4.BeautifulSoup(resp.content, features="xml") 
            
            # 1. 查找所有的 loc 标签
            all_locs = [loc.text.strip() for loc in soup.find_all("loc")]
            sitemaps = [l for l in all_locs if l.endswith(".xml")]
            pages    = [l for l in all_locs if not l.endswith(".xml")]
            final_urls = pages
            
            # 3. 如果存在子 Sitemap，递归抓取一层
            if sitemaps:
                log.append(f"检测到 {len(sitemaps)} 个子索引文件，正在深度解析...")
                for s_url in sitemaps:
                    try:
                        s_resp = self.session.get(
                            s_url,
                            headers=get_headers(self.config.target_languages),
                            timeout=10,
                        )
                        s_soup = bs4.BeautifulSoup(s_resp.content, features="xml")
                        final_urls.extend(
                            [loc.text.strip() for loc in s_soup.find_all("loc")]
                        )
                    # FIX-06: named exception
                    # Catch ALL exceptions (not just RequestException) so that
                    # bs4 parse errors, encoding issues, etc. don't bubble up
                    # to the outer try and wipe out all collected URLs.
                    except Exception as exc:
                        logger.warning("Sub-sitemap %s failed: %s", s_url, exc)
            
            # 4. 执行过滤逻辑 (这里要确保过滤的是最终的页面 URL)
            target_langs = self.config.target_languages
            # 预先判断是否需要进行深度检查，避免在循环中重复判断 None
            limit_depth = self.config.max_depth is not None and self.config.max_depth > 0
            filtered     = []
            for u in final_urls:
                # 路径自动过滤 (解决 sohu.com/doc 的需求)
                if path_filter and path_filter not in u:
                    continue

                # 检查网址包含逻辑
                if self.config.include_patterns and not any(
                    p in u for p in self.config.include_patterns
                ):
                    continue
                # 检查网址排除逻辑 (OR 逻辑，它也会排除前面包含逻辑已经包含过的网址)
                if self.config.exclude_patterns and any(
                    p in u for p in self.config.exclude_patterns
                ):
                    continue

                # 第一层语言过滤：基于 URL 模式拦截
                if target_langs:
                    # ⚠️ 此时 seed_urls 应该是 real_sitemap_url 或者 raw_url
                    is_target, _ = self._is_url_target_lang(
                        url=u,
                        target_langs=target_langs,
                        seed_urls=[raw_url, real_sitemap_url],
                    )
                    if not is_target:
                        continue


                # 目录深度过滤 (只有当 max_depth 有值且 > 0 时才限制)
                if limit_depth:
                    if len([s for s in urlparse(u).path.split("/") if s]) > self.config.max_depth:
                        continue

                filtered.append(u)
            
            # 去重
            filtered = list(set(filtered))
            
            # 爬取网页最大数量限制 (只有当 max_limit 有值且 > 0 时才限制)
            if self.config.max_limit and len(filtered) > self.config.max_limit:
                log.append(
                    f"⚠️ URL count {len(filtered)} exceeds max_limit {self.config.max_limit} – truncating."
                )
                filtered = filtered[: self.config.max_limit]
            else:
                log.append(f"ℹ️ 根据当前过滤条件，初步估计将处理 {len(filtered)} 个 URL。")

            # 💡 极其重要：更新 config 中的 sitemap_url 为真实的 XML 地址
            # 这样后续 run_ingestion 就会进入“高速 Sitemap 模式”而不是“低速爬虫模式”
            self._real_sitemap_url = real_sitemap_url 
 
            # FIX-12: cache the result so run_generator can reuse it
            self._cached_urls = filtered
            return filtered, log
            
        except Exception as exc:
            logger.exception("preview_sitemap crashed")
            return [], [f"❌ Sitemap error: {exc}"]

    # -----------------------------------------------------------------------
    # FIX-13: process_url now also returns the fetched soup for link extraction
    # Returns (success, msg, soup_or_None)
    # -----------------------------------------------------------------------
    def process_url(self, url: str):
        #通用 URL 处理核心：包含 3 次指数退避重试, 阶梯式自适应抓取 (Requests极速 -> Firecrawl重载) 保证效率
        #支持：静态 HTML, 动态 JS 渲染, PDF, 多语言自动识别
        #动态 Header 确保目标语言内容
        #严格遵循用户 UI 选择的语言进行入库过滤
        max_retries = 3
        retry_delay = 2 
        content = ""
        raw_meta = {} # 初始化，确保 Branch B 不报错
        lang_trace = []
        
        if self.vector_store is None:
            return False, "Milvus 连接失败, 无法入库(如果是仅预览模式可忽略)", None
        
        # 预判文件类型
        lower_url = url.lower()
        is_pdf = lower_url.endswith('.pdf')
        # 工业级过滤：跳过纯图片/音频/视频，但保留 office 文档(未来可扩展)
        skip_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp', '.mp4', '.mp3', '.zip', '.exe', '.css')
        if any(lower_url.endswith(ext) for ext in skip_extensions):
            return False, "跳过媒体/二进制文件", None

        # === 新增：强制引入 URL 语言识别结果 ===
        is_allowed, url_lang = self._is_url_target_lang(url, self.config.target_languages)
        lang_trace.append(f"URL:{url_lang}")
        if not is_allowed:
            return False, f"跳过：URL 路径明确为非目标语言", None

        # 获取用户在 UI 选定的语言头 (由 config.target_languages 传入)
        # 确保向服务器请求时，它更倾向于返回用户想要的语言版本
        current_headers = get_headers(self.config.target_languages)
        fetched_soup    = None  # FIX-13: will be returned for link reuse

        for attempt in range(max_retries):
            try:
                metadata = {"source": url, "title": os.path.basename(url)}
                page_lang = url_lang # 默认继承 URL 识别出来的语言

                # --- 阶段 A：快速路径 (Requests) ---
                # 针对大部分静态网页，速度极快（几百毫秒）
                if not is_pdf:
                    try:
                        resp = self.session.get(url, headers=current_headers, timeout=15)
                        if resp.status_code == 200:
                            soup = bs4.BeautifulSoup(resp.content, "html.parser")
                            fetched_soup = soup  # FIX-13: capture for caller
                            for tag in soup(["script", "style", "nav", "footer", "header"]):
                                tag.decompose()
                            temp_content = soup.get_text(separator="\n").strip()
                            # 判定：如果提取出的纯文本大于 500 字，则认为是有效静态页，不再启动 Firecrawl
                            if len(temp_content) > 500:
                                content = temp_content
                                if soup.title:
                                    metadata["title"] = soup.title.string
                    # FIX-06: named exception
                    except requests.RequestException as exc:
                        logger.debug("Static fetch failed for %s: %s", url, exc)

                # --- 阶段 B：深度路径 (Firecrawl) ---
                # 触发条件：PDF文件、快速路径请求失败、或者快速路径抓取的内容太短（暗示是 JS 动态渲染的单页应用）
                if not content or len(content) < 500:
                    if self.config.firecrawl_url:
                        api_url = f"{self.config.firecrawl_url.rstrip('/')}/v1/scrape"
                        # 重点：传递 waitFor 让 JS 充分加载，并带上动态语言 headers
                        payload = {
                            "url": url, 
                            "formats": ["markdown"], 
                            "onlyMainContent": True,
                            "waitFor": 1500,
                            "headers": current_headers,
                        }
                        try:
                            fc_resp = self.session.post(api_url, json=payload, timeout=90)
                            if fc_resp.status_code == 200:
                                data     = fc_resp.json().get("data", {})
                                content  = data.get("markdown", "")
                                raw_meta = data.get("metadata", {})
                                metadata.update({
                                    "title":       raw_meta.get("title") or metadata["title"],
                                    "description": raw_meta.get("description", ""),
                                    "language":    raw_meta.get("language", ""),
                                })
                        except requests.RequestException as exc:
                            logger.debug("Firecrawl failed for %s: %s", url, exc)

                # --- 阶段 C：PDF 本地兜底逻辑 ---
                # 如果是 PDF 且 Firecrawl 没有成功处理，则用本地内存流快速解析
                if not content and is_pdf:
                    try:
                        from pypdf import PdfReader
                        
                        # 显式发起请求获取 PDF 二进制流
                        pdf_resp = self.session.get(url, headers=current_headers, timeout=60)
                        if pdf_resp.status_code == 200:
                            reader  = PdfReader(io.BytesIO(pdf_resp.content))
                            content = "\n".join(
                                [p.extract_text() for p in reader.pages if p.extract_text()]
                            )
                            metadata["title"] = os.path.basename(url)
                    except Exception as exc:
                        logger.warning("PDF local parse failed for %s: %s", url, exc)

                # --- 检查内容质量 ---
                if not content or len(content.strip()) < 50:
                    return False, "内容在渲染后依然为空或过短", fetched_soup

                # 如果之前根据URL第一层过滤不能准确过滤的，这里根据网页HTML代码里的语言元数据属性进行第二层过滤，如果元数据为空则调用langdetect进行网页内容精准语言检测的第三层过滤：
                if self.config.target_languages:
                    # 如果 URL 没有识别出语言 (即 url_lang == "unknown")，我们才动用底层检测
                    # 1. 优先尝试从元数据获取 (Firecrawl 抓取的 HTML lang 标签)
                    # 转换成小写并取前缀，例如 'zh-CN' -> 'zh'
                    if page_lang == "unknown":
                        page_lang = raw_meta.get("language", "").lower().split("-")[0]
                        
                    # 2. 如果元数据为空，则对 content 内容进行采样检测语言种类 (取前 500 字)
                    if not page_lang:
                        try:
                            # 采样检测，避免全文检测浪费性能
                            # 跳过前200字（通常是导航栏），取200-1000字的正文采样
                            sample = content[200:1000]
                            results = detect_langs(sample)
                            if results and results[0].prob >= 0.90:
                                page_lang = results[0].lang.lower().split("-")[0]
                            else:
                                # 置信度不足 → 标为 unknown，由下游放行
                                page_lang = "unknown"
                        except Exception:
                            page_lang = "unknown"
                        
                    # 3. 拦截逻辑：先归一化到基础语言码（zh-hant→zh，en-us→en），再做精确匹配
                    base_lang = page_lang.split("-")[0] if page_lang != "unknown" else "unknown"
                    if base_lang != "unknown" and base_lang not in self.config.target_languages:
                        return False, f"过滤：检测到非目标语言 '{page_lang}'", fetched_soup
                        
                metadata["language"] = page_lang # 记录最终识别出的语言

                # --- 切片与入库 ---
                doc = Document(page_content=content, metadata=metadata)
                splits = self.text_splitter.split_documents([doc])
                    
                # 还原 ID 生成去重逻辑, 为每个切片生成唯一的派生 ID（Derived ID），格式为 URL_Hash_序号
                base_id = self.generate_doc_id(url=url)
                # 核心：让 ID 和切片索引绑定，确保生成带后缀的唯一ID且可复用
                ids = [f"{base_id}_{i}" for i in range(len(splits))]

                self.vector_store.add_documents(splits, ids=ids)
                return True, f"成功 ({len(splits)} 切片, 语言: {page_lang})", fetched_soup

            except Exception as exc:
                # 指数退避重试逻辑 (2s -> 4s -> 8s)
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))
                else:
                    return False, f"Exception after {max_retries} retries: {exc}", fetched_soup

        return False, "All retries exhausted", fetched_soup

    #🚀 多线程递归爬虫引擎，用于处理没有 Sitemap 的网站，采用分层并发策略
    # FIX-10 + FIX-13: thread-safe recursive crawl; reuse fetched soup for links
    def _recursive_crawl(self, seed_url, target_langs):
        visited = set()
        current_layer = [seed_url]
        # 确保 existing_urls_set 存在 (防止未初始化报错)
        existing_set  = self.existing_urls_set  # shared reference
        
        # 定义内部处理函数
        def _fetch_and_extract(u):
            # 如果线程开得多，自动微调随机延迟
            # 线程越多，每个线程爬完后的休息时间稍微拉长一点点
            sleep_time = 0.1 * (self.config.max_threads / 5)
            time.sleep(random.uniform(0.1, sleep_time))
                
            # FIX-13: process_url now returns the soup it already fetched
            success, msg, fetched_soup = self.process_url(u)

            # 2. 提取子链接用于下一层
            new_links = set()
            base_domain  = urlparse(seed_url).netloc
                
            # === A 计划：极速模式 (Requests) ===
            # 针对 80% 的静态网站，0.2秒解决战斗
            # Reuse fetched_soup for link extraction (no second HTTP request)
            if fetched_soup is not None:
                for a in fetched_soup.find_all("a", href=True):
                    link = a["href"]
                    if link.startswith("/"):
                        parsed = urlparse(u)
                        link   = f"{parsed.scheme}://{parsed.netloc}{link}"
                    elif not link.startswith("http"):
                        continue
                    if urlparse(link).netloc == base_domain:
                        new_links.add(link.split("#")[0].rstrip("/"))
            else:
                # Fallback: Firecrawl link extraction if soup unavailable
                # === B 计划：重型模式 (Firecrawl) ===
                # 只有当 A 计划失败，或者 A 计划觉得“链接太少不正常”时才触发
                if self.config.firecrawl_url:
                    try:
                        api_url = f"{self.config.firecrawl_url.rstrip('/')}/v1/scrape"
                        payload = {
                            "url": u,
                            "formats": ["links"], # 专用模式：只提取链接
                            "waitFor": 2000,      # 等待 2秒让 JS 菜单加载
                            "headers": get_headers(self.config.target_languages),
                        }
                        fc_resp = self.session.post(api_url, json=payload, timeout=60)
                        
                        if fc_resp.status_code == 200:
                            data = fc_resp.json().get("data", {})
                            for link in data.get("links", []):
                                if link and urlparse(link).netloc == base_domain:
                                    new_links.add(link.split("#")[0].rstrip("/"))
                    except requests.RequestException as exc:
                        logger.debug("Firecrawl link extraction failed: %s", exc)

            return success, msg, new_links

        for depth in range((self.config.max_depth or 2) + 1):
            if not current_layer:
                break
            yield {"type": "log", "message": f"🌊 递归爬取第 {depth} 层，待处理: {len(current_layer)} 个页面"}
            
            next_layer = set()
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.config.max_threads
            ) as executor:
                future_to_url = {}
                
                # 🔥【关键修改位置 1】在此处进行内存去重拦截
                for u in current_layer:
                    if u in visited:
                        continue
                    clean_u = u.split("#")[0].rstrip("/")

                    # FIX-10: thread-safe check-and-add
                    with self._url_set_lock:
                        if clean_u in existing_set:
                            yield {"type": "log", "message": f"⏭️ Already in DB: {u}"}
                            visited.add(u)
                            continue
                        # Reserve this URL before submitting to thread pool
                        existing_set.add(clean_u)
                    future_to_url[executor.submit(_fetch_and_extract, u)] = u
                    visited.add(u)

                for future in concurrent.futures.as_completed(future_to_url):
                    u = future_to_url[future]
                    success, msg, links = future.result()
                    
                    # 🔥【关键】成功入库后，立即更新内存 Set，防止同一次运行中后续重复爬取
                    # FIX-10: if ingestion failed, remove the reservation
                    if not success:
                        clean_u = u.split("#")[0].rstrip("/")
                        with self._url_set_lock:
                            existing_set.discard(clean_u)

                    # 汇报进度给 UI
                    yield {
                        "type":      "progress",
                        "completed": len(visited),
                        "total":     self.config.max_limit or 999, # 递归模式总数未知，用 limit 占位
                        "url":       u,
                        "status":    "✅" if success else "❌",
                        "detail":    msg,
                    }
                    
                    # 过滤并收集下一层
                    if depth < (self.config.max_depth or 2):
                        for link in links:
                            if link not in visited:
                                is_target, _ = self._is_url_target_lang(link, target_langs)
                                if not is_target:
                                    continue
                                if self.config.include_patterns and not any(
                                    p in link for p in self.config.include_patterns
                                ):
                                    continue
                                if self.config.exclude_patterns and any(
                                    p in link for p in self.config.exclude_patterns
                                ):
                                    continue
                                next_layer.add(link)

            current_layer = list(next_layer)
            if self.config.max_limit and len(visited) >= self.config.max_limit:
                yield {"type": "log", "message": f"🛑 已达到最大爬取数量限制 {self.config.max_limit}"}
                break

    # -----------------------------------------------------------------------
    # FIX-12: run_generator reuses cached sitemap URLs
    # FIX-11: use enumerate() instead of batch.index(url)
    # FIX-10: thread-safe check-and-add in Sitemap mode
    # -----------------------------------------------------------------------
    def run_generator(self, start_index: int = 0):
        # 执行器：负责断点续传 (start_index) 和向 UI yield 状态, 支持中途检查外部状态
        # 获取 URL (复用之前的预览逻辑)
        # FIX-12: reuse cached URLs from preview_sitemap if available
        if self._cached_urls is not None:
            target_urls = self._cached_urls
            logs        = [
                "♻️ Reusing sitemap URLs from preview (no re-fetch needed).",
                f"Total URLs: {len(target_urls)}, start index: {start_index}",
            ]
        else:
            target_urls, logs = self.preview_sitemap()
        
        if self.vector_store is None:
            yield {"type": "log", "message": "Milvus 连接失败, 无法入库"}
            return
        

        for log in logs:
            yield {"type": "log", "message": log}

        # 确保 existing_urls_set 存在
        existing_set = self.existing_urls_set

            
        # --- 模式 A: Sitemap 模式 (有 URL 列表) ---
        if target_urls:
            actual_total = len(target_urls)
            current_target = target_urls[start_index:]
            total_remaining = len(current_target)  # explicit alias – referenced by some merged code
            yield {"type": "log", "message": f"🚀 启动 Sitemap 高速模式。总计: {actual_total}, 起始索引: {start_index}，剩余 {total_remaining} 个, 并发线程: {self.config.max_threads}"}
            for batch_start in range(0, len(current_target), self.config.batch_size):
                batch = current_target[batch_start: batch_start + self.config.batch_size]

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.config.max_threads
                ) as executor:
                    future_to_idx_url = {}
                    # 🔥【关键修改位置 2】Sitemap 模式的去重拦截
                    # FIX-11: enumerate gives correct O(1) index, no batch.index()
                    for local_idx, url in enumerate(batch):
                        global_idx = start_index + batch_start + local_idx
                        clean_url  = url.split("#")[0].rstrip("/")

                        # FIX-10: thread-safe check-and-reserve
                        with self._url_set_lock:
                            if clean_url in existing_set:
                                yield {
                                    "type":      "progress",
                                    "completed": global_idx + 1,
                                    "total":     actual_total,
                                    "url":       url,
                                    "status":    "⏭️",
                                    "detail":    "库中已存在",
                                }
                                continue
                            existing_set.add(clean_url)

                        future_to_idx_url[
                            executor.submit(self.process_url, url)
                        ] = (global_idx, url, clean_url)

                    for future in concurrent.futures.as_completed(future_to_idx_url):
                        global_idx, url, clean_url = future_to_idx_url[future]
                        # FIX-13: process_url returns 3-tuple; unpack third value
                        success, msg, _ = future.result()

                        # FIX-10: roll back reservation on failure
                        if not success:
                            with self._url_set_lock:
                                existing_set.discard(clean_url)

                        yield {
                            "type":      "progress",
                            "completed": global_idx + 1,
                            "total":     actual_total,
                            "url":       url,
                            "status":    "✅" if success else "❌",
                            "detail":    msg,
                        }
        
        # --- 模式 B: 递归模式 (Sitemap 未找到) ---
        else:
            yield {"type": "log", "message": "🔍 进入多线程递归爬虫模式..."}
            # yield from 能够把 _recursive_crawl 生成的所有状态直接转发给 UI
            
            yield from self._recursive_crawl(self._original_sitemap_url, self.config.target_languages)


# --- 聊天机器人 (Graph Logic) ---
class RAGChatBot:
    def __init__(
        self, 
        fireworks_key: str,
        google_key:    str,
        collection_name: str = "rag_docs",
        llm_model:    str = "accounts/fireworks/models/llama-v3p3-70b-instruct",
        embedding_model: str = "models/text-embedding-004",
        embedding_device: str = "cpu", # kept for signature compatibility, not used。 如果是跑BAAI/bge-m3本地模型，有显卡写 cuda，Mac 写 mps，没有写 cpu。 
        embeddings=None,
        # FIX-04: LLM base URL now configurable
        llm_base_url: str = "https://api.fireworks.ai/inference/v1",
        # FIX-04: Milvus connection configurable
        milvus_host: str = "127.0.0.1",
        milvus_port: str = "19530",
        # FIX-15: retrieval parameters configurable
        retrieval_k:  int = 10,
        rerank_top_n: int = 3,
    ):
        self.retrieval_k  = retrieval_k
        self.rerank_top_n = rerank_top_n

        # 1. 初始化 LLM (Llama 3.3 70B)
        self.llm = ChatOpenAI(
            model=llm_model,
            api_key=fireworks_key,
            base_url=llm_base_url,   # FIX-04: no longer hardcoded
            temperature=0,
        )
        
        
        if embeddings is None:
            raise ValueError("RAGChatBot requires embeddings to be passed from app.py.")
        self.embeddings = embeddings

        # 3. Milvus – FIX-04: use configurable host/port
        self.vector_store = Milvus(
            embedding_function=self.embeddings,
            connection_args={"host": milvus_host, "port": milvus_port},
            collection_name=collection_name,
        )
        
        # 4. 初始化 FlashRank (手动重排序模式)
        # model_name 选择 ms-marco-MultiBERT-L-12 效果最好
        self.ranker = Ranker(model_name="ms-marco-MultiBERT-L-12", cache_dir="./opt")
        
        # 5. 构建图
        self.graph = self._build_graph()

    # 能列出Milvus所有数据库集合及数据量的工具函数
    @staticmethod #定义静态函数方法：无需实例化这个classic类，即可调用它获取所有集合信息
    def get_milvus_collections_info(
        milvus_host: str = "127.0.0.1",
        milvus_port: str = "19530",
    ) -> list:
        #获取所有集合名称及对应的精确实体数量
        # --- 隔离连接策略开始 ---
        # 定义一个专门用于查询列表的别名，不使用 'default'
        try:
            # 1. 使用隔离别名，确保不干扰主程序
            query_alias = f"list_alias_{uuid.uuid4().hex[:8]}"
            if connections.has_connection(query_alias):
                connections.disconnect(query_alias)
            
            # 建立一个全新的、独立的连接
            connections.connect(host=milvus_host, port=milvus_port, alias=query_alias)
            info_list = []
            for name in utility.list_collections(using=query_alias):
                try:
                    # 2. 这里的 Collection 现在能被正确识别了
                    coll = Collection(name, using=query_alias)
                    
                    # 3. 尝试多种方式获取数量，确保万无一失
                    # 优先使用你 Jupyter 验证成功的 num_entities
                    count = coll.num_entities
                    
                    # 如果 num_entities 返回 0，尝试穿透查询 count(*)
                    if count == 0:
                        try:
                            # 只有加载了才能查询，这里尝试微量查询
                            res   = coll.query(expr="", output_fields=["count(*)"])
                            count = res[0]["count(*)"]
                        # FIX-06: named exception
                        except Exception as exc:
                            logger.debug("count(*) failed for %s: %s", name, exc)
                            try:
                            # 如果没加载，尝试读取 stats 账本
                                stats = utility.get_collection_stats(name, using=query_alias)
                                count = int(stats.get("row_count", 0))
                            except Exception as exc2:
                                logger.debug("get_collection_stats failed for %s: %s", name, exc2)
                                count = 0
                            #for item in stats:
                            #    if 'row_count' in item:
                            #        count = int(item['row_count'])
                            #        break
                    
                    info_list.append({"name": name, "count": count})
                except Exception as exc:
                    # 如果出错，在终端打印具体原因，方便我们继续排查
                    logger.warning("Failed to inspect collection %s: %s", name, exc)
                    info_list.append({"name": name, "count": 0})
            
            # 4. 释放连接
            connections.disconnect(query_alias)
            return info_list
            
        except Exception as exc:
            logger.error("get_milvus_collections_info failed: %s", exc)
            return []
    # --- 节点 2: 查询改写 ---
    def _rewrite(self, state: GraphState):
        # --- 修复：获取历史上下文 ---
        question = state["messages"][-1].content
        # 获取最近 3 轮对话作为上下文
        recent_msgs = state["messages"][:-1][-3:] 
        history_text = "\n".join([f"{m.type}: {m.content}" for m in recent_msgs])
        
        prompt = f"""你是一个搜索优化专家。请根据对话历史将用户的最新问题改写为一个独立的、精准的搜索关键词。
        
        对话历史：
        {history_text}
        
        用户最新问题：{question}
        
        改写后的搜索词（只输出结果，不要解释）："""
        
        return {"rewritten_query": self.llm.invoke(prompt).content}

    # --- 节点 3: 检索与重排序 (核心修复部分) ---
    def _retrieve(self, state: GraphState):
        #1. 向量检索 Top-K. 2. FlashRank 重排序 Top-N
        query = state.get("rewritten_query") or state["messages"][-1].content
        
        retrieval_results = []
        context = "检索失败。"
        ok = False  

        try:
            # Step A: 粗排 (获取 10 条)
            base_docs = self.vector_store.as_retriever(
                search_kwargs={"k": self.retrieval_k}   # FIX-15
            ).invoke(f"search_query: {query}")
            if not base_docs:
                context = "向量库中没有检索到相关文档。"
            else:
                # Step B: 构造 FlashRank 输入格式
                passages = [
                    {"id": i, "text": d.page_content, "meta": d.metadata}
                    for i, d in enumerate(base_docs)
                ]
                # Step C: 精排 (Rerank)
                results = self.ranker.rerank(RerankRequest(query=query, passages=passages))
                # Step D: 截取 Top 3
                context_parts = []
                for r in results[: self.rerank_top_n]:   # FIX-15
                    # 尝试获取来源，回答时告诉用户“这也信息来自哪篇文档”，增加可信度。如果没有则显示未知
                    meta = r.get("meta", {}) or {}
                    source = meta.get("source", "未知来源")
                    text = r.get("text", "") or ""
                    score = r.get("score", None)

                    # 收集给 UI 用的结构化结果
                    retrieval_results.append(
                        {
                            "source": source,
                            "score": float(score) if score is not None else None,
                            "snippet": text[:500],
                        }
                    )

                    context_parts.append(
                        f"📄 来源: {source}\n内容: {text}"
                    )
                
                context = "\n\n".join(context_parts) if context_parts else "检索结果为空。"
                ok = bool(context_parts)

        except Exception as exc:
            logger.warning("Retrieval failed: %s", exc)
            context = "Retrieval failed."

        return {
            "messages": [SystemMessage(content=f"参考资料：\n{context}")],
            "retrieval_results": retrieval_results,
            "retrieval_ok": ok,
        }

    # --- 节点 4: 生成回答 ---
    def _generate(self, state: GraphState):
        #生成最终回复，注入用户画像
        profile = state.get("user_profile", {})
        
        # 动态 System Prompt
        system_prompt = f"""你是一个专业的数据分析专家助手。
        用户画像：{profile}
        请根据参考资料回答用户问题。如果资料不足，请诚实告知。"""
        
        # 构造消息列表：System + History
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        return {"messages": [self.llm.invoke(messages)]}

    # --- 核心节点逻辑 ---
    def _profile(self, state: GraphState):
        #基于详细 Schema 提取用户画像
        # 1. 获取现有画像（如果有）
        current_profile = state.get("user_profile", {})
        turn_count      = len(state["messages"])

        # skip LLM call when profile is fresh (every 4 turns), only update profile when it's empty OR every 4 turns
        if current_profile and turn_count % 4 != 0:
            return {"user_profile": current_profile}

        
        # 2. 获取最近的对话历史（避免 Token 消耗过大，只看最近 5 条）
        recent_messages = state["messages"][-5:]
        dialogue_text = "\n".join([f"{m.type}: {m.content}" for m in recent_messages])

        # 3. 定义LLM提取用户画像的 System Prompt，采用：通用+技术混合画像 ---
        extract_system_prompt = """
        你是一个专业的各种用户信息分析师。你的任务是构建用户的“数字孪生”画像，以便提供更个性化的服务。
        
        请提取以下维度的信息：
        1. **身份特征**：姓名 (name)、称呼、职业 (job_title)、所在行业 (industry)。
        2. **能力与偏好**：专业技能 (skills)、兴趣爱好 (interests)、语言风格 (style: 严谨/幽默/简洁)。
        3. **技术上下文**（针对技术问题）：偏好的编程语言 (tech_stack)、使用的软件版本 (version)、部署环境 (env)。
        4. **当前目标**：用户这次对话的核心诉求 (current_goal)。

        请基于【现有画像】和【最新对话】，输出合并更新后的 JSON 数据。
        必须返回纯 JSON 格式。
        """
        #You are a user-profiling analyst. Build a "digital twin" profile to personalise service.

        #Extract these dimensions:
        #1. Identity: name, title, industry
        #2. Skills & preferences: skills, interests, communication style (formal/casual/concise)
        #3. Technical context: preferred language/framework, software version, deployment env
        #4. Current goal: core objective of this conversation

        #Merge the current profile with fresh dialogue and return only valid JSON.
        

        # 4. 构建输入给 LLM 的最终提示词
        human_input = f"""
        【现有画像 (JSON)】:
        {json.dumps(current_profile, ensure_ascii=False)}

        【最新对话】:
        {dialogue_text}

        请输出更新后的用户画像 JSON：
        """

        try:
            # 5. 调用 LLM
            response = self.llm.invoke([
                SystemMessage(content=extract_system_prompt),
                HumanMessage(content=human_input),
            ])
            
            # 6. 清洗数据（防止 LLM 有时候会加 ```json 包裹）
            content = response.content.strip()
            # Strip markdown fences if present
            if content.startswith("```"):
                content = content.split("```", 2)[1]
                if content.startswith("json"):
                    content = content[4:]
            content     = content.rstrip("`").strip()
            # 7. 解析 JSON
            new_profile = json.loads(content)
            
            # 返回更新后的状态
            return {"user_profile": new_profile}

        except Exception as exc:
            logger.warning("画像提取失败: %s", exc)
            # 失败了就返回旧的，保证不报错
            return {"user_profile": current_profile}

    def _summarize(self, state: GraphState):
        summary = state.get("summary", "")
        
        # 只有当消息超过 6 条 (约 3 轮) 时才触发摘要
        if len(state["messages"]) > 6:
            content = f"之前的摘要：{summary}\n\n请简要总结上述对话要点：" if summary else "请简要总结上述对话要点："
            # 调用 LLM 生成新摘要
            response        = self.llm.invoke(state["messages"] + [HumanMessage(content=content)])
            
            # 关键：保留最后 6 条消息，删除其余的 (为了节省 Token)
            # 使用 RemoveMessage 标记需要删除的消息 ID
            delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-6]]
            return {"summary": response.content, "messages": delete_messages}
        return {}

    # --- 构建状态图 ---
    def _build_graph(self):
        workflow = StateGraph(GraphState)
        #添加节点 (顺序与执行逻辑保持一致，方便阅读) ---
        workflow.add_node("profile", self._profile)     # 第一步：提取画像
        workflow.add_node("rewrite", self._rewrite)     # 第二步：改写查询
        workflow.add_node("retrieve", self._retrieve)   # 第三步：检索重排
        workflow.add_node("generate", self._generate)   # 第四步：生成回答
        workflow.add_node("summarize", self._summarize) # 第五步：总结修剪
        
        # 定义边 (Edge)
        workflow.add_edge(START, "profile")
        workflow.add_edge("profile", "rewrite")
        workflow.add_edge("rewrite", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "summarize")
        workflow.add_edge("summarize", END)
        
        # 编译图 (启用 Memory)
        return workflow.compile(checkpointer=MemorySaver())