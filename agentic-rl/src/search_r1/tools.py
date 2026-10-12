import re
import time
import random
from dataclasses import dataclass
from typing import List, Any, Dict, Optional
from ddgs import DDGS
from src.infra.environment.tools import Action, Tool, ToolResult, ToolRouter
from src.infra.log import get_logger


logger = get_logger(__name__)


@dataclass
class SearchResult:
    title: str
    snippet: str
    url: str


class SearchTool(Tool):
    """基于 DDGS 的搜索工具。"""

    name = "search"   # 内部唯一标识，同时作为 TagRouter 的 tag

    def __init__(
        self,
        top_k: int = 3,
        region: str = "wt-wt",
        safesearch: str = "moderate",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: int = 10,
        cache_size: int = 1000,
        backend: str = "duckduckgo",
    ):
        self._DDGS = DDGS
        self.top_k = top_k
        self.region = region
        self.safesearch = safesearch
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.backend = backend
        self._cache: Dict[tuple, List[SearchResult]] = {}
        self._cache_size = cache_size

    # ============================================================
    # 【工具层】Tool 接口
    # ============================================================

    def parse_args(self, raw_text: str) -> Dict[str, Any]:
        """把 <search> 内的裸文本解析成 {"query": ...}。"""
        query = raw_text.strip()
        if not query:
            raise RuntimeError("empty query")
        return {"query": query}

    def run(self, args: Dict[str, Any]) -> ToolResult:
        """执行搜索。返回 ToolResult，失败不抛异常。"""
        query = args["query"]
        top_k = args.get("top_k", self.top_k)

        results = self._search_with_retry(query, top_k)
        if results is None:
            return ToolResult(
                success=False,
                content="",
                error=f"search failed for query: {query}",
            )

        content = self._format_results(results)
        return ToolResult(
            success=True,
            content=content,
            metadata={
                "query": query,
                "num_results": len(results),
            },
        )

    def format_observation(self, result: ToolResult) -> str:
        """把 ToolResult 包成 <information>...</information> 文本。"""
        if not result.success:
            return f"<information>Search failed: {result.error}</information>"
        if not result.content:
            return "<information>No results found.</information>"
        return f"<information>{result.content}</information>"

    # ============================================================
    # 内部实现：DDGS 调用 + 缓存 + 重试
    # ============================================================

    def _search_with_retry(
        self, query: str, top_k: int
    ) -> Optional[List[SearchResult]]:
        """带缓存和退避重试。返回 None 表示彻底失败。"""
        query = query.strip()
        if not query:
            return []

        cache_key = (query, top_k)
        if cache_key in self._cache:
            return self._cache[cache_key]

        for attempt in range(self.max_retries):
            try:
                results = self._raw_search(query, top_k)
                # 写缓存（简单淘汰：超出上限清空最早的 1/4）
                if len(self._cache) >= self._cache_size:
                    for k in list(self._cache.keys())[: self._cache_size // 4]:
                        del self._cache[k]
                self._cache[cache_key] = results
                return results
            except Exception as e:
                logger.warning(f"[SearchTool] 请求失败 (attempt {attempt+1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    logger.warning(f"[SearchTool] 查询 '{query}' 最终失败，返回空结果")
                    return None
                sleep = self.retry_delay * (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(sleep)

        return None

    def _raw_search(self, query: str, top_k: int) -> List[SearchResult]:
        with self._DDGS(timeout=self.timeout) as ddgs:
            raw = list(ddgs.text(
                query,
                backend=self.backend,
                region=self.region,
                safesearch=self.safesearch,
                max_results=top_k,
            ))

        return [
            SearchResult(
                title=item.get("title", "").strip(),
                snippet=item.get("body", "").strip(),
                url=item.get("href", "").strip(),
            )
            for item in raw
        ]

    def _format_results(self, results: List[SearchResult]) -> str:
        if not results:
            return ""
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(f"[{i}] {r.title}\n{r.snippet}")
        return "\n\n".join(parts)

_WS_RE = re.compile(r"\s+")

class TagRouter(ToolRouter):
    """
    <search>query</search> / <answer>...</answer> 格式。
    
    tag 与 tool.name 一一对应（简化约定）。
    answer / invalid 是保留 kind，不属于任何工具。
    """

    def __init__(
        self,
        tools: List[Tool],
        answer_kind: str = "answer",
        invalid_kind: str = "invalid",
    ):
        super().__init__(tools)
        self.answer_kind = answer_kind
        self.invalid_kind = invalid_kind
        # tag 就是工具名
        self._tags = list(self._tools_by_name.keys())

    def route(self, text: str) -> Action:
        # 1. 优先检测 answer
        content = self._extract(text, self.answer_kind)
        if content is not None:
            if content:
                return Action(kind=self.answer_kind, is_done=True, raw_text=content)
            return Action(kind=self.invalid_kind, is_done=True)

        # 2. 遍历所有工具 tag
        for tag in self._tags:
            content = self._extract(text, tag)
            if content is not None:
                if content:
                    return Action(kind=tag, is_done=False, raw_text=content)
                return Action(kind=self.invalid_kind, is_done=True)

        # 3. 无匹配
        return Action(kind=self.invalid_kind, is_done=True)

    @staticmethod
    def _extract(text: str, tag: str) -> "str | None":
        """提取 <tag>...</tag> 内内容。未找到返回 None。"""
        m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        if m is None:
            return None
        return _WS_RE.sub(" ", m.group(1)).strip()
