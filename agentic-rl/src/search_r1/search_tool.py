# search_tool.py
import time
import random
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
from ddgs import DDGS

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    title: str
    snippet: str
    url: str


class SearchTool(ABC):
    """搜索工具的抽象接口"""
    
    @abstractmethod
    def search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        ...


class DuckDuckGoSearchTool(SearchTool):
    """
    基于 duckduckgo_search 库的实现。
    免费、无需 API key、无需本地索引。
    """
    
    def __init__(
        self,
        top_k: int = 3,
        region: str = "wt-wt",       # worldwide
        safesearch: str = "moderate",
        max_retries: int = 3,
        retry_delay: float = 1.0,    # 基础退避时间（秒）
        timeout: int = 10,
        cache_size: int = 1000,
    ):
        self._DDGS = DDGS
        
        self.top_k = top_k
        self.region = region
        self.safesearch = safesearch
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        
        # 简单 LRU 缓存，避免同一 query 重复请求被限流
        self._cache = {}
        self._cache_size = cache_size
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        query = query.strip()
        if not query:
            return []
        
        top_k = top_k or self.top_k
        
        # 1. 命中缓存
        cache_key = (query, top_k)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # 2. 带退避的重试
        results = []
        for attempt in range(self.max_retries):
            try:
                results = self._raw_search(query, top_k)
                break
            except Exception as e:
                logger.warning(
                    f"DuckDuckGo 请求失败 (attempt {attempt+1}/{self.max_retries}): {e}"
                )
                if attempt == self.max_retries - 1:
                    logger.error(f"查询 '{query}' 最终失败，返回空结果")
                    return []
                # 指数退避 + 抖动，避免多个 worker 同时重试
                sleep = self.retry_delay * (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(sleep)
        
        # 3. 写缓存（简单淘汰：超出上限就清空最早的 1/4）
        if len(self._cache) >= self._cache_size:
            for k in list(self._cache.keys())[: self._cache_size // 4]:
                del self._cache[k]
        self._cache[cache_key] = results
        return results
    
    def _raw_search(self, query: str, top_k: int) -> List[SearchResult]:
        """真正调用 DDGS，解析成 SearchResult 列表。"""
        with self._DDGS(
            region=self.region,
            safesearch=self.safesearch,
            timeout=self.timeout,
        ) as ddgs:
            raw = list(ddgs.text(query, max_results=top_k))
        
        results = []
        for item in raw:
            results.append(SearchResult(
                title=item.get("title", "").strip(),
                snippet=item.get("body", "").strip(),
                url=item.get("href", "").strip(),
            ))
        return results