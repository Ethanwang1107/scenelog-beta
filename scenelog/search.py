"""scenelog search — 全文检索"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from scenelog.config import INDEX_FILE, SCENELOG_DIR

logger = logging.getLogger(__name__)


class Searcher:
    """全文检索器 — 在 transcripts_index.jsonl 中搜索。"""

    def __init__(self, source_dir: Path, output_dir: Path | None = None):
        self.source_dir = source_dir
        self.scenelog_dir = output_dir or (source_dir / SCENELOG_DIR)
        self.index_path = self.scenelog_dir / INDEX_FILE

    def search(self, query: str, limit: int = 20) -> list[dict]:
        """在索引中搜索匹配 query 的片段。

        使用子串匹配（不区分大小写）。完全匹配优先，再按文件和时间排序。
        """
        if not self.index_path.exists():
            logger.warning("索引文件不存在: %s", self.index_path)
            return []

        query_lower = query.lower().strip()
        query_terms = query_lower.split()
        results = []

        with open(self.index_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                text = record.get("text", "")
                text_lower = text.lower()
                matches = query_lower in text_lower or (
                    len(query_terms) > 1
                    and all(term in text_lower for term in query_terms)
                )
                if matches:
                    results.append({
                        "file_name": record.get("file_name", ""),
                        "rel_path": record.get("rel_path", ""),
                        "start_time": record.get("start_time", ""),
                        "end_time": record.get("end_time", ""),
                        "text": text,
                        "material_id": record.get("material_id", ""),
                        "source": record.get("source", "audio"),
                        "person_id": record.get("person_id", ""),
                    })

        results.sort(key=lambda r: (
            0 if r["text"].lower() == query_lower else 1,
            r["file_name"],
            r["start_time"],
        ))

        # 去重（相同文件+相同时间+相同文本）
        seen = set()
        unique_results = []
        for r in results:
            key = (r["material_id"], r["source"], r["start_time"], r["text"])
            if key not in seen:
                seen.add(key)
                unique_results.append(r)

        return unique_results[:limit]
