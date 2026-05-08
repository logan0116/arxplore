from typing import Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    keyword: Optional[str] = None
    semantic_query: Optional[str] = None
    categories: Optional[list[str]] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    sort_by: str = "relevance"
    sort_order: str = "desc"
    limit: int = Field(default=20, ge=1, le=200)
    offset: int = Field(default=0, ge=0)


class Paper(BaseModel):
    arxiv_id: str
    title: str
    authors: str
    abstract: str
    categories: str
    published_date: str
    updated_date: Optional[str] = None
    pdf_url: str
    abs_url: str
    source: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class SearchResponse(BaseModel):
    total: int
    papers: list[Paper]


class StatsResponse(BaseModel):
    total: int
    distribution: dict
    last_updated: Optional[str] = None


class TriggerFetchResponse(BaseModel):
    status: str
    message: str
    papers_fetched: Optional[int] = None