"""Shared pydantic models — the contract between gateway, worker, and clients.

These types are the public API surface: they render the OpenAPI schema on the
gateway and they are what the worker persists to disk / PostgreSQL. Changing a
field here is a breaking API change.
"""
from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(UTC)


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}


class EventType(str, Enum):
    JOB_STARTED = "job_started"
    PAGE_STARTED = "page_started"
    TOKEN = "token"
    PAGE_COMPLETE = "page_complete"
    PROGRESS = "progress"
    JOB_COMPLETE = "job_complete"
    JOB_FAILED = "job_failed"
    JOB_CANCELLED = "job_cancelled"

    @property
    def terminal(self) -> bool:
        return self in {EventType.JOB_COMPLETE, EventType.JOB_FAILED, EventType.JOB_CANCELLED}


class PageSource(str, Enum):
    OCR = "ocr"                # went through the serving stack
    NATIVE = "native"          # PDF had an embedded text layer
    DUPLICATE = "duplicate"    # identical to an earlier page
    BLANK = "blank"


class Region(BaseModel):
    """A layout region (or tile of one) sent to the engine as a single request."""

    bbox: tuple[int, int, int, int]        # x0, y0, x1, y1 in page pixels
    cls: str = "text"                      # text | table | title | figure | list
    order: int = 0                         # reading order within the page
    score: float = 1.0


class PageResult(BaseModel):
    index: int
    text: str = ""
    source: PageSource = PageSource.OCR
    regions: int = 1
    chars: int = 0
    duration_ms: float = 0.0
    ttft_s: float | None = None
    tokens: int = 0
    dpi: int = 0
    width: int = 0
    height: int = 0
    skew_deg: float = 0.0
    duplicate_of: int | None = None   # index of the page this one repeats
    error: str | None = None

    @property
    def skipped(self) -> bool:
        return self.source in {PageSource.BLANK, PageSource.DUPLICATE}


class Artifacts(BaseModel):
    json_path: str | None = None
    markdown_path: str | None = None
    pdf_path: str | None = None


class JobTimings(BaseModel):
    queued_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    queue_wait_s: float | None = None
    process_s: float | None = None


class OCRResult(BaseModel):
    job_id: str
    status: JobStatus
    filename: str = ""
    tenant: str = "default"
    page_count: int = 0
    pages: list[PageResult] = Field(default_factory=list)
    full_text: str = ""
    error: str | None = None
    model: str = ""
    engine: str = ""
    attempts: int = 1
    timings: JobTimings = Field(default_factory=JobTimings)
    artifacts: Artifacts = Field(default_factory=Artifacts)

    @property
    def ocr_pages(self) -> int:
        return sum(1 for p in self.pages if p.source is PageSource.OCR)


class StreamEvent(BaseModel):
    """One item on ``job:{id}:events`` (Redis Stream, replayable via Last-Event-ID)."""

    type: EventType
    job_id: str
    page: int | None = None
    data: str = ""
    seq: int = 0
    ts: datetime = Field(default_factory=utcnow)
    #: Redis stream id, filled in on read. Doubles as the SSE event id.
    id: str | None = None


class JobAccepted(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.QUEUED
    stream_url: str
    ws_url: str
    result_url: str


class JobState(BaseModel):
    """Live view of the ``job:{id}`` hash — what ``GET /v1/ocr/{id}`` returns pre-completion."""

    job_id: str
    status: JobStatus
    filename: str = ""
    tenant: str = "default"
    attempts: int = 0
    pages_done: int = 0
    page_count: int = 0
    error: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @property
    def progress(self) -> float:
        return round(self.pages_done / self.page_count, 3) if self.page_count else 0.0


class ErrorResponse(BaseModel):
    """RFC 7807-flavoured error body used by every gateway failure path."""

    error: str
    detail: str = ""
    request_id: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    checks: dict[str, str] = Field(default_factory=dict)
