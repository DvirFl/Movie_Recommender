"""SQLAlchemy ORM models for all PostgreSQL schemas.

Schemas:
  raw      — ingested MovieLens files
  features — engineered features + split indices
  serving  — pre-computed recommendations
  pipeline — trigger event log + watermarks
"""
from __future__ import annotations

import datetime as dt
from typing import Optional

from sqlalchemy import (
    BigInteger, Boolean, DateTime, Float, ForeignKey, Integer,
    String, Text, UniqueConstraint, func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Schema: raw
# ---------------------------------------------------------------------------

class RawRating(Base):
    __tablename__ = "ratings"
    __table_args__ = (
        UniqueConstraint("user_id", "movie_id", name="uq_raw_ratings_user_movie"),
        {"schema": "raw"},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    movie_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    rating: Mapped[float] = mapped_column(Float, nullable=False)
    timestamp: Mapped[int] = mapped_column(BigInteger, nullable=False)
    inserted_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class RawMovie(Base):
    __tablename__ = "movies"
    __table_args__ = {"schema": "raw"}

    movie_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    genres: Mapped[str] = mapped_column(Text, nullable=False)
    inserted_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class RawTag(Base):
    __tablename__ = "tags"
    __table_args__ = (
        UniqueConstraint("user_id", "movie_id", "tag", "timestamp", name="uq_raw_tags_user_movie_tag_time"),
        {"schema": "raw"},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False)
    movie_id: Mapped[int] = mapped_column(Integer, nullable=False)
    tag: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[int] = mapped_column(BigInteger, nullable=False)
    inserted_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class RawLink(Base):
    __tablename__ = "links"
    __table_args__ = {"schema": "raw"}

    movie_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    imdb_id: Mapped[Optional[str]] = mapped_column(String(16))
    tmdb_id: Mapped[Optional[str]] = mapped_column(String(16))
    inserted_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class RawGenomeScore(Base):
    __tablename__ = "genome_scores"
    __table_args__ = (
        UniqueConstraint("movie_id", "tag_id"),
        {"schema": "raw"},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    movie_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    tag_id: Mapped[int] = mapped_column(Integer, nullable=False)
    relevance: Mapped[float] = mapped_column(Float, nullable=False)
    inserted_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


# ---------------------------------------------------------------------------
# Schema: features
# ---------------------------------------------------------------------------

class UserFeature(Base):
    __tablename__ = "user_features"
    __table_args__ = {"schema": "features"}

    user_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    decayed_history: Mapped[dict] = mapped_column(JSONB, nullable=False)
    # {movie_id: decayed_weight, ...}
    genre_affinity: Mapped[dict] = mapped_column(JSONB, nullable=False)
    # {genre: score, ...}
    rating_count: Mapped[int] = mapped_column(Integer, nullable=False)
    avg_rating: Mapped[float] = mapped_column(Float, nullable=False)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class ItemFeature(Base):
    __tablename__ = "item_features"
    __table_args__ = {"schema": "features"}

    movie_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    genre_multihot: Mapped[list] = mapped_column(ARRAY(Float), nullable=False)
    tag_tfidf: Mapped[dict] = mapped_column(JSONB, nullable=False)
    release_year: Mapped[Optional[int]] = mapped_column(Integer)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class SplitIndex(Base):
    __tablename__ = "split_indices"
    __table_args__ = {"schema": "features"}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    rating_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    split: Mapped[str] = mapped_column(String(8), nullable=False)
    # 'train' | 'val' | 'test'
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# ---------------------------------------------------------------------------
# Schema: serving
# ---------------------------------------------------------------------------

class TopNUserGenre(Base):
    __tablename__ = "top_n_user_genre"
    __table_args__ = (
        UniqueConstraint("user_id", "genre", "model_name", "scoring_method"),
        {"schema": "serving"},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    genre: Mapped[str] = mapped_column(String(64), nullable=False)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    scoring_method: Mapped[str] = mapped_column(String(32), nullable=False)
    movie_ids: Mapped[list] = mapped_column(ARRAY(Integer), nullable=False)
    scores: Mapped[list] = mapped_column(ARRAY(Float), nullable=False)
    computed_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class ColdStartGenre(Base):
    __tablename__ = "cold_start_genre"
    __table_args__ = (
        UniqueConstraint("genre", "model_name", "scoring_method"),
        {"schema": "serving"},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    genre: Mapped[str] = mapped_column(String(64), nullable=False)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    scoring_method: Mapped[str] = mapped_column(String(32), nullable=False)
    movie_ids: Mapped[list] = mapped_column(ARRAY(Integer), nullable=False)
    scores: Mapped[list] = mapped_column(ARRAY(Float), nullable=False)
    computed_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# ---------------------------------------------------------------------------
# Schema: pipeline
# ---------------------------------------------------------------------------

class TriggerWatermark(Base):
    __tablename__ = "trigger_watermarks"
    __table_args__ = {"schema": "pipeline"}

    table_name: Mapped[str] = mapped_column(String(64), primary_key=True)
    last_inserted_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class TriggerLog(Base):
    __tablename__ = "trigger_log"
    __table_args__ = {"schema": "pipeline"}

    trigger_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    trigger_type: Mapped[str] = mapped_column(String(16), nullable=False)
    # 'data' | 'schedule' | 'on_demand'
    requested_losses: Mapped[Optional[list]] = mapped_column(ARRAY(String))
    requested_architectures: Mapped[Optional[list]] = mapped_column(ARRAY(String))
    airflow_run_id: Mapped[Optional[str]] = mapped_column(String(256))
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    # 'pending' | 'running' | 'success' | 'failed'
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    completed_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True))
    requester: Mapped[Optional[str]] = mapped_column(String(128))
