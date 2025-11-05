"""
Redis-backed enrichment cache for performance optimization.

Provides caching layer for enrichment results to reduce redundant
KB queries and improve response times.

Target Performance:
- Cache hit rate: >60%
- Cache lookup: <5ms
- TTL: 24 hours default
"""

import hashlib
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import asyncio
import structlog

import redis.asyncio as redis

from src.rag.schemas import EnrichmentResponse, CacheEntry

logger = structlog.get_logger(__name__)


class EnrichmentCache:
    """
    Redis-backed cache for enrichment results.

    Features:
    - Async Redis operations
    - Automatic TTL management
    - Cache invalidation by KB type
    - Access counting for analytics
    - Configurable expiration

    Usage:
        cache = EnrichmentCache(redis_url="redis://localhost:6379")
        await cache.initialize()

        # Store enrichment
        await cache.set(claim_data, enrichment_response)

        # Retrieve enrichment
        cached = await cache.get(claim_data)

        # Cleanup
        await cache.close()
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        default_ttl_seconds: int = 86400,  # 24 hours
        key_prefix: str = "enrichment:",
        max_retries: int = 3
    ):
        """
        Initialize enrichment cache.

        Args:
            redis_url: Redis connection URL
            default_ttl_seconds: Default TTL for cache entries
            key_prefix: Prefix for all cache keys
            max_retries: Maximum connection retry attempts
        """
        self.redis_url = redis_url
        self.default_ttl_seconds = default_ttl_seconds
        self.key_prefix = key_prefix
        self.max_retries = max_retries
        self._client: Optional[redis.Redis] = None
        self._cache_hits = 0
        self._cache_misses = 0

    async def initialize(self) -> None:
        """Initialize Redis connection."""
        try:
            self._client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5.0,
                socket_timeout=5.0
            )
            # Test connection
            await self._client.ping()
            logger.info("enrichment_cache_initialized", redis_url=self.redis_url)
        except Exception as e:
            logger.error("enrichment_cache_init_failed", error=str(e))
            raise

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.aclose()
            logger.info("enrichment_cache_closed")

    def _hash_claim(self, claim_data: Dict[str, Any]) -> str:
        """
        Generate consistent hash for claim data.

        Args:
            claim_data: Claim data dictionary

        Returns:
            SHA256 hash of normalized claim data
        """
        # Normalize and sort keys for consistent hashing
        normalized = json.dumps(claim_data, sort_keys=True, default=str)
        claim_hash = hashlib.sha256(normalized.encode()).hexdigest()
        return f"{self.key_prefix}{claim_hash}"

    async def get(self, claim_data: Dict[str, Any]) -> Optional[EnrichmentResponse]:
        """
        Retrieve cached enrichment result.

        Args:
            claim_data: Original claim data

        Returns:
            Cached EnrichmentResponse if found, None otherwise
        """
        if not self._client:
            logger.warning("cache_get_no_client")
            return None

        cache_key = self._hash_claim(claim_data)

        try:
            cached_json = await self._client.get(cache_key)

            if cached_json:
                self._cache_hits += 1

                # Parse cached entry
                cached_dict = json.loads(cached_json)
                cache_entry = CacheEntry(**cached_dict)

                # Increment access count
                cache_entry.access_count += 1
                await self._client.set(
                    cache_key,
                    cache_entry.model_dump_json(),
                    ex=self.default_ttl_seconds
                )

                logger.info(
                    "cache_hit",
                    cache_key=cache_key[:16],
                    access_count=cache_entry.access_count
                )
                return cache_entry.enrichment_response
            else:
                self._cache_misses += 1
                logger.debug("cache_miss", cache_key=cache_key[:16])
                return None

        except Exception as e:
            logger.error("cache_get_error", error=str(e), cache_key=cache_key[:16])
            return None

    async def set(
        self,
        claim_data: Dict[str, Any],
        enrichment_response: EnrichmentResponse,
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """
        Store enrichment result in cache.

        Args:
            claim_data: Original claim data
            enrichment_response: Enrichment result to cache
            ttl_seconds: Custom TTL (uses default if None)

        Returns:
            True if cached successfully, False otherwise
        """
        if not self._client:
            logger.warning("cache_set_no_client")
            return False

        cache_key = self._hash_claim(claim_data)
        ttl = ttl_seconds or self.default_ttl_seconds

        try:
            # Create cache entry
            cache_entry = CacheEntry(
                claim_hash=cache_key,
                enrichment_response=enrichment_response,
                cached_at=datetime.utcnow(),
                access_count=0,
                ttl_seconds=ttl
            )

            # Store in Redis
            await self._client.set(
                cache_key,
                cache_entry.model_dump_json(),
                ex=ttl
            )

            logger.info("cache_set", cache_key=cache_key[:16], ttl_seconds=ttl)
            return True

        except Exception as e:
            logger.error("cache_set_error", error=str(e), cache_key=cache_key[:16])
            return False

    async def invalidate(self, claim_data: Dict[str, Any]) -> bool:
        """
        Invalidate specific cache entry.

        Args:
            claim_data: Claim data to invalidate

        Returns:
            True if invalidated, False otherwise
        """
        if not self._client:
            return False

        cache_key = self._hash_claim(claim_data)

        try:
            deleted = await self._client.delete(cache_key)
            if deleted:
                logger.info("cache_invalidated", cache_key=cache_key[:16])
            return bool(deleted)
        except Exception as e:
            logger.error("cache_invalidate_error", error=str(e))
            return False

    async def invalidate_by_kb_update(self, kb_type: str) -> int:
        """
        Invalidate all cache entries when a KB is updated.

        This is a full cache flush since we can't track which entries
        came from which KB without additional metadata.

        Args:
            kb_type: Knowledge base type that was updated

        Returns:
            Number of keys invalidated
        """
        if not self._client:
            return 0

        try:
            # Pattern match all enrichment cache keys
            pattern = f"{self.key_prefix}*"
            cursor = 0
            invalidated_count = 0

            # Scan and delete in batches
            async for key in self._client.scan_iter(match=pattern, count=100):
                await self._client.delete(key)
                invalidated_count += 1

            logger.info(
                "cache_kb_invalidation",
                kb_type=kb_type,
                invalidated_count=invalidated_count
            )
            return invalidated_count

        except Exception as e:
            logger.error("cache_kb_invalidation_error", error=str(e))
            return 0

    async def clear_all(self) -> bool:
        """
        Clear all enrichment cache entries.

        Returns:
            True if successful, False otherwise
        """
        if not self._client:
            return False

        try:
            pattern = f"{self.key_prefix}*"
            cursor = 0

            async for key in self._client.scan_iter(match=pattern, count=100):
                await self._client.delete(key)

            logger.info("cache_cleared")
            return True

        except Exception as e:
            logger.error("cache_clear_error", error=str(e))
            return False

    def get_hit_rate(self) -> float:
        """
        Calculate cache hit rate.

        Returns:
            Hit rate as percentage [0.0, 1.0]
        """
        total = self._cache_hits + self._cache_misses
        if total == 0:
            return 0.0
        return self._cache_hits / total

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        stats = {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": self.get_hit_rate(),
            "total_requests": self._cache_hits + self._cache_misses,
        }

        if self._client:
            try:
                # Get Redis info
                info = await self._client.info("stats")
                stats["redis_connected_clients"] = info.get("connected_clients", 0)
                stats["redis_total_commands_processed"] = info.get("total_commands_processed", 0)

                # Count enrichment keys
                pattern = f"{self.key_prefix}*"
                key_count = 0
                async for _ in self._client.scan_iter(match=pattern, count=100):
                    key_count += 1
                stats["cached_enrichments"] = key_count

            except Exception as e:
                logger.error("cache_stats_error", error=str(e))

        return stats
