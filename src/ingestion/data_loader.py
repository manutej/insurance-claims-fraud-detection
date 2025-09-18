"""
Main data loading module for insurance claims.

Provides comprehensive data loading capabilities including batch processing,
streaming, different claim types, and robust error handling.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator, AsyncGenerator, Union, Callable
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import aiofiles
from pydantic import ValidationError as PydanticValidationError

from ..models.claim_models import (
    BaseClaim, MedicalClaim, PharmacyClaim, NoFaultClaim,
    ClaimBatch, ProcessingResult, ValidationError,
    claim_factory
)
from .validator import ClaimValidator
from .preprocessor import ClaimPreprocessor

logger = logging.getLogger(__name__)


class DataLoaderConfig:
    """Configuration for data loader."""

    def __init__(
        self,
        batch_size: int = 1000,
        max_workers: int = 4,
        validate_on_load: bool = True,
        preprocess_on_load: bool = False,
        chunk_size: int = 10000,
        max_file_size_mb: int = 500,
        supported_extensions: List[str] = None
    ):
        """
        Initialize data loader configuration.

        Args:
            batch_size: Number of claims to process in each batch
            max_workers: Maximum number of worker threads
            validate_on_load: Whether to validate claims during loading
            preprocess_on_load: Whether to preprocess claims during loading
            chunk_size: Size of chunks for streaming
            max_file_size_mb: Maximum file size in MB
            supported_extensions: List of supported file extensions
        """
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.validate_on_load = validate_on_load
        self.preprocess_on_load = preprocess_on_load
        self.chunk_size = chunk_size
        self.max_file_size_mb = max_file_size_mb
        self.supported_extensions = supported_extensions or ['.json']


class ClaimDataLoader:
    """Main data loader for insurance claims with batch and streaming support."""

    def __init__(
        self,
        data_directory: Union[str, Path],
        config: Optional[DataLoaderConfig] = None,
        validator: Optional[ClaimValidator] = None,
        preprocessor: Optional[ClaimPreprocessor] = None
    ):
        """
        Initialize data loader.

        Args:
            data_directory: Path to data directory
            config: Loader configuration
            validator: Optional validator instance
            preprocessor: Optional preprocessor instance
        """
        self.data_directory = Path(data_directory)
        self.config = config or DataLoaderConfig()
        self.validator = validator or ClaimValidator()
        self.preprocessor = preprocessor or ClaimPreprocessor()

        self.stats = {
            'files_processed': 0,
            'claims_loaded': 0,
            'validation_errors': 0,
            'processing_time': 0.0
        }

        # Validate data directory
        if not self.data_directory.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_directory}")

    def load_claims_batch(
        self,
        file_paths: Optional[List[Union[str, Path]]] = None,
        claim_types: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> ClaimBatch:
        """
        Load claims in batch mode.

        Args:
            file_paths: Specific file paths to load (if None, loads all)
            claim_types: Filter by claim types
            progress_callback: Optional callback for progress updates

        Returns:
            ClaimBatch with loaded claims

        Raises:
            ValueError: If no valid files found
            FileNotFoundError: If specified files don't exist
        """
        start_time = time.time()
        logger.info("Starting batch claim loading")

        # Get file paths
        if file_paths is None:
            file_paths = self._discover_claim_files(claim_types)
        else:
            file_paths = [Path(fp) for fp in file_paths]
            # Validate file paths
            for file_path in file_paths:
                if not file_path.exists():
                    raise FileNotFoundError(f"File not found: {file_path}")

        if not file_paths:
            raise ValueError("No valid claim files found")

        logger.info(f"Found {len(file_paths)} files to process")

        # Load claims from files
        all_claims = []
        total_files = len(file_paths)

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit file loading tasks
            future_to_file = {
                executor.submit(self._load_single_file, file_path): file_path
                for file_path in file_paths
            }

            # Process completed tasks
            for i, future in enumerate(as_completed(future_to_file)):
                file_path = future_to_file[future]
                try:
                    claims = future.result()
                    all_claims.extend(claims)
                    self.stats['files_processed'] += 1

                    logger.debug(f"Loaded {len(claims)} claims from {file_path.name}")

                    # Progress callback
                    if progress_callback:
                        progress_callback(i + 1, total_files)

                except Exception as e:
                    logger.error(f"Error loading file {file_path}: {e}")
                    continue

        # Create batch
        batch_id = f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        batch = ClaimBatch(
            claims=all_claims,
            batch_id=batch_id,
            processed_at=datetime.utcnow(),
            total_count=len(all_claims)
        )

        # Validation
        if self.config.validate_on_load:
            logger.info("Validating loaded claims")
            validation_result = self._validate_batch(batch)
            self.stats['validation_errors'] = validation_result.error_count

        self.stats['claims_loaded'] = len(all_claims)
        self.stats['processing_time'] = time.time() - start_time

        logger.info(f"Batch loading complete: {len(all_claims)} claims loaded in {self.stats['processing_time']:.2f}s")
        return batch

    def stream_claims(
        self,
        file_paths: Optional[List[Union[str, Path]]] = None,
        claim_types: Optional[List[str]] = None,
        chunk_size: Optional[int] = None
    ) -> Generator[List[BaseClaim], None, None]:
        """
        Stream claims in chunks.

        Args:
            file_paths: Specific file paths to stream
            claim_types: Filter by claim types
            chunk_size: Size of each chunk

        Yields:
            Lists of claims in chunks

        Raises:
            ValueError: If no valid files found
        """
        chunk_size = chunk_size or self.config.chunk_size
        logger.info(f"Starting claim streaming with chunk size {chunk_size}")

        # Get file paths
        if file_paths is None:
            file_paths = self._discover_claim_files(claim_types)
        else:
            file_paths = [Path(fp) for fp in file_paths]

        if not file_paths:
            raise ValueError("No valid claim files found")

        current_chunk = []

        for file_path in file_paths:
            try:
                claims = self._load_single_file(file_path)

                for claim in claims:
                    current_chunk.append(claim)

                    if len(current_chunk) >= chunk_size:
                        yield current_chunk
                        current_chunk = []

            except Exception as e:
                logger.error(f"Error streaming file {file_path}: {e}")
                continue

        # Yield remaining claims
        if current_chunk:
            yield current_chunk

    async def stream_claims_async(
        self,
        file_paths: Optional[List[Union[str, Path]]] = None,
        claim_types: Optional[List[str]] = None,
        chunk_size: Optional[int] = None
    ) -> AsyncGenerator[List[BaseClaim], None]:
        """
        Asynchronously stream claims in chunks.

        Args:
            file_paths: Specific file paths to stream
            claim_types: Filter by claim types
            chunk_size: Size of each chunk

        Yields:
            Lists of claims in chunks
        """
        chunk_size = chunk_size or self.config.chunk_size
        logger.info(f"Starting async claim streaming with chunk size {chunk_size}")

        # Get file paths
        if file_paths is None:
            file_paths = self._discover_claim_files(claim_types)
        else:
            file_paths = [Path(fp) for fp in file_paths]

        if not file_paths:
            raise ValueError("No valid claim files found")

        current_chunk = []

        for file_path in file_paths:
            try:
                claims = await self._load_single_file_async(file_path)

                for claim in claims:
                    current_chunk.append(claim)

                    if len(current_chunk) >= chunk_size:
                        yield current_chunk
                        current_chunk = []

                # Allow other coroutines to run
                await asyncio.sleep(0)

            except Exception as e:
                logger.error(f"Error streaming file {file_path}: {e}")
                continue

        # Yield remaining claims
        if current_chunk:
            yield current_chunk

    def load_specific_claim_type(
        self,
        claim_type: str,
        subdirectory: Optional[str] = None
    ) -> List[BaseClaim]:
        """
        Load claims of a specific type.

        Args:
            claim_type: Type of claims to load ('medical', 'pharmacy', 'no_fault')
            subdirectory: Optional subdirectory to search in

        Returns:
            List of claims of specified type

        Raises:
            ValueError: If claim type is unsupported
        """
        logger.info(f"Loading {claim_type} claims")

        # Map claim types to file patterns
        type_patterns = {
            'medical': ['*medical*.json', '*professional*.json'],
            'pharmacy': ['*pharmacy*.json', '*drug*.json'],
            'no_fault': ['*no_fault*.json', '*auto*.json'],
            'fraud': ['*fraud*.json'],
            'valid': ['*valid*.json']
        }

        if claim_type not in type_patterns:
            raise ValueError(f"Unsupported claim type: {claim_type}")

        # Build search directory
        search_dir = self.data_directory
        if subdirectory:
            search_dir = search_dir / subdirectory

        # Find matching files
        file_paths = []
        for pattern in type_patterns[claim_type]:
            file_paths.extend(search_dir.glob(pattern))

        if not file_paths:
            logger.warning(f"No {claim_type} claim files found in {search_dir}")
            return []

        # Load claims
        all_claims = []
        for file_path in file_paths:
            try:
                claims = self._load_single_file(file_path)
                all_claims.extend(claims)
            except Exception as e:
                logger.error(f"Error loading {claim_type} claims from {file_path}: {e}")

        logger.info(f"Loaded {len(all_claims)} {claim_type} claims")
        return all_claims

    def _discover_claim_files(self, claim_types: Optional[List[str]] = None) -> List[Path]:
        """Discover claim files in the data directory."""
        file_paths = []

        # Search patterns
        if claim_types:
            patterns = []
            for claim_type in claim_types:
                if claim_type == 'medical':
                    patterns.extend(['*medical*.json', '*professional*.json'])
                elif claim_type == 'pharmacy':
                    patterns.extend(['*pharmacy*.json', '*drug*.json'])
                elif claim_type == 'no_fault':
                    patterns.extend(['*no_fault*.json', '*auto*.json'])
                else:
                    patterns.append(f'*{claim_type}*.json')
        else:
            patterns = ['**/*.json']

        # Find files
        for pattern in patterns:
            for file_path in self.data_directory.glob(pattern):
                if file_path.is_file() and file_path.suffix in self.config.supported_extensions:
                    # Check file size
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    if size_mb <= self.config.max_file_size_mb:
                        file_paths.append(file_path)
                    else:
                        logger.warning(f"Skipping large file {file_path} ({size_mb:.1f}MB)")

        return list(set(file_paths))  # Remove duplicates

    def _load_single_file(self, file_path: Path) -> List[BaseClaim]:
        """Load claims from a single file."""
        logger.debug(f"Loading file: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle different file structures
            if isinstance(data, dict) and 'claims' in data:
                claims_data = data['claims']
            elif isinstance(data, list):
                claims_data = data
            else:
                logger.warning(f"Unexpected file structure in {file_path}")
                return []

            # Convert to claim objects
            claims = []
            for claim_data in claims_data:
                try:
                    claim = claim_factory(claim_data)
                    claims.append(claim)
                except Exception as e:
                    logger.warning(f"Error creating claim from data in {file_path}: {e}")
                    continue

            return claims

        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            raise

    async def _load_single_file_async(self, file_path: Path) -> List[BaseClaim]:
        """Asynchronously load claims from a single file."""
        logger.debug(f"Async loading file: {file_path}")

        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                data = json.loads(content)

            # Handle different file structures
            if isinstance(data, dict) and 'claims' in data:
                claims_data = data['claims']
            elif isinstance(data, list):
                claims_data = data
            else:
                logger.warning(f"Unexpected file structure in {file_path}")
                return []

            # Convert to claim objects
            claims = []
            for claim_data in claims_data:
                try:
                    claim = claim_factory(claim_data)
                    claims.append(claim)
                except Exception as e:
                    logger.warning(f"Error creating claim from data in {file_path}: {e}")
                    continue

            return claims

        except Exception as e:
            logger.error(f"Error async loading file {file_path}: {e}")
            raise

    def _validate_batch(self, batch: ClaimBatch) -> ProcessingResult:
        """Validate a batch of claims."""
        # Convert claims to dictionaries for validation
        claims_data = []
        for claim in batch.claims:
            claim_dict = claim.dict()
            claims_data.append(claim_dict)

        return self.validator.validate_batch(claims_data)

    def get_statistics(self) -> Dict[str, Any]:
        """Get loader statistics."""
        return {
            'files_processed': self.stats['files_processed'],
            'claims_loaded': self.stats['claims_loaded'],
            'validation_errors': self.stats['validation_errors'],
            'processing_time_seconds': self.stats['processing_time'],
            'avg_claims_per_file': (
                self.stats['claims_loaded'] / self.stats['files_processed']
                if self.stats['files_processed'] > 0 else 0
            ),
            'claims_per_second': (
                self.stats['claims_loaded'] / self.stats['processing_time']
                if self.stats['processing_time'] > 0 else 0
            )
        }

    def reset_statistics(self):
        """Reset loader statistics."""
        self.stats = {
            'files_processed': 0,
            'claims_loaded': 0,
            'validation_errors': 0,
            'processing_time': 0.0
        }

    def get_file_summary(self) -> Dict[str, Any]:
        """Get summary of available files."""
        files = self._discover_claim_files()
        summary = {
            'total_files': len(files),
            'files_by_type': {},
            'total_size_mb': 0,
            'file_details': []
        }

        for file_path in files:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            summary['total_size_mb'] += size_mb

            # Determine file type
            file_type = 'unknown'
            name_lower = file_path.name.lower()
            if 'medical' in name_lower or 'professional' in name_lower:
                file_type = 'medical'
            elif 'pharmacy' in name_lower or 'drug' in name_lower:
                file_type = 'pharmacy'
            elif 'no_fault' in name_lower or 'auto' in name_lower:
                file_type = 'no_fault'
            elif 'fraud' in name_lower:
                file_type = 'fraud'
            elif 'valid' in name_lower:
                file_type = 'valid'

            summary['files_by_type'][file_type] = summary['files_by_type'].get(file_type, 0) + 1

            summary['file_details'].append({
                'path': str(file_path),
                'name': file_path.name,
                'type': file_type,
                'size_mb': round(size_mb, 2),
                'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            })

        return summary


# Convenience functions

def load_claims_from_directory(
    data_directory: Union[str, Path],
    claim_types: Optional[List[str]] = None,
    validate: bool = True
) -> ClaimBatch:
    """
    Convenience function to load claims from a directory.

    Args:
        data_directory: Path to data directory
        claim_types: Optional list of claim types to filter
        validate: Whether to validate claims

    Returns:
        ClaimBatch with loaded claims
    """
    config = DataLoaderConfig(validate_on_load=validate)
    loader = ClaimDataLoader(data_directory, config)
    return loader.load_claims_batch(claim_types=claim_types)


def stream_claims_from_directory(
    data_directory: Union[str, Path],
    chunk_size: int = 1000,
    claim_types: Optional[List[str]] = None
) -> Generator[List[BaseClaim], None, None]:
    """
    Convenience function to stream claims from a directory.

    Args:
        data_directory: Path to data directory
        chunk_size: Size of each chunk
        claim_types: Optional list of claim types to filter

    Yields:
        Lists of claims in chunks
    """
    config = DataLoaderConfig(chunk_size=chunk_size)
    loader = ClaimDataLoader(data_directory, config)
    yield from loader.stream_claims(claim_types=claim_types, chunk_size=chunk_size)