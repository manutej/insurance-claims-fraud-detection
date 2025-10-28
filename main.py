#!/usr/bin/env python3
"""
Main entry point for the Insurance Claims Data Ingestion Pipeline.

This script provides a command-line interface for loading, validating, and
preprocessing insurance claims data. It supports both batch and streaming modes
with comprehensive error handling and reporting.
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import click
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.logging import RichHandler

# Import our modules
from src.ingestion import ClaimDataLoader, DataLoaderConfig, ClaimValidator, ClaimPreprocessor
from src.models import ClaimBatch, ProcessingResult

# Setup rich console for better output
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger("claims_pipeline")


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--data-dir", "-d", default="data", help="Data directory path")
@click.pass_context
def cli(ctx, verbose, data_dir):
    """Insurance Claims Data Ingestion Pipeline."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["data_dir"] = Path(data_dir)

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")


@cli.command()
@click.option(
    "--claim-types", "-t", multiple=True, help="Claim types to load (medical, pharmacy, no_fault)"
)
@click.option("--batch-size", "-b", default=1000, help="Batch size for processing")
@click.option(
    "--validate", "-val", is_flag=True, default=True, help="Validate claims during loading"
)
@click.option("--preprocess", "-prep", is_flag=True, help="Preprocess claims for ML")
@click.option("--output", "-o", help="Output file path for processed data")
@click.option(
    "--format",
    "-f",
    type=click.Choice(["json", "csv", "parquet"]),
    default="json",
    help="Output format",
)
@click.pass_context
def load(ctx, claim_types, batch_size, validate, preprocess, output, format):
    """Load claims in batch mode."""
    data_dir = ctx.obj["data_dir"]

    console.print(f"[bold green]Loading claims from:[/bold green] {data_dir}")

    if claim_types:
        console.print(f"[bold blue]Claim types:[/bold blue] {', '.join(claim_types)}")

    # Configure loader
    config = DataLoaderConfig(
        batch_size=batch_size, validate_on_load=validate, preprocess_on_load=preprocess
    )

    # Initialize components
    validator = ClaimValidator() if validate else None
    preprocessor = ClaimPreprocessor() if preprocess else None
    loader = ClaimDataLoader(data_dir, config, validator, preprocessor)

    # Progress callback
    def progress_callback(current: int, total: int):
        percentage = (current / total) * 100
        console.print(f"Progress: {current}/{total} files ({percentage:.1f}%)")

    try:
        # Load claims
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:

            task = progress.add_task("Loading claims...", total=100)

            # Get file summary first
            file_summary = loader.get_file_summary()
            console.print(
                f"Found {file_summary['total_files']} files ({file_summary['total_size_mb']:.1f} MB)"
            )

            batch = loader.load_claims_batch(
                claim_types=list(claim_types) if claim_types else None,
                progress_callback=lambda c, t: progress.update(task, completed=(c / t) * 100),
            )

            progress.update(task, completed=100)

        # Display results
        _display_batch_summary(batch, loader.get_statistics())

        # Save output if requested
        if output:
            _save_batch_output(batch, output, format, preprocess, preprocessor)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


@cli.command()
@click.option("--claim-types", "-t", multiple=True, help="Claim types to stream")
@click.option("--chunk-size", "-c", default=1000, help="Chunk size for streaming")
@click.option("--max-chunks", "-m", default=10, help="Maximum chunks to process (0 for unlimited)")
@click.option("--validate", is_flag=True, help="Validate chunks during streaming")
@click.pass_context
def stream(ctx, claim_types, chunk_size, max_chunks, validate):
    """Stream claims in chunks."""
    data_dir = ctx.obj["data_dir"]

    console.print(f"[bold green]Streaming claims from:[/bold green] {data_dir}")
    console.print(f"[bold blue]Chunk size:[/bold blue] {chunk_size}")

    # Configure loader
    config = DataLoaderConfig(chunk_size=chunk_size, validate_on_load=validate)

    validator = ClaimValidator() if validate else None
    loader = ClaimDataLoader(data_dir, config, validator)

    try:
        chunk_count = 0
        total_claims = 0

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.completed:>3.0f} chunks"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:

            task = progress.add_task(
                "Streaming claims...", total=max_chunks if max_chunks > 0 else None
            )

            for chunk in loader.stream_claims(
                claim_types=list(claim_types) if claim_types else None, chunk_size=chunk_size
            ):
                chunk_count += 1
                total_claims += len(chunk)

                console.print(f"Processed chunk {chunk_count}: {len(chunk)} claims")

                if validate:
                    # Validate chunk
                    chunk_data = [claim.dict() for claim in chunk]
                    validation_result = validator.validate_batch(chunk_data)

                    if validation_result.error_count > 0:
                        console.print(
                            f"[yellow]Validation errors in chunk {chunk_count}: {validation_result.error_count}[/yellow]"
                        )

                progress.update(task, advance=1)

                if max_chunks > 0 and chunk_count >= max_chunks:
                    break

        console.print(
            f"[bold green]Streaming complete:[/bold green] {chunk_count} chunks, {total_claims} total claims"
        )

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


@cli.command()
@click.option("--file-path", "-f", required=True, help="Path to claims file to validate")
@click.option("--detailed", "-det", is_flag=True, help="Show detailed validation results")
@click.pass_context
def validate(ctx, file_path, detailed):
    """Validate a specific claims file."""
    file_path = Path(file_path)

    if not file_path.exists():
        console.print(f"[bold red]Error:[/bold red] File not found: {file_path}")
        sys.exit(1)

    console.print(f"[bold green]Validating:[/bold green] {file_path}")

    # Load and validate
    try:
        loader = ClaimDataLoader(file_path.parent)
        claims = loader._load_single_file(file_path)

        validator = ClaimValidator()
        claims_data = [claim.dict() for claim in claims]
        result = validator.validate_batch(claims_data)

        _display_validation_results(result, detailed)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


@cli.command()
@click.pass_context
def info(ctx):
    """Display information about available data files."""
    data_dir = ctx.obj["data_dir"]

    if not data_dir.exists():
        console.print(f"[bold red]Error:[/bold red] Data directory not found: {data_dir}")
        sys.exit(1)

    console.print(f"[bold green]Data Directory:[/bold green] {data_dir}")

    try:
        loader = ClaimDataLoader(data_dir)
        summary = loader.get_file_summary()

        # Create summary table
        table = Table(title="Data Files Summary")
        table.add_column("Type", style="cyan")
        table.add_column("Count", justify="right", style="magenta")
        table.add_column("Size (MB)", justify="right", style="green")

        total_size = 0
        for file_type, count in summary["files_by_type"].items():
            type_size = sum(
                detail["size_mb"]
                for detail in summary["file_details"]
                if detail["type"] == file_type
            )
            table.add_row(file_type.title(), str(count), f"{type_size:.1f}")
            total_size += type_size

        table.add_row("TOTAL", str(summary["total_files"]), f"{total_size:.1f}", style="bold")

        console.print(table)

        # Detailed file list if requested
        if ctx.obj.get("verbose"):
            console.print("\n[bold blue]File Details:[/bold blue]")
            for detail in summary["file_details"]:
                console.print(f"  {detail['name']} ({detail['type']}) - {detail['size_mb']} MB")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


def _display_batch_summary(batch: ClaimBatch, stats: dict):
    """Display batch loading summary."""
    table = Table(title="Batch Loading Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="magenta")

    table.add_row("Total Claims", str(batch.total_count))
    table.add_row("Files Processed", str(stats["files_processed"]))
    table.add_row("Processing Time", f"{stats['processing_time_seconds']:.2f}s")
    table.add_row("Claims/Second", f"{stats['claims_per_second']:.0f}")

    if stats["validation_errors"] > 0:
        table.add_row("Validation Errors", str(stats["validation_errors"]), style="red")

    console.print(table)


def _display_validation_results(result: ProcessingResult, detailed: bool = False):
    """Display validation results."""
    if result.success:
        console.print("[bold green]✓ Validation passed[/bold green]")
    else:
        console.print("[bold red]✗ Validation failed[/bold red]")

    table = Table(title="Validation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right", style="magenta")

    table.add_row("Processed", str(result.processed_count))
    table.add_row(
        "Errors", str(result.error_count), style="red" if result.error_count > 0 else "green"
    )
    table.add_row(
        "Warnings",
        str(result.warnings_count),
        style="yellow" if result.warnings_count > 0 else "green",
    )
    table.add_row("Processing Time", f"{result.processing_time_seconds:.2f}s")

    console.print(table)

    if detailed and result.errors:
        console.print("\n[bold blue]Detailed Errors:[/bold blue]")
        for error in result.errors[:10]:  # Show first 10 errors
            console.print(f"  [red]ERROR[/red] {error.field_name}: {error.error_message}")
            if error.claim_id:
                console.print(f"    Claim ID: {error.claim_id}")

        if len(result.errors) > 10:
            console.print(f"  ... and {len(result.errors) - 10} more errors")


def _save_batch_output(
    batch: ClaimBatch,
    output_path: str,
    format: str,
    preprocess: bool,
    preprocessor: Optional[ClaimPreprocessor],
):
    """Save batch output to file."""
    output_path = Path(output_path)

    try:
        if preprocess and preprocessor:
            # Save preprocessed data
            df = preprocessor.preprocess_claims(batch.claims)
        else:
            # Save raw claims data
            claims_data = [claim.dict() for claim in batch.claims]
            df = pd.DataFrame(claims_data)

        if format == "json":
            if preprocess:
                df.to_json(output_path, orient="records", indent=2)
            else:
                with open(output_path, "w") as f:
                    json.dump(
                        {
                            "batch_id": batch.batch_id,
                            "processed_at": batch.processed_at.isoformat(),
                            "total_count": batch.total_count,
                            "claims": claims_data,
                        },
                        f,
                        indent=2,
                        default=str,
                    )

        elif format == "csv":
            df.to_csv(output_path, index=False)

        elif format == "parquet":
            df.to_parquet(output_path, index=False)

        console.print(f"[bold green]Output saved:[/bold green] {output_path}")

    except Exception as e:
        console.print(f"[bold red]Error saving output:[/bold red] {e}")


# Async entry point for streaming
@cli.command()
@click.option("--claim-types", "-t", multiple=True, help="Claim types to stream")
@click.option("--chunk-size", "-c", default=1000, help="Chunk size for streaming")
@click.option("--max-chunks", "-m", default=10, help="Maximum chunks to process")
@click.pass_context
def stream_async(ctx, claim_types, chunk_size, max_chunks):
    """Stream claims asynchronously."""
    data_dir = ctx.obj["data_dir"]

    async def async_stream():
        config = DataLoaderConfig(chunk_size=chunk_size)
        loader = ClaimDataLoader(data_dir, config)

        chunk_count = 0
        total_claims = 0

        async for chunk in loader.stream_claims_async(
            claim_types=list(claim_types) if claim_types else None, chunk_size=chunk_size
        ):
            chunk_count += 1
            total_claims += len(chunk)

            console.print(f"Async processed chunk {chunk_count}: {len(chunk)} claims")

            if chunk_count >= max_chunks:
                break

        console.print(
            f"[bold green]Async streaming complete:[/bold green] {chunk_count} chunks, {total_claims} total claims"
        )

    try:
        asyncio.run(async_stream())
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
