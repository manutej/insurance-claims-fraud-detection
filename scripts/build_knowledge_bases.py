#!/usr/bin/env python3
"""
Build all 4 knowledge bases for RAG enrichment system.

Usage:
    python scripts/build_knowledge_bases.py --api-key YOUR_OPENAI_KEY

This script:
1. Loads enhanced synthetic data from data/ directory
2. Builds all 4 KBs in sequence:
   - Patient Claim History KB
   - Provider Behavior Pattern KB
   - Medical Coding Standards KB
   - Regulatory Guidance & Fraud Patterns KB
3. Validates KB completeness
4. Generates KB statistics
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict

from qdrant_client import QdrantClient
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from src.rag.knowledge_bases.patient_kb import PatientClaimHistoryKB

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

console = Console()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Build RAG knowledge bases")
    parser.add_argument(
        "--api-key",
        type=str,
        required=False,
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--qdrant-url",
        type=str,
        default="http://localhost:6333",
        help="Qdrant server URL (default: http://localhost:6333)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Data directory path (default: data/)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing KBs without rebuilding",
    )
    return parser.parse_args()


def build_patient_kb(
    qdrant_client: QdrantClient, openai_api_key: str, data_dir: Path
) -> Dict:
    """
    Build Patient Claim History KB.

    Args:
        qdrant_client: Qdrant client
        openai_api_key: OpenAI API key
        data_dir: Data directory

    Returns:
        Dict with build statistics
    """
    console.print("\n[bold cyan]Building Patient Claim History KB...[/bold cyan]")

    kb = PatientClaimHistoryKB(qdrant_client=qdrant_client, openai_api_key=openai_api_key)

    # Create collection
    kb.create_collection()

    # Build from data (combine valid + fraudulent claims)
    # TODO: Load from actual data files once enhanced data is ready
    # For now, this is a placeholder
    console.print("  ⚠️  Waiting for enhanced synthetic data generation...")
    console.print("  ℹ️  Patient KB structure ready, data loading pending")

    # Validate
    is_valid = kb.validate()

    # Get statistics
    stats = kb.get_statistics()

    return {
        "kb_name": "Patient Claim History",
        "collection_name": kb.collection_name,
        "is_valid": is_valid,
        "total_documents": stats.total_documents,
        "vector_dimensions": stats.vector_dimensions,
        "avg_query_latency_ms": stats.avg_query_latency_ms,
        "status": "Ready (awaiting data)" if is_valid else "Created",
    }


def build_provider_kb(
    qdrant_client: QdrantClient, openai_api_key: str, data_dir: Path
) -> Dict:
    """Build Provider Behavior Pattern KB (placeholder)."""
    console.print("\n[bold cyan]Building Provider Behavior Pattern KB...[/bold cyan]")
    console.print("  ⚠️  Implementation pending (Week 1, Day 3-4)")

    return {
        "kb_name": "Provider Behavior Pattern",
        "collection_name": "provider_behavior_patterns",
        "is_valid": False,
        "total_documents": 0,
        "vector_dimensions": 1536,
        "avg_query_latency_ms": 0.0,
        "status": "Pending",
    }


def build_medical_coding_kb(
    qdrant_client: QdrantClient, openai_api_key: str, data_dir: Path
) -> Dict:
    """Build Medical Coding Standards KB (placeholder)."""
    console.print("\n[bold cyan]Building Medical Coding Standards KB...[/bold cyan]")
    console.print("  ⚠️  Implementation pending (Week 2, Day 1-2)")

    return {
        "kb_name": "Medical Coding Standards",
        "collection_name": "medical_coding_standards",
        "is_valid": False,
        "total_documents": 0,
        "vector_dimensions": 1536,
        "avg_query_latency_ms": 0.0,
        "status": "Pending",
    }


def build_regulatory_kb(
    qdrant_client: QdrantClient, openai_api_key: str, data_dir: Path
) -> Dict:
    """Build Regulatory Guidance & Fraud Patterns KB (placeholder)."""
    console.print("\n[bold cyan]Building Regulatory Guidance & Fraud Patterns KB...[/bold cyan]")
    console.print("  ⚠️  Implementation pending (Week 2, Day 3-4)")

    return {
        "kb_name": "Regulatory Guidance & Fraud Patterns",
        "collection_name": "regulatory_guidance",
        "is_valid": False,
        "total_documents": 0,
        "vector_dimensions": 1536,
        "avg_query_latency_ms": 0.0,
        "status": "Pending",
    }


def display_statistics_table(kb_stats: list) -> None:
    """Display KB statistics in a rich table."""
    table = Table(title="Knowledge Base Build Statistics", show_header=True, header_style="bold magenta")

    table.add_column("KB Name", style="cyan", no_wrap=True)
    table.add_column("Collection", style="green")
    table.add_column("Documents", justify="right", style="yellow")
    table.add_column("Dimensions", justify="right")
    table.add_column("Valid", justify="center")
    table.add_column("Status", justify="center")

    for stats in kb_stats:
        valid_icon = "✅" if stats["is_valid"] else "❌"
        status_color = "green" if stats["is_valid"] else "yellow"

        table.add_row(
            stats["kb_name"],
            stats["collection_name"],
            f"{stats['total_documents']:,}",
            str(stats["vector_dimensions"]),
            valid_icon,
            f"[{status_color}]{stats['status']}[/{status_color}]",
        )

    console.print("\n")
    console.print(table)


def main() -> None:
    """Main build script."""
    args = parse_args()

    # Get API key
    openai_api_key = args.api_key
    if not openai_api_key:
        import os

        openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key and not args.validate_only:
        console.print("[bold red]Error:[/bold red] OpenAI API key required")
        console.print("Set --api-key or OPENAI_API_KEY environment variable")
        sys.exit(1)

    # Initialize Qdrant client
    console.print(f"[bold]Connecting to Qdrant at {args.qdrant_url}...[/bold]")
    try:
        qdrant_client = QdrantClient(url=args.qdrant_url)
        console.print("[green]✓ Connected to Qdrant[/green]")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] Could not connect to Qdrant: {e}")
        console.print("Make sure Qdrant is running: docker run -p 6333:6333 qdrant/qdrant")
        sys.exit(1)

    # Build all KBs
    kb_stats = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Building knowledge bases...", total=4)

        # KB 1: Patient Claim History
        stats = build_patient_kb(qdrant_client, openai_api_key, args.data_dir)
        kb_stats.append(stats)
        progress.update(task, advance=1)

        # KB 2: Provider Behavior Pattern
        stats = build_provider_kb(qdrant_client, openai_api_key, args.data_dir)
        kb_stats.append(stats)
        progress.update(task, advance=1)

        # KB 3: Medical Coding Standards
        stats = build_medical_coding_kb(qdrant_client, openai_api_key, args.data_dir)
        kb_stats.append(stats)
        progress.update(task, advance=1)

        # KB 4: Regulatory Guidance
        stats = build_regulatory_kb(qdrant_client, openai_api_key, args.data_dir)
        kb_stats.append(stats)
        progress.update(task, advance=1)

    # Display statistics
    display_statistics_table(kb_stats)

    # Summary
    total_docs = sum(s["total_documents"] for s in kb_stats)
    valid_kbs = sum(1 for s in kb_stats if s["is_valid"])

    console.print("\n[bold]Build Summary:[/bold]")
    console.print(f"  Total Documents: [cyan]{total_docs:,}[/cyan]")
    console.print(f"  Valid KBs: [green]{valid_kbs}/4[/green]")
    console.print(f"  Status: [yellow]Phase 2A - 25% Complete[/yellow]")

    console.print("\n[bold]Next Steps:[/bold]")
    console.print("  1. Complete Provider KB (Week 1, Day 3-4)")
    console.print("  2. Complete Medical Coding KB (Week 2, Day 1-2)")
    console.print("  3. Complete Regulatory KB (Week 2, Day 3-4)")
    console.print("  4. Generate enhanced synthetic data for all KBs")
    console.print("  5. Run integration tests")

    console.print("\n[bold green]✨ Build process complete![/bold green]\n")


if __name__ == "__main__":
    main()
