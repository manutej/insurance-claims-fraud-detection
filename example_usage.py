#!/usr/bin/env python3
"""
Example usage of the Insurance Claims Data Ingestion Pipeline.

This script demonstrates how to use the various components of the pipeline
for different use cases.
"""

import logging
from pathlib import Path

from src.ingestion import ClaimDataLoader, DataLoaderConfig, ClaimValidator, ClaimPreprocessor
from src.models import claim_factory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_batch_loading():
    """Example of batch loading claims."""
    print("\n" + "=" * 50)
    print("EXAMPLE 1: Batch Loading")
    print("=" * 50)

    # Configure data loader
    data_dir = Path("data")
    config = DataLoaderConfig(batch_size=500, validate_on_load=True, max_workers=2)

    # Initialize loader
    loader = ClaimDataLoader(data_dir, config)

    try:
        # Load all medical claims
        batch = loader.load_claims_batch(claim_types=["medical"])

        print(f"Loaded {batch.total_count} medical claims")
        print(f"Batch ID: {batch.batch_id}")
        print(f"Processing time: {loader.get_statistics()['processing_time_seconds']:.2f}s")

        # Show sample claims
        if batch.claims:
            print(f"\nSample claim: {batch.claims[0].claim_id}")
            print(f"Patient ID: {batch.claims[0].patient_id}")
            print(f"Amount: ${batch.claims[0].billed_amount}")

    except Exception as e:
        print(f"Error: {e}")


def example_streaming():
    """Example of streaming claims."""
    print("\n" + "=" * 50)
    print("EXAMPLE 2: Streaming Claims")
    print("=" * 50)

    data_dir = Path("data")
    config = DataLoaderConfig(chunk_size=100)
    loader = ClaimDataLoader(data_dir, config)

    try:
        chunk_count = 0
        total_claims = 0

        # Stream claims in chunks
        for chunk in loader.stream_claims(chunk_size=100):
            chunk_count += 1
            total_claims += len(chunk)
            print(f"Processed chunk {chunk_count}: {len(chunk)} claims")

            # Process only first 3 chunks for demo
            if chunk_count >= 3:
                break

        print(f"Total processed: {total_claims} claims in {chunk_count} chunks")

    except Exception as e:
        print(f"Error: {e}")


def example_validation():
    """Example of validation."""
    print("\n" + "=" * 50)
    print("EXAMPLE 3: Claim Validation")
    print("=" * 50)

    # Create a validator
    validator = ClaimValidator()

    # Example claim data (valid)
    valid_claim_data = {
        "claim_id": "CLM-2024-001234",
        "patient_id": "PAT-78901",
        "provider_id": "PRV-45678",
        "provider_npi": "1234567890",
        "date_of_service": "2024-03-15",
        "diagnosis_codes": ["E11.9"],
        "diagnosis_descriptions": ["Type 2 diabetes mellitus without complications"],
        "procedure_codes": ["99213"],
        "procedure_descriptions": ["Office visit, established patient, low complexity"],
        "billed_amount": 125.00,
        "service_location": "11",
        "claim_type": "professional",
        "fraud_indicator": False,
    }

    # Example claim data (invalid)
    invalid_claim_data = {
        "claim_id": "INVALID-ID",
        "patient_id": "PAT-78901",
        "provider_npi": "123",  # Invalid NPI
        "date_of_service": "2025-12-31",  # Future date
        "billed_amount": -100,  # Negative amount
        "claim_type": "professional",
        # Missing required fields
    }

    # Validate claims
    claims_data = [valid_claim_data, invalid_claim_data]
    result = validator.validate_batch(claims_data)

    print(f"Validation result: {'PASSED' if result.success else 'FAILED'}")
    print(f"Processed: {result.processed_count}")
    print(f"Errors: {result.error_count}")
    print(f"Warnings: {result.warnings_count}")

    # Show errors
    if result.errors:
        print("\nValidation errors:")
        for error in result.errors[:5]:  # Show first 5 errors
            print(f"  - {error.field_name}: {error.error_message}")


def example_preprocessing():
    """Example of preprocessing."""
    print("\n" + "=" * 50)
    print("EXAMPLE 4: Data Preprocessing")
    print("=" * 50)

    try:
        # Load some claims first
        data_dir = Path("data")
        loader = ClaimDataLoader(data_dir)
        claims = loader.load_specific_claim_type("medical")[:10]  # First 10 claims

        if not claims:
            print("No medical claims found for preprocessing example")
            return

        # Initialize preprocessor
        preprocessor = ClaimPreprocessor()

        # Preprocess claims
        df = preprocessor.preprocess_claims(claims)

        print(f"Preprocessed {len(claims)} claims")
        print(f"Feature columns: {len(preprocessor.feature_columns)}")
        print(f"DataFrame shape: {df.shape}")

        # Show some features
        print(f"\nSample features: {preprocessor.feature_columns[:10]}")

        # Feature importance data
        feature_info = preprocessor.get_feature_importance_data(df)
        print(f"Total features created: {feature_info['feature_count']}")

        if "correlations" in feature_info:
            top_corr = sorted(
                feature_info["correlations"].items(), key=lambda x: x[1], reverse=True
            )[:5]
            print(f"\nTop correlated features with fraud:")
            for feature, corr in top_corr:
                print(f"  {feature}: {corr:.3f}")

    except Exception as e:
        print(f"Error: {e}")


def example_file_info():
    """Example of getting file information."""
    print("\n" + "=" * 50)
    print("EXAMPLE 5: Data File Information")
    print("=" * 50)

    try:
        data_dir = Path("data")
        loader = ClaimDataLoader(data_dir)

        # Get file summary
        summary = loader.get_file_summary()

        print(f"Total files: {summary['total_files']}")
        print(f"Total size: {summary['total_size_mb']:.1f} MB")
        print(f"\nFiles by type:")

        for file_type, count in summary["files_by_type"].items():
            print(f"  {file_type}: {count} files")

        # Show file details
        print(f"\nFile details:")
        for detail in summary["file_details"][:5]:  # First 5 files
            print(f"  {detail['name']} ({detail['type']}) - {detail['size_mb']:.1f} MB")

    except Exception as e:
        print(f"Error: {e}")


def example_claim_factory():
    """Example of using claim factory."""
    print("\n" + "=" * 50)
    print("EXAMPLE 6: Claim Factory")
    print("=" * 50)

    # Medical claim data
    medical_data = {
        "claim_id": "CLM-2024-001234",
        "patient_id": "PAT-78901",
        "provider_id": "PRV-45678",
        "provider_npi": "1234567890",
        "date_of_service": "2024-03-15",
        "diagnosis_codes": ["E11.9"],
        "diagnosis_descriptions": ["Type 2 diabetes mellitus"],
        "procedure_codes": ["99213"],
        "procedure_descriptions": ["Office visit"],
        "billed_amount": 125.00,
        "service_location": "11",
        "claim_type": "professional",
        "fraud_indicator": False,
    }

    # Pharmacy claim data
    pharmacy_data = {
        "claim_id": "CLM-2024-001235",
        "patient_id": "PAT-78902",
        "provider_id": "PRV-45679",
        "provider_npi": "1234567891",
        "date_of_service": "2024-03-16",
        "ndc_code": "12345-1234-12",
        "drug_name": "Metformin",
        "strength": "500mg",
        "quantity": 90,
        "days_supply": 30,
        "prescriber_npi": "1234567892",
        "pharmacy_npi": "1234567893",
        "fill_date": "2024-03-16",
        "billed_amount": 25.00,
        "claim_type": "pharmacy",
        "fraud_indicator": False,
    }

    try:
        # Create claims using factory
        medical_claim = claim_factory(medical_data)
        pharmacy_claim = claim_factory(pharmacy_data)

        print(f"Medical claim: {type(medical_claim).__name__}")
        print(f"  ID: {medical_claim.claim_id}")
        print(f"  Amount: ${medical_claim.billed_amount}")
        print(f"  Procedures: {len(medical_claim.procedure_codes)}")

        print(f"\nPharmacy claim: {type(pharmacy_claim).__name__}")
        print(f"  ID: {pharmacy_claim.claim_id}")
        print(f"  Drug: {pharmacy_claim.drug_name}")
        print(f"  Quantity: {pharmacy_claim.quantity}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("Insurance Claims Data Ingestion Pipeline - Examples")
    print("=" * 60)

    # Run examples
    example_file_info()
    example_claim_factory()
    example_validation()
    example_batch_loading()
    example_streaming()
    example_preprocessing()

    print("\n" + "=" * 60)
    print("Examples completed! Check the output above.")
    print("=" * 60)
