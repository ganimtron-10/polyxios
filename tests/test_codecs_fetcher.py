"""Dynamic integration test suite utilizing the default codec registry for asset roundtrips."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from polyxios import _REGISTRY, validate
from polyxios.exceptions import FetcherError, ValidationError
from polyxios.fetcher import fetch_by_extension


def test_codec_registry_release_packs() -> None:
    """
    Dynamically query the default codec registry, download corresponding release packs,
    and execute thorough validation and roundtrip evaluations.
    """
    # 1. Dynamically build the format registry from the package state
    registry = _REGISTRY
    if not registry:
        pytest.skip("The default codec registry is empty. No codecs loaded.")

    # 2. Operational Metrics and Tracking Logs
    metrics = {
        "formats_found": len(registry),
        "formats_attempted": 0,
        "total_files_discovered": 0,
        "successful_reads": 0,
        "successful_validations": 0,
        "successful_writes": 0,
        "successful_roundtrips": 0,
    }
    failed_models: list[dict[str, str]] = []

    # 3. Iterate dynamically over registered formats
    for raw_ext, codec in registry.items():
        metrics["formats_attempted"] += 1

        # Clean the extension token to match repository subfolder mapping (e.g., '.vtk' -> 'vtk')
        ext_clean = raw_ext.lower().lstrip(".")
        if not ext_clean:
            continue

        try:
            discovered_files = fetch_by_extension(ext_clean)
        except FetcherError as e:
            # Catch network or system IO constraints gracefully without crashing the suite
            failed_models.append(
                {
                    "file": f"[{ext_clean.upper()} Package Download]",
                    "error_type": "FetcherError",
                    "message": str(e),
                }
            )
            continue

        if not discovered_files:
            continue

        metrics["total_files_discovered"] += len(discovered_files)

        # 4. Cycle through each localized asset file within the format suite
        for original_path in discovered_files:
            filename = os.path.basename(original_path)

            # Test Stage A: Read Pipeline
            try:
                poly_data_orig = codec.read(original_path)
                metrics["successful_reads"] += 1
            except Exception as e:
                failed_models.append(
                    {
                        "file": filename,
                        "error_type": f"ReadError ({ext_clean})",
                        "message": str(e),
                    }
                )
                continue

            # Test Stage B: Structural Validation
            try:
                validate(poly_data_orig)
                metrics["successful_validations"] += 1
            except ValidationError as e:
                failed_models.append(
                    {
                        "file": filename,
                        "error_type": "ValidationError",
                        "message": str(e),
                    }
                )
                continue

            # Test Stage C: Isolated Write Pipeline & Roundtrip Evaluation
            with tempfile.NamedTemporaryFile(suffix=raw_ext, delete=False) as f:
                tmp_output_path = f.name

            try:
                codec.write(poly_data_orig, tmp_output_path)
                metrics["successful_writes"] += 1

                poly_data_roundtrip = codec.read(tmp_output_path)

                # Asset Parity Assertions
                if (
                    hasattr(poly_data_orig, "vertices")
                    and poly_data_orig.vertices is not None
                ):
                    np.testing.assert_allclose(
                        poly_data_roundtrip.vertices, poly_data_orig.vertices, atol=1e-5
                    )

                if (
                    hasattr(poly_data_orig, "connectivity")
                    and poly_data_orig.connectivity is not None
                ):
                    np.testing.assert_array_equal(
                        poly_data_roundtrip.connectivity, poly_data_orig.connectivity
                    )

                if (
                    hasattr(poly_data_orig, "offsets")
                    and poly_data_orig.offsets is not None
                ):
                    np.testing.assert_array_equal(
                        poly_data_roundtrip.offsets, poly_data_orig.offsets
                    )

                for attr_key, original_attr in poly_data_orig.element_attrs.items():
                    assert attr_key in poly_data_roundtrip.element_attrs
                    if isinstance(original_attr, np.ndarray) and np.issubdtype(
                        original_attr.dtype, np.number
                    ):
                        np.testing.assert_allclose(
                            poly_data_roundtrip.element_attrs[attr_key],
                            original_attr,
                            atol=1e-5,
                        )

                metrics["successful_roundtrips"] += 1

            except Exception as e:
                failed_models.append(
                    {
                        "file": filename,
                        "error_type": f"RoundtripError ({ext_clean})",
                        "message": str(e),
                    }
                )
            finally:
                if os.path.exists(tmp_output_path):
                    os.remove(tmp_output_path)

    # 5. Output Execution Metrics Summary
    print("\n" + "=" * 60)
    print("POLYXIOS INTEGRATION TEST METRICS SUMMARY")
    print("=" * 60)
    print(f"Total Extension Codecs Registered: {metrics['formats_found']}")
    print(f"Total Extension Codecs Attempted:  {metrics['formats_attempted']}")
    print(f"Total Asset Files Downloaded:      {metrics['total_files_discovered']}")
    print(f"Successful File Reads:             {metrics['successful_reads']}")
    print(f"Successful File Validations:       {metrics['successful_validations']}")
    print(f"Successful File Writes:            {metrics['successful_writes']}")
    print(f"Successful Full Roundtrips:        {metrics['successful_roundtrips']}")
    print("=" * 60)

    if failed_models:
        print(f"\n❌ DETECTED COMPLIANCE FAILURES ({len(failed_models)}):")
        for failure in failed_models:
            print(f"  • File/Asset: {failure['file']}")
            print(f"    Error Stage: {failure['error_type']}")
            print(f"    Log Trace:   {failure['message']}")
            print("-" * 40)
        pytest.fail(
            f"Integration roundtrip failures detected for {len(failed_models)} assets."
        )
    else:
        print(
            "\n✅ All registered codecs and downloaded assets passed compliance roundtrips perfectly."
        )
