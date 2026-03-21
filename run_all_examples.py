"""
Run All Examples - Comprehensive Test Suite

This script runs all example files to verify they work correctly.
"""

import sys
import subprocess
from pathlib import Path
import importlib.util


def run_example(example_path):
    """Run a single example file."""
    example_name = example_path.name
    print(f"\n{'='*70}")
    print(f"Running: {example_name}")
    print(f"{'='*70}")
    
    try:
        # Check if file has main guard
        with open(example_path, 'r') as f:
            content = f.read()
        
        if 'if __name__ == "__main__":' not in content:
            print(f"⚠️  Skipping {example_name} - no main guard")
            return False
        
        # Run the example with the repo root on PYTHONPATH so the local
        # dev version of ragfallback is used, not any installed release.
        import os
        env = os.environ.copy()
        repo_root = str(Path(__file__).parent)
        env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            [sys.executable, str(example_path)],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            env=env,
        )
        
        if result.returncode == 0:
            print(f"✅ {example_name} completed successfully")
            if result.stdout:
                # Show last few lines of output
                lines = result.stdout.strip().split('\n')
                if len(lines) > 5:
                    print("   Output (last 5 lines):")
                    for line in lines[-5:]:
                        print(f"   {line}")
                else:
                    print("   Output:")
                    for line in lines:
                        print(f"   {line}")
            return True
        else:
            print(f"❌ {example_name} failed with return code {result.returncode}")
            if result.stderr:
                print("   Error output:")
                for line in result.stderr.strip().split('\n')[:10]:
                    print(f"   {line}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏱️  {example_name} timed out (may be waiting for user input)")
        return False
    except Exception as e:
        print(f"❌ {example_name} crashed: {e}")
        return False


def run_all_examples():
    """Run all example files."""
    print("="*70)
    print("ragfallback - Running All Examples")
    print("="*70)
    print("\nThis will test all example files to ensure they work correctly.")
    print("Note: Some examples may require API keys or external services.\n")
    
    examples_dir = Path(__file__).parent / "examples"
    
    if not examples_dir.exists():
        print(f"❌ Examples directory not found: {examples_dir}")
        return 1
    
    example_files = sorted(examples_dir.glob("*.py"))
    
    if not example_files:
        print("❌ No example files found")
        return 1
    
    print(f"Found {len(example_files)} example file(s)\n")
    
    # Scripts that need an external service or optional extra — skip with a reason.
    SKIP_WITH_REASON: dict[str, str] = {
        "qdrant_local_demo.py":             "requires Docker + Qdrant",
        "uc5_hybrid_failover.py":           "requires ragfallback[hybrid]",
        "uc6_multi_hop_demo.py":            "requires Ollama or LLM",
        "uc6_adaptive_rag.py":              "requires Ollama or LLM",
        "production_reliability_example.py": "requires Ollama or LLM",
        "legal_document_analysis.py":       "requires ragfallback[real-data] + LLM",
        "medical_research_synthesis.py":    "requires ragfallback[real-data] + LLM",
        "financial_risk_analysis.py":       "requires ragfallback[real-data] + LLM",
    }
    # Internal helpers / non-runnable files (silently skipped, no output line).
    SILENT_SKIP = {"_kb_common.py", "__init__.py"}

    results = []
    for example_file in example_files:
        name = example_file.name
        if name in SILENT_SKIP:
            continue
        if name in SKIP_WITH_REASON:
            print(f"\n⏭️  Skipping {name} ({SKIP_WITH_REASON[name]})")
            continue

        success = run_example(example_file)
        results.append((name, success))
    
    # Summary
    print("\n" + "="*70)
    print("EXAMPLES SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for example_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {example_name}")
    
    print(f"\nTotal: {passed}/{total} examples passed")
    
    if passed == total:
        print("\n🎉 All examples ran successfully!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} example(s) failed (may require API keys or services)")
        print("   Check individual outputs above for details")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_examples())





