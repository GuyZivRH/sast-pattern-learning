#!/usr/bin/env python3
"""
Run All Phase 3 Experiments
Optimized to run 10 combinations: 2 baseline + (2 models × 4 pattern types)
All patterns from: /Users/gziv/Dev/sast-pattern-learning/outputs/test_run/run_20260120_200345
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

# Configuration
VAL_DIR = Path("/Users/gziv/Dev/sast-pattern-learning/data/val")
PHASE1_DIR = Path("/Users/gziv/Dev/sast-pattern-learning/outputs/test_run/run_20260120_200345")
OUTPUT_BASE = Path("/Users/gziv/Dev/sast-pattern-learning/outputs/phase3_experiments")
ENV_FILE = Path("/Users/gziv/Dev/sast-pattern-learning/.env")

MODELS = {
    'slm': 'qwen/qwen2.5-coder-7b-instruct',
    'llm': 'qwen/qwen3-next-80b-a3b-instruct'
}

PATTERN_CONFIGS = [
    {'name': 'step_0', 'pattern_glob': '*_step_0_patterns.json', 'include_tp': True},
    {'name': 'step_1', 'pattern_glob': '*_step_1_patterns.json', 'include_tp': True},
    {'name': 'step_2', 'pattern_glob': '*_step_2_patterns.json', 'include_tp': True},
    {'name': 'step_2_fp_only', 'pattern_glob': '*_step_2_patterns.json', 'include_tp': False},
]


def update_env_model(model_name):
    """Update .env file with specified model."""
    print(f"\n{'='*80}")
    print(f"Updating .env with model: {model_name}")
    print(f"{'='*80}")

    if not ENV_FILE.exists():
        print(f"Warning: .env file not found at {ENV_FILE}")
        return False

    # Read current .env
    with open(ENV_FILE, 'r') as f:
        lines = f.readlines()

    # Update MODEL_NAME line
    updated = False
    new_lines = []
    for line in lines:
        if line.startswith('NIM_MODEL='):
            new_lines.append(f'NIM_MODEL={model_name}\n')
            updated = True
        else:
            new_lines.append(line)

    # If MODEL_NAME not found, add it
    if not updated:
        new_lines.append(f'NIM_MODEL={model_name}\n')

    # Write back
    with open(ENV_FILE, 'w') as f:
        f.writelines(new_lines)

    print(f"✓ Updated NIM_MODEL={model_name}")
    return True


def get_current_model():
    """Read current model from .env."""
    if not ENV_FILE.exists():
        return None

    with open(ENV_FILE, 'r') as f:
        for line in f:
            if line.startswith('MODEL_NAME='):
                return line.strip().split('=', 1)[1]
    return None


def load_patterns(pattern_glob, include_tp=True):
    """Load patterns from PHASE1_DIR matching glob pattern."""
    pattern_files = sorted(PHASE1_DIR.glob(pattern_glob))

    if not pattern_files:
        print(f"  Warning: No pattern files found matching {pattern_glob} in {PHASE1_DIR}")
        return {}

    patterns_by_issue_type = {}

    for pattern_file in pattern_files:
        # Extract issue type from filename
        # e.g., "COMPILER_WARNING_step_0_patterns.json" -> "COMPILER_WARNING"
        issue_type = pattern_file.stem.replace('_step_0_patterns', '').replace('_step_1_patterns', '').replace('_step_2_patterns', '')

        with open(pattern_file, 'r') as f:
            patterns = json.load(f)

        if not isinstance(patterns, dict):
            continue

        fp_patterns = patterns.get('fp', [])
        tp_patterns = patterns.get('tp', []) if include_tp else []

        patterns_by_issue_type[issue_type] = {
            'fp': fp_patterns,
            'tp': tp_patterns
        }

    return patterns_by_issue_type


def create_temp_pattern_dir(patterns_by_issue_type, temp_dir):
    """Create temporary directory with pattern files for phase3 script."""
    # Clear directory if it exists to avoid stale pattern files
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    for issue_type, patterns in patterns_by_issue_type.items():
        # Always use step_2 naming for consistency with run_phase3_from_phase1.py
        pattern_file = temp_dir / f"{issue_type}_step_2_patterns.json"
        with open(pattern_file, 'w') as f:
            json.dump(patterns, f, indent=2)

    return temp_dir


def is_run_completed(output_dir):
    """Check if a run has already been completed by looking for results file."""
    results_file = output_dir / "phase3_test_results" / "overall_results.json"
    return results_file.exists()


def run_phase3_evaluation(model_key, model_name, pattern_config, baseline_mode, run_number, total_runs, issue_type_filter=None):
    """Run a single phase3 evaluation."""

    print(f"\n{'='*80}")
    print(f"RUN {run_number}/{total_runs}")
    print(f"{'='*80}")
    print(f"Model: {model_name} ({model_key})")
    print(f"Mode: {'BASELINE' if baseline_mode else 'PATTERN'}")
    print(f"Pattern Config: {pattern_config['name']}")
    print(f"Include TP patterns: {pattern_config['include_tp']}")
    if issue_type_filter:
        print(f"Issue Type Filter: {issue_type_filter}")
    print(f"{'='*80}\n")

    # Update .env with model
    if not update_env_model(model_name):
        print(f"✗ Failed to update .env, skipping run")
        return False

    # Verify model is set correctly
    current_model = get_current_model()
    print(f"\n✓ Verified MODEL_NAME from .env: {current_model}")
    if current_model != model_name:
        print(f"✗ ERROR: Model mismatch! Expected {model_name}, got {current_model}")
        return False

    # Setup output directory
    mode_str = 'baseline' if baseline_mode else 'pattern'
    filter_str = f"_{issue_type_filter}" if issue_type_filter else ""
    output_dir = OUTPUT_BASE / f"{model_key}_{pattern_config['name']}_{mode_str}{filter_str}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Load patterns
    print(f"\nLoading patterns from: {PHASE1_DIR}")
    print(f"Pattern glob: {pattern_config['pattern_glob']}")
    patterns = load_patterns(pattern_config['pattern_glob'], pattern_config['include_tp'])

    if not patterns:
        print(f"✗ No patterns loaded, skipping run")
        return False

    # Filter to single issue type if specified
    if issue_type_filter:
        if issue_type_filter in patterns:
            patterns = {issue_type_filter: patterns[issue_type_filter]}
            print(f"\n✓ Filtered to single issue type: {issue_type_filter}")
        else:
            print(f"✗ Issue type {issue_type_filter} not found in patterns, skipping run")
            return False

    print(f"✓ Loaded patterns for {len(patterns)} issue types:")
    for issue_type, pats in sorted(patterns.items()):
        print(f"  {issue_type}: {len(pats['fp'])} FP patterns, {len(pats['tp'])} TP patterns")

    # For baseline mode, replace all with empty patterns
    if baseline_mode:
        print("\nBaseline mode: Replacing all patterns with empty lists")
        for issue_type in patterns:
            patterns[issue_type] = {'fp': [], 'tp': []}

    # Create temporary pattern directory
    temp_pattern_dir = OUTPUT_BASE / f"temp_patterns_{model_key}_{pattern_config['name']}_{mode_str}"
    create_temp_pattern_dir(patterns, temp_pattern_dir)

    print(f"\nCreated temporary pattern directory: {temp_pattern_dir}")

    # Build command
    cmd = [
        'python3',
        '/Users/gziv/Dev/sast-pattern-learning/run_phase3_from_phase1.py',
        '--phase1-dir', str(temp_pattern_dir),
        '--test-dir', str(VAL_DIR),
        '--output-dir', str(output_dir),
        '--platform', 'nim',
        '--workers', '4'
    ]

    # Add baseline flag if in baseline mode
    if baseline_mode:
        cmd.append('--baseline')

    print(f"\nRunning command:")
    print(f"  {' '.join(cmd)}")

    # Run evaluation
    start_time = datetime.now()
    try:
        # Run with real-time output (no capture)
        subprocess.run(cmd, check=True)
        print(f"\n✓ Evaluation completed successfully")

        # Cleanup temp directory
        shutil.rmtree(temp_pattern_dir)

        elapsed = datetime.now() - start_time
        print(f"\n✓ Run completed in {elapsed}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n✗ Evaluation failed with exit code {e.returncode}")

        # Cleanup temp directory even on failure
        if temp_pattern_dir.exists():
            shutil.rmtree(temp_pattern_dir)

        return False


def main():
    # Set to None to run on all issue types, or specify an issue type for testing
    issue_type_filter = None

    print("="*80)
    if issue_type_filter:
        print("PHASE 3 EXPERIMENTS (SINGLE ISSUE TYPE)")
        print("="*80)
        print(f"Issue Type Filter: {issue_type_filter}")
    else:
        print("PHASE 3 EXPERIMENTS (ALL ISSUE TYPES)")
        print("="*80)
        print(f"Issue Type Filter: ALL")
    print(f"Total runs: 10 (2 baseline + 8 pattern runs)")
    print(f"Optimization: Baseline runs only once per model (not per pattern config)")
    print(f"Validation data: {VAL_DIR}")
    print(f"Pattern source: {PHASE1_DIR}")
    print(f"Output base: {OUTPUT_BASE}")
    print("="*80)

    # Validate directories
    if not VAL_DIR.exists():
        print(f"✗ Validation directory not found: {VAL_DIR}")
        sys.exit(1)

    if not PHASE1_DIR.exists():
        print(f"✗ Phase 1 directory not found: {PHASE1_DIR}")
        sys.exit(1)

    # Create output directory
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    # Track results
    results = []
    run_number = 0
    total_runs = 10  # 2 baselines + 8 pattern runs

    # STEP 1: Run baselines (once per model, using step_0 patterns to get issue types)
    print("\n" + "="*80)
    print("STEP 1: BASELINE RUNS (2 runs)")
    print("="*80)

    baseline_config = PATTERN_CONFIGS[0]  # Use step_0 to get issue types

    for model_key, model_name in MODELS.items():
        run_number += 1

        # Check if we have an existing baseline run (could be from any pattern config)
        filter_str = f"_{issue_type_filter}" if issue_type_filter else ""
        existing_baseline_dirs = list(OUTPUT_BASE.glob(f"{model_key}_step_*_baseline{filter_str}"))

        if existing_baseline_dirs:
            # Use the first existing baseline directory
            output_dir = existing_baseline_dirs[0]
            print(f"\n{'='*80}")
            print(f"RUN {run_number}/{total_runs} - SKIPPED (already completed)")
            print(f"{'='*80}")
            print(f"Model: {model_name} ({model_key})")
            print(f"Mode: BASELINE")
            print(f"Using existing results from: {output_dir}")
            print(f"{'='*80}\n")

            results.append({
                'run_number': run_number,
                'model_key': model_key,
                'model_name': model_name,
                'pattern_config': 'baseline',
                'baseline_mode': True,
                'issue_type_filter': issue_type_filter,
                'success': True,
                'skipped': True
            })
            continue

        # Run baseline
        success = run_phase3_evaluation(
            model_key=model_key,
            model_name=model_name,
            pattern_config=baseline_config,
            baseline_mode=True,
            run_number=run_number,
            total_runs=total_runs,
            issue_type_filter=issue_type_filter
        )

        results.append({
            'run_number': run_number,
            'model_key': model_key,
            'model_name': model_name,
            'pattern_config': 'baseline',
            'baseline_mode': True,
            'issue_type_filter': issue_type_filter,
            'success': success,
            'skipped': False
        })

    # STEP 2: Run pattern evaluations (all pattern configs)
    print("\n" + "="*80)
    print("STEP 2: PATTERN RUNS (8 runs)")
    print("="*80)

    for model_key, model_name in MODELS.items():
        for pattern_config in PATTERN_CONFIGS:
            run_number += 1

            # Check if already completed
            mode_str = 'pattern'
            filter_str = f"_{issue_type_filter}" if issue_type_filter else ""
            output_dir = OUTPUT_BASE / f"{model_key}_{pattern_config['name']}_{mode_str}{filter_str}"

            if is_run_completed(output_dir):
                print(f"\n{'='*80}")
                print(f"RUN {run_number}/{total_runs} - SKIPPED (already completed)")
                print(f"{'='*80}")
                print(f"Model: {model_name} ({model_key})")
                print(f"Mode: PATTERN")
                print(f"Pattern Config: {pattern_config['name']}")
                print(f"Output directory: {output_dir}")
                print(f"{'='*80}\n")

                results.append({
                    'run_number': run_number,
                    'model_key': model_key,
                    'model_name': model_name,
                    'pattern_config': pattern_config['name'],
                    'baseline_mode': False,
                    'issue_type_filter': issue_type_filter,
                    'success': True,
                    'skipped': True
                })
                continue

            success = run_phase3_evaluation(
                model_key=model_key,
                model_name=model_name,
                pattern_config=pattern_config,
                baseline_mode=False,
                run_number=run_number,
                total_runs=total_runs,
                issue_type_filter=issue_type_filter
            )

            results.append({
                'run_number': run_number,
                'model_key': model_key,
                'model_name': model_name,
                'pattern_config': pattern_config['name'],
                'baseline_mode': False,
                'issue_type_filter': issue_type_filter,
                'success': success,
                'skipped': False
            })

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    successful = sum(1 for r in results if r['success'])
    skipped = sum(1 for r in results if r.get('skipped', False))
    failed = len(results) - successful

    print(f"\nTotal runs: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Skipped (already completed): {skipped}")
    print(f"Failed: {failed}")

    if failed > 0:
        print(f"\nFailed runs:")
        for r in results:
            if not r['success']:
                print(f"  Run {r['run_number']}: {r['model_key']} / {r['pattern_config']} / {'baseline' if r['baseline_mode'] else 'pattern'}")

    # Save summary to JSON
    summary_file = OUTPUT_BASE / "experiment_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Summary saved to: {summary_file}")
    print(f"✓ All results saved to: {OUTPUT_BASE}")
    print("="*80)


if __name__ == "__main__":
    main()