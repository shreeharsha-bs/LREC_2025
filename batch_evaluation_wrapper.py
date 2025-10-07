#!/usr/bin/env python3
"""
Batch Evaluation Wrapper Script
Runs personality profiling and longform evaluations on multiple audio files and outputs CSV results.
"""

import os
import sys
import csv
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

# Import the workflow components
from test_complete_workflow import complete_workflow_test, longform_evaluation_workflow

# Task type mapping for longform evaluations
TASK_MAP = {
    "therapy": "therapy",
    "career_advice": "career",
    "interview_screening": "interview",
    "story": "story"
}

# Personality dimensions (from personality_score_extractor.py)
PERSONALITY_DIMENSIONS = [
    "Openness to Experience",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Emotional Stability (vs Neuroticism)"
]

# Longform task dimensions (from longform_evaluator.py)
LONGFORM_DIMENSIONS = {
    "therapy": [
        "Agency of Advice",
        "Emotional Validation",
        "Improvement vs Retreat"
    ],
    "career": [
        "Role Status",
        "STEM vs Care Orientation",
        "Actionability of Plan"
    ],
    "interview": [
        "Shortlist Decision",
        "Salary Offer Bias",
        "Leadership Endorsement"
    ],
    "story": [
        "Heroic Agency",
        "Person in Distress",
        "Achievement vs Relational Arc"
    ]
}


def get_audio_files_from_subfolders(base_dir: str, num_files: int = 3) -> List[Tuple[str, str]]:
    """
    Get audio files from subfolders, selecting num_files alphabetically from each.
    Returns list of (subfolder_name, file_path) tuples.
    """
    base_path = Path(base_dir)
    audio_files = []
    
    if not base_path.exists():
        print(f"Warning: Directory not found: {base_dir}")
        return audio_files
    
    # Get all subfolders
    subfolders = [d for d in base_path.iterdir() if d.is_dir()]
    
    for subfolder in sorted(subfolders):
        # Get wav files from this subfolder
        wav_files = sorted(list(subfolder.glob("*.wav")))
        
        # Take first num_files alphabetically
        selected_files = wav_files[:num_files]
        
        for wav_file in selected_files:
            audio_files.append((subfolder.name, str(wav_file)))
    
    return audio_files


def extract_task_from_filename(filename: str) -> str:
    """
    Extract task type from filename like '1_s01_career_advice.wav'
    Returns the normalized task name (e.g., 'career')
    """
    basename = Path(filename).stem  # Remove .wav extension
    parts = basename.split('_')
    
    # Last part should be the task type (e.g., 'therapy', 'career_advice')
    if len(parts) >= 3:
        task_suffix = '_'.join(parts[2:])  # Handle multi-word tasks
        
        # Map to standardized task name
        for key, value in TASK_MAP.items():
            if key in task_suffix:
                return value
    
    return None


async def run_personality_evaluations(audio_files: List[Tuple[str, str]]) -> List[Dict]:
    """Run personality profiling workflow on audio files."""
    results = []
    
    print("\n" + "="*80)
    print("PERSONALITY PROFILING MODE")
    print("="*80 + "\n")
    
    total = len(audio_files)
    for idx, (subfolder, audio_file) in enumerate(audio_files, 1):
        print(f"\n[{idx}/{total}] Processing: {Path(audio_file).name} (from {subfolder})")
        print("-" * 80)
        
        try:
            result = await complete_workflow_test(audio_file)
            
            if not result or "personality_scores" not in result:
                print(f"❌ Failed to process {audio_file}")
                print("Stopping execution due to error.")
                sys.exit(1)
            
            # Extract scores
            row_data = {
                "file": Path(audio_file).name,
                "subfolder": subfolder,
                "full_path": audio_file
            }
            
            # Add personality dimension scores
            for dim_name in PERSONALITY_DIMENSIONS:
                dim_key = dim_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
                if dim_name in result["personality_scores"]:
                    row_data[dim_name] = result["personality_scores"][dim_name]["score"]
                else:
                    row_data[dim_name] = None
            
            results.append(row_data)
            print(f"✅ Successfully processed {Path(audio_file).name}")
            
        except Exception as e:
            print(f"❌ Error processing {audio_file}: {e}")
            print("Stopping execution due to error.")
            sys.exit(1)
    
    return results


async def run_longform_evaluations(audio_files: List[Tuple[str, str]]) -> List[Dict]:
    """Run longform evaluation workflow on audio files."""
    results = []
    
    print("\n" + "="*80)
    print("LONGFORM EVALUATION MODE")
    print("="*80 + "\n")
    
    total = len(audio_files)
    for idx, (subfolder, audio_file) in enumerate(audio_files, 1):
        print(f"\n[{idx}/{total}] Processing: {Path(audio_file).name} (from {subfolder})")
        print("-" * 80)
        
        # Extract task type from filename
        task_type = extract_task_from_filename(audio_file)
        
        if not task_type:
            print(f"❌ Could not extract task type from filename: {audio_file}")
            print("Stopping execution due to error.")
            sys.exit(1)
        
        print(f"📋 Task type: {task_type}")
        
        try:
            result = await longform_evaluation_workflow(audio_file, task_type)
            
            if not result or "scores" not in result:
                print(f"❌ Failed to process {audio_file}")
                print("Stopping execution due to error.")
                sys.exit(1)
            
            # Extract scores
            row_data = {
                "file": Path(audio_file).name,
                "subfolder": subfolder,
                "task_type": task_type,
                "full_path": audio_file
            }
            
            # Add task-specific dimension scores
            task_dimensions = LONGFORM_DIMENSIONS.get(task_type, [])
            for dim_name in task_dimensions:
                if dim_name in result["scores"]:
                    row_data[dim_name] = result["scores"][dim_name]["score"]
                else:
                    row_data[dim_name] = None
            
            results.append(row_data)
            print(f"✅ Successfully processed {Path(audio_file).name}")
            
        except Exception as e:
            print(f"❌ Error processing {audio_file}: {e}")
            print("Stopping execution due to error.")
            sys.exit(1)
    
    return results


def save_personality_csv(results: List[Dict], output_file: str):
    """Save personality profiling results to CSV."""
    if not results:
        print("No results to save for personality profiling.")
        return
    
    fieldnames = ["file", "subfolder", "full_path"] + PERSONALITY_DIMENSIONS
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✅ Personality results saved to: {output_file}")


def save_longform_csv(results: List[Dict], output_file: str):
    """Save longform evaluation results to CSV."""
    if not results:
        print("No results to save for longform evaluation.")
        return
    
    # Collect all unique dimension names across all tasks
    all_dimensions = set()
    for result in results:
        task_type = result.get("task_type")
        if task_type and task_type in LONGFORM_DIMENSIONS:
            all_dimensions.update(LONGFORM_DIMENSIONS[task_type])
    
    fieldnames = ["file", "subfolder", "task_type", "full_path"] + sorted(all_dimensions)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✅ Longform results saved to: {output_file}")


async def main():
    parser = argparse.ArgumentParser(description="Batch evaluation wrapper for personality and longform tasks")
    parser.add_argument("--corpus-dir", default="corpus",
                       help="Directory containing personality profiling audio subfolders (default: corpus)")
    parser.add_argument("--longform-dir", default="modified_response_task",
                       help="Directory containing longform task audio subfolders (default: modified_response_task)")
    parser.add_argument("--num-files", type=int, default=3,
                       help="Number of files to process from each subfolder (default: 3)")
    parser.add_argument("--personality-csv", default="personality_results.csv",
                       help="Output CSV filename for personality results (default: personality_results.csv)")
    parser.add_argument("--longform-csv", default="longform_results.csv",
                       help="Output CSV filename for longform results (default: longform_results.csv)")
    parser.add_argument("--mode", choices=["personality", "longform", "both"], default="both",
                       help="Evaluation mode: personality, longform, or both (default: both)")
    parser.add_argument("--openai-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--google-key", help="Google API key (or set GOOGLE_API_KEY env var)")
    
    args = parser.parse_args()
    
    # Set API keys if provided
    if args.openai_key:
        os.environ["OPENAI_API_KEY"] = args.openai_key
    if args.google_key:
        os.environ["GOOGLE_API_KEY"] = args.google_key
    
    # Verify API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set. Use --openai-key or set environment variable.")
        sys.exit(1)
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ GOOGLE_API_KEY not set. Use --google-key or set environment variable.")
        sys.exit(1)
    
    print("🚀 Batch Evaluation Wrapper")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Files per subfolder: {args.num_files}")
    print("=" * 80)
    
    # Run personality profiling
    if args.mode in ["personality", "both"]:
        print(f"\n📂 Collecting audio files from: {args.corpus_dir}")
        personality_files = get_audio_files_from_subfolders(args.corpus_dir, args.num_files)
        print(f"Found {len(personality_files)} audio files for personality profiling")
        
        if personality_files:
            personality_results = await run_personality_evaluations(personality_files)
            save_personality_csv(personality_results, args.personality_csv)
    
    # Run longform evaluations
    if args.mode in ["longform", "both"]:
        print(f"\n📂 Collecting audio files from: {args.longform_dir}")
        longform_files = get_audio_files_from_subfolders(args.longform_dir, args.num_files)
        print(f"Found {len(longform_files)} audio files for longform evaluation")
        
        if longform_files:
            longform_results = await run_longform_evaluations(longform_files)
            save_longform_csv(longform_results, args.longform_csv)
    
    print("\n" + "=" * 80)
    print("🎉 Batch evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
