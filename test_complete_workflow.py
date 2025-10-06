#!/usr/bin/env python3
"""
Complete workflow test: Speaker Profiling + Local ASR Score Extraction
Supports both personality profiling and long-form task evaluation modes.
"""

import asyncio
import os
import argparse
from speaker_profiling_single import SpeakerProfilerSingle
from longform_evaluator import LongFormEvaluator
from personality_score_extractor import process_personality_audio

async def complete_workflow_test(audio_file: str):
    """Test the complete workflow from audio input to personality scores."""
    
    print("🎯 Complete Speaker Profiling Workflow Test")
    print("=" * 50)
    
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set")
        return
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ GOOGLE_API_KEY not set")
        return
    
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return
    
    print(f"📁 Input audio: {audio_file}")
    print()
    
    # Step 1: Generate AI personality analysis
    print("🔊 Step 1: Generating AI Personality Analysis...")
    profiler = SpeakerProfilerSingle(os.getenv("OPENAI_API_KEY"))
    
    try:
        result1 = await profiler.profile_speaker(audio_file)
        analysis_audio = result1.get("audio_response")
        
        if not analysis_audio:
            print("❌ Failed to generate personality analysis audio")
            return
        
        print(f"✅ Generated: {analysis_audio}")
        print()
        
        # Step 2: Extract scores using local ASR + Gemini
        print("🤖 Step 2: Extracting Scores with Local ASR + Gemini...")
        result2 = process_personality_audio(analysis_audio)
        
        if "personality_scores" not in result2:
            print("❌ Failed to extract personality scores")
            return
        
        print("✅ Score extraction completed!")
        print()
        
        # Display final results
        print("📊 FINAL PERSONALITY SCORES:")
        print("-" * 40)
        scores = result2["personality_scores"]
        
        total_score = 0
        for dimension, data in scores.items():
            dim_name = dimension.replace('_', ' ').title()
            score = data['score']
            total_score += score
            print(f"{dim_name}: {score}/5")
            print(f"  → {data['reasoning'][:100]}...")
            print()
        
        avg_score = total_score / len(scores)
        print(f"Average Score: {avg_score:.1f}/5")
        
        return result2
        
    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        return None


async def longform_evaluation_workflow(audio_file: str, task: str = "all"):
    """Test the longform evaluation workflow with OpenAI + Gemini."""
    
    print("🎯 Long-form Evaluation Workflow (OpenAI + Gemini)")
    print("=" * 50)
    
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set")
        return None
        
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ GOOGLE_API_KEY not set")
        return None
    
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return None
    
    print(f"📁 Input audio: {audio_file}")
    print(f"🎯 Task mode: {task}")
    print("🤖 OpenAI: Response generation")
    print("🧠 Gemini: Response evaluation")
    print()
    
    try:
        # Initialize evaluator with both API keys
        evaluator = LongFormEvaluator(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Run evaluation(s)
        if task == "all":
            print("🔄 Running all longform evaluation tasks...")
            result = await evaluator.evaluate_all_tasks(audio_file)
        else:
            print(f"🔄 Running {task} evaluation...")
            result = await evaluator.evaluate_task(audio_file, task)
        
        print("✅ Longform evaluation completed!")
        return result
        
    except Exception as e:
        print(f"❌ Longform evaluation failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Complete workflow with mode selection")
    parser.add_argument("audio_file", help="Path to input WAV file")
    parser.add_argument("--mode", choices=["personality", "longform"], default="personality",
                       help="Evaluation mode: personality profiling or longform tasks (default: personality)")
    parser.add_argument("--task", help="For longform mode: specific task or 'all' (default: all)")
    parser.add_argument("--openai-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--google-key", help="Google API key (or set GOOGLE_API_KEY env var)")
    
    args = parser.parse_args()
    
    # Set API keys if provided
    if args.openai_key:
        os.environ["OPENAI_API_KEY"] = args.openai_key
    if args.google_key:
        os.environ["GOOGLE_API_KEY"] = args.google_key
    
    # Verify audio file exists
    if not os.path.exists(args.audio_file):
        print(f"❌ Audio file not found: {args.audio_file}")
        return
    
    # Run appropriate workflow
    if args.mode == "personality":
        print("🧠 Running Personality Profiling Workflow...")
        result = asyncio.run(complete_workflow_test(args.audio_file))
        if result:
            print("\n🎉 Personality profiling workflow PASSED!")
        else:
            print("\n💥 Personality profiling workflow FAILED!")
            
    elif args.mode == "longform":
        task = args.task or "all"
        print(f"📝 Running Long-form Evaluation Workflow (task: {task})...")
        result = asyncio.run(longform_evaluation_workflow(args.audio_file, task))
        if result:
            print("\n🎉 Long-form evaluation workflow PASSED!")
        else:
            print("\n💥 Long-form evaluation workflow FAILED!")


if __name__ == "__main__":
    main()