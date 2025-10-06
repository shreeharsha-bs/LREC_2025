#!/usr/bin/env python3
"""
Quick test script to verify both evaluation modes work
"""

import os
import sys
import asyncio
from pathlib import Path


def check_requirements():
    """Check if all required packages and files are available."""
    print("🔍 Checking requirements...")
    
    # Check API keys
    api_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY")
    }
    
    missing_keys = [key for key, value in api_keys.items() if not value]
    if missing_keys:
        print(f"❌ Missing API keys: {', '.join(missing_keys)}")
        return False
    
    # Check audio file
    test_audio = "s0101a_71_79.wav"
    if not os.path.exists(test_audio):
        print(f"❌ Test audio file not found: {test_audio}")
        return False
    
    # Check Python files
    required_files = [
        "speaker_profiling_single.py",
        "longform_evaluator.py", 
        "personality_score_extractor.py",
        "test_complete_workflow.py"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    
    print("✅ All requirements satisfied!")
    return True


async def test_personality_mode():
    """Test personality profiling mode."""
    print("\n" + "="*60)
    print("🧠 TESTING PERSONALITY PROFILING MODE")
    print("="*60)
    
    try:
        from speaker_profiling_single import SpeakerProfilerSingle
        
        profiler = SpeakerProfilerSingle(os.getenv("OPENAI_API_KEY"))
        result = await profiler.profile_speaker("s0101a_71_79.wav")
        
        if result and "scores" in result:
            print("✅ Personality profiling mode works!")
            print(f"Generated scores for {len(result['scores'])} dimensions")
            return True
        else:
            print("❌ Personality profiling failed")
            return False
            
    except Exception as e:
        print(f"❌ Personality profiling error: {e}")
        return False


async def test_longform_mode():
    """Test longform evaluation mode."""
    print("\n" + "="*60)
    print("📝 TESTING LONGFORM EVALUATION MODE (OpenAI + Gemini)")
    print("="*60)
    
    try:
        from longform_evaluator import LongFormEvaluator
        
        # Initialize with both API keys
        evaluator = LongFormEvaluator(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Test one task (therapy) as example
        result = await evaluator.evaluate_task("s0101a_71_79.wav", "therapy")
        
        if result and "scores" in result:
            print("✅ Longform evaluation mode works!")
            print(f"Generated scores for {len(result['scores'])} dimensions in therapy task")
            print("🤖 OpenAI response generation: OK")
            print("🧠 Gemini evaluation: OK")
            return True
        else:
            print("❌ Longform evaluation failed")
            return False
            
    except Exception as e:
        print(f"❌ Longform evaluation error: {e}")
        return False


async def main():
    """Main test function."""
    print("🚀 COMPREHENSIVE EVALUATION SYSTEM TEST")
    print("="*60)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements check failed. Please fix the issues above.")
        return False
    
    # Test both modes
    personality_ok = await test_personality_mode()
    longform_ok = await test_longform_mode()
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    print(f"Personality Profiling: {'✅ PASS' if personality_ok else '❌ FAIL'}")
    print(f"Longform Evaluation:   {'✅ PASS' if longform_ok else '❌ FAIL'}")
    
    if personality_ok and longform_ok:
        print("\n🎉 ALL TESTS PASSED! Both evaluation modes are working.")
        print("\nUsage examples:")
        print("  Personality mode: python test_complete_workflow.py s0101a_71_79.wav --mode personality")
        print("  Longform mode:    python test_complete_workflow.py s0101a_71_79.wav --mode longform")
        print("  Specific task:    python test_complete_workflow.py s0101a_71_79.wav --mode longform --task therapy")
        return True
    else:
        print("\n💥 SOME TESTS FAILED! Please check the errors above.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)