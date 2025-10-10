#!/usr/bin/env python3
"""
Comprehensive Pipeline Runner
Runs complete 3-step pipeline for speaker profiling or longform evaluation.
Automatically generates filenames based on input audio file.
"""

import os
import sys
import asyncio
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class PipelineRunner:
    def __init__(self, input_audio: str, pipeline_type: str, task: str = None):
        self.input_audio = Path(input_audio)
        self.pipeline_type = pipeline_type  # 'profiling' or 'longform'
        self.task = task  # Only for longform: therapy, career, interview, story
        
        # Verify input file exists
        if not self.input_audio.exists():
            raise FileNotFoundError(f"Input audio file not found: {input_audio}")
        
        # Generate base filename from input audio
        self.base_name = self.input_audio.stem  # e.g., "s0101a_71_79"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get API keys
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.google_key = os.getenv("GOOGLE_API_KEY")
        
        if not self.openai_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        if not self.google_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")
    
    def generate_filenames(self):
        """Generate all filenames for the pipeline."""
        if self.pipeline_type == "profiling":
            prefix = f"{self.base_name}_profiling"
            return {
                'audio': f"{prefix}_{self.timestamp}.wav",
                'transcript': f"{prefix}_{self.timestamp}_transcript.txt",
                'scores': f"{prefix}_{self.timestamp}_scores.json",
            }
        else:  # longform
            prefix = f"{self.base_name}_{self.task}"
            return {
                'audio': f"{prefix}_{self.timestamp}.wav",
                'transcript': f"{prefix}_{self.timestamp}_transcript.txt",
                'scores': f"{prefix}_{self.timestamp}_scores.json",
            }
    
    def run_command(self, command: list, step_name: str) -> bool:
        """Run a command and handle errors."""
        print(f"\n{'='*70}")
        print(f"STEP: {step_name}")
        print(f"{'='*70}")
        print(f"Command: {' '.join(command)}")
        print(f"{'='*70}\n")
        
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=False,  # Show output in real-time
                text=True
            )
            print(f"\n✓ {step_name} completed successfully\n")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\n✗ {step_name} failed with error code {e.returncode}\n")
            return False
        except Exception as e:
            print(f"\n✗ {step_name} failed: {e}\n")
            return False
    
    def run_speaker_profiling(self):
        """Run complete speaker profiling pipeline."""
        print(f"\n{'#'*70}")
        print(f"# SPEAKER PROFILING PIPELINE")
        print(f"# Input: {self.input_audio}")
        print(f"# Base name: {self.base_name}")
        print(f"# Timestamp: {self.timestamp}")
        print(f"{'#'*70}\n")
        
        filenames = self.generate_filenames()
        
        # Step 1: Generate personality analysis audio
        step1_cmd = [
            "python", "1_speaker_profiling_generate.py",
            str(self.input_audio),
            filenames['audio'],
            "--api-key", self.openai_key
        ]
        if not self.run_command(step1_cmd, "Step 1: Generate Personality Analysis"):
            return False
        
        # Step 2: Transcribe analysis
        audio_path = Path("speaker_profiles_audio") / filenames['audio']
        step2_cmd = [
            "python", "2_speaker_profiling_transcribe.py",
            str(audio_path),
            filenames['transcript']
        ]
        if not self.run_command(step2_cmd, "Step 2: Transcribe Analysis"):
            return False
        
        # Step 3: Extract scores
        transcript_path = Path("speaker_profiles_transcripts") / filenames['transcript']
        step3_cmd = [
            "python", "3_speaker_profiling_evaluate.py",
            str(transcript_path),
            filenames['audio'],
            filenames['scores'],
            "--api-key", self.google_key
        ]
        if not self.run_command(step3_cmd, "Step 3: Extract Personality Scores"):
            return False
        
        # Success summary
        print(f"\n{'='*70}")
        print(f"✓ SPEAKER PROFILING PIPELINE COMPLETED SUCCESSFULLY")
        print(f"{'='*70}")
        print(f"\nOutput files:")
        print(f"  Audio:      speaker_profiles_audio/{filenames['audio']}")
        print(f"  Transcript: speaker_profiles_transcripts/{filenames['transcript']}")
        print(f"  Scores:     speaker_profiles_results/{filenames['scores']}")
        print(f"\n{'='*70}\n")
        
        return True
    
    def run_longform_evaluation(self):
        """Run complete longform evaluation pipeline."""
        print(f"\n{'#'*70}")
        print(f"# LONGFORM EVALUATION PIPELINE - {self.task.upper()}")
        print(f"# Input: {self.input_audio}")
        print(f"# Base name: {self.base_name}")
        print(f"# Task: {self.task}")
        print(f"# Timestamp: {self.timestamp}")
        print(f"{'#'*70}\n")
        
        filenames = self.generate_filenames()
        
        # Step 4: Generate task response audio
        step4_cmd = [
            "python", "4_longform_generate.py",
            str(self.input_audio),
            self.task,
            filenames['audio'],
            "--api-key", self.openai_key
        ]
        if not self.run_command(step4_cmd, f"Step 4: Generate {self.task.title()} Response"):
            return False
        
        # Step 5: Transcribe response
        audio_path = Path("longform_audio") / filenames['audio']
        step5_cmd = [
            "python", "5_longform_transcribe.py",
            str(audio_path),
            filenames['transcript']
        ]
        if not self.run_command(step5_cmd, "Step 5: Transcribe Response"):
            return False
        
        # Step 6: Evaluate response
        transcript_path = Path("longform_transcripts") / filenames['transcript']
        step6_cmd = [
            "python", "6_longform_evaluate.py",
            str(transcript_path),
            self.task,
            filenames['audio'],
            filenames['scores'],
            "--api-key", self.google_key
        ]
        if not self.run_command(step6_cmd, f"Step 6: Evaluate {self.task.title()} Response"):
            return False
        
        # Success summary
        print(f"\n{'='*70}")
        print(f"✓ LONGFORM EVALUATION PIPELINE COMPLETED SUCCESSFULLY")
        print(f"{'='*70}")
        print(f"\nOutput files:")
        print(f"  Audio:      longform_audio/{filenames['audio']}")
        print(f"  Metadata:   longform_audio/{filenames['audio'].replace('.wav', '_metadata.json')}")
        print(f"  Transcript: longform_transcripts/{filenames['transcript']}")
        print(f"  Scores:     longform_results/{filenames['scores']}")
        print(f"\n{'='*70}\n")
        
        return True
    
    def run(self):
        """Run the appropriate pipeline."""
        if self.pipeline_type == "profiling":
            return self.run_speaker_profiling()
        else:
            return self.run_longform_evaluation()


def main():
    parser = argparse.ArgumentParser(
        description="Run complete 3-step pipeline for speaker profiling or longform evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Speaker profiling
  python run_full_pipeline.py profiling input.wav
  
  # Longform evaluation - therapy task
  python run_full_pipeline.py longform input.wav --task therapy
  
  # Longform evaluation - career task
  python run_full_pipeline.py longform input.wav --task career
  
  # With explicit API keys
  python run_full_pipeline.py profiling input.wav --openai-key sk-... --google-key AIza...

Output files are automatically named based on input filename with timestamp.
        """
    )
    
    parser.add_argument(
        "pipeline",
        choices=["profiling", "longform"],
        help="Pipeline type to run"
    )
    parser.add_argument(
        "input_audio",
        help="Path to input WAV file"
    )
    parser.add_argument(
        "--task",
        choices=["therapy", "career", "interview", "story"],
        help="Task type (required for longform pipeline)"
    )
    parser.add_argument(
        "--openai-key",
        help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--google-key",
        help="Google API key (or set GOOGLE_API_KEY env var)"
    )
    
    args = parser.parse_args()
    
    # Validate task for longform
    if args.pipeline == "longform" and not args.task:
        parser.error("--task is required for longform pipeline")
    
    # Override environment variables if provided
    if args.openai_key:
        os.environ["OPENAI_API_KEY"] = args.openai_key
    if args.google_key:
        os.environ["GOOGLE_API_KEY"] = args.google_key
    
    try:
        # Create and run pipeline
        runner = PipelineRunner(args.input_audio, args.pipeline, args.task)
        success = runner.run()
        
        return 0 if success else 1
        
    except FileNotFoundError as e:
        print(f"\n✗ ERROR: {e}\n")
        return 1
    except ValueError as e:
        print(f"\n✗ ERROR: {e}\n")
        print("Make sure API keys are set in .env file or passed as arguments.\n")
        return 1
    except KeyboardInterrupt:
        print("\n\n✗ Pipeline interrupted by user\n")
        return 1
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
