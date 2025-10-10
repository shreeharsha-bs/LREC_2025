#!/usr/bin/env python3
"""
Step 2: Speaker Profiling - Transcribe Audio
Uses Whisper ASR to transcribe the personality analysis audio.
"""

import os
import argparse
from pathlib import Path

try:
    from transformers import pipeline
    import torch
except ImportError:
    print("✗ ERROR: transformers library not installed")
    print("Install with: pip install transformers torch")
    exit(1)


class SpeakerProfilingTranscriber:
    def __init__(self):
        self.transcript_output_dir = Path("speaker_profiles_transcripts")
        self.transcript_output_dir.mkdir(exist_ok=True)
        self.whisper_pipeline = None
        
    def initialize_whisper(self):
        """Initialize Whisper ASR model."""
        print("Loading Whisper model from HuggingFace...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper_pipeline = pipeline(
            "automatic-speech-recognition", 
            model="openai/whisper-small",
            device=device
        )
        print(f"✓ Whisper model loaded on {device}")
        
    def transcribe_audio(self, audio_path: str, output_filename: str) -> str:
        """Transcribe audio file using Whisper."""
        print(f"\n{'='*60}")
        print(f"SPEAKER PROFILING - TRANSCRIPTION")
        print(f"{'='*60}")
        print(f"Input: {audio_path}")
        print(f"Output: {output_filename}")
        print(f"{'='*60}\n")
        
        if self.whisper_pipeline is None:
            self.initialize_whisper()

        print("Transcribing audio...")
        try:
            result = self.whisper_pipeline(
                audio_path,
                return_timestamps=True,
                chunk_length_s=30,
                generate_kwargs={"language": "en", "task": "transcribe"}
            )
            transcript = result['text']
            
            # Save transcript
            output_path = self.transcript_output_dir / output_filename
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(transcript)
            
            print(f"✓ Transcription completed ({len(transcript)} characters)")
            print(f"✓ Saved to: {output_path}")
            print(f"\nTranscript Preview:")
            print(f"{'-'*60}")
            print(transcript[:500] + ("..." if len(transcript) > 500 else ""))
            print(f"{'-'*60}\n")
            
            return str(output_path)
            
        except Exception as e:
            print(f"✗ ERROR during transcription: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Transcribe personality profiling audio using Whisper ASR"
    )
    parser.add_argument("input_audio", help="Path to audio WAV file to transcribe")
    parser.add_argument("output_filename", help="Output transcript filename (e.g., speaker_001_transcript.txt)")
    
    args = parser.parse_args()
    
    # Verify input file exists
    if not os.path.exists(args.input_audio):
        print(f"✗ ERROR: Input audio file not found: {args.input_audio}")
        return 1
    
    # Ensure output filename ends with .txt
    output_filename = args.output_filename
    if not output_filename.endswith('.txt'):
        output_filename += '.txt'
    
    # Run transcriber
    transcriber = SpeakerProfilingTranscriber()
    result = transcriber.transcribe_audio(args.input_audio, output_filename)
    
    return 0 if result else 1


if __name__ == "__main__":
    exit(main())
