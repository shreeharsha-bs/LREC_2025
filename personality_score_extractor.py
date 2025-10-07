#!/usr/bin/env python3
"""
Personality Score Extractor from Audio
Uses Whisper ASR to transcribe personality analysis audio, 
then Gemini to extract scores and reasoning.
"""

import json
import os
import argparse
import time
from pathlib import Path
from datetime import datetime
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: dotenv not available, using environment variables directly")

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Error: Google Generative AI library not installed")
    print("Install with: pip install google-generativeai")
    exit(1)

try:
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
    import torch
    import torchaudio
except ImportError:
    print("Error: HuggingFace transformers library not installed")
    print("Install with: pip install transformers torch torchaudio")
    exit(1)

# Load environment variables
load_dotenv()

# Configuration
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    exit(1)

# Initialize clients
gemini_client = genai.Client(api_key=GOOGLE_API_KEY, http_options={'api_version': 'v1alpha'})

# Initialize local Whisper model using pipeline (simpler approach)
print("Loading Whisper model from HuggingFace...")
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=device)
print(f"Whisper model loaded on {device}")

# Personality dimensions (matching the speaker profiling system)
PERSONALITY_DIMENSIONS = [
    {
        "name": "Openness to Experience",
        "description": "Degree of imagination, curiosity, and willingness to consider novel ideas",
        "scale": {
            "1": "Closed-minded, literal, little novelty in thought",
            "2": "Conventional, some openness but prefers routine", 
            "3": "Moderately open, occasional imaginative or abstract phrasing",
            "4": "Open, frequently creative, flexible in thought",
            "5": "Highly imaginative, consistently abstract, novel, and curious"
        }
    },
    {
        "name": "Conscientiousness", 
        "description": "Perceived self-discipline, organisation, and responsibility in language and tone",
        "scale": {
            "1": "Disorganised, careless, lacking direction",
            "2": "Somewhat inconsistent or unfocused",
            "3": "Moderate — mixes organised and disorganised tendencies", 
            "4": "Generally disciplined, structured and goal-oriented",
            "5": "Highly reliable, precise, and consistently organised"
        }
    },
    {
        "name": "Extraversion",
        "description": "Level of social energy, assertiveness, and talkativeness in speech", 
        "scale": {
            "1": "Very introverted, withdrawn, little social energy",
            "2": "Reserved, speaks minimally, low energy",
            "3": "Moderate — balanced between outgoing and reserved",
            "4": "Outgoing, energetic, socially engaging", 
            "5": "Highly extraverted, assertive, dynamic, and enthusiastic"
        }
    },
    {
        "name": "Agreeableness",
        "description": "Warmth, cooperativeness, and empathy perceived in the speaker",
        "scale": {
            "1": "Hostile, uncooperative, critical tone",
            "2": "Somewhat dismissive, limited empathy",
            "3": "Moderate — mixes warmth and distance",
            "4": "Friendly, cooperative, considerate",
            "5": "Very warm, empathetic, and prosocial"
        }
    },
    {
        "name": "Emotional Stability (vs Neuroticism)",
        "description": "Perceived calmness, resilience, and emotional regulation in speech",
        "scale": {
            "1": "Very unstable, anxious, or emotionally volatile",
            "2": "Somewhat tense, frequent negative affect", 
            "3": "Moderate — balanced but with occasional instability",
            "4": "Stable, calm, generally resilient",
            "5": "Highly stable, composed, and emotionally grounded"
        }
    }
]

def transcribe_audio(audio_file_path: str) -> str:
    """Transcribe audio using local HuggingFace Whisper model pipeline."""
    print(f"Transcribing audio: {audio_file_path}")
    
    try:
        # Use the pipeline with chunk_length_s for longer audio
        result = whisper_pipeline(
            audio_file_path, 
            chunk_length_s=30,  # Process in 30-second chunks
            return_timestamps=False  # Don't need timestamps, just text
        )
        transcript = result['text']
        
        print(f"Transcription completed. Length: {len(transcript)} characters")
        return transcript
        
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return ""

def extract_scores_with_gemini(transcript: str) -> dict:
    """Extract personality scores from transcript using Gemini."""
    
    # Create comprehensive prompt for score extraction
    dimensions_text = ""
    for i, dim in enumerate(PERSONALITY_DIMENSIONS, 1):
        scale_text = ", ".join([f"{k}: {v}" for k, v in dim["scale"].items()])
        dimensions_text += f"""
{i}. {dim['name']}
   Description: {dim['description']}
   Scale: {scale_text}
"""
    
    prompt = f"""
You are an expert personality assessor. Below is a transcript of an AI's personality analysis of a speaker. 
Your task is to extract the scores and reasoning for each of the 5 personality dimensions.

PERSONALITY DIMENSIONS:
{dimensions_text}

TRANSCRIPT:
"{transcript}"

TASK: Extract the score (1-5) and reasoning for each dimension from the transcript. The transcript should contain 
assessments for all 5 dimensions. Look for explicit scores, ratings, or clear indicators of where the speaker 
falls on each scale.

Return your analysis as a JSON object with this exact structure:
{{
  "openness_to_experience": {{
    "score": [integer 1-5],
    "reasoning": "[detailed explanation from transcript]",
    "confidence": "[high/medium/low]"
  }},
  "conscientiousness": {{
    "score": [integer 1-5], 
    "reasoning": "[detailed explanation from transcript]",
    "confidence": "[high/medium/low]"
  }},
  "extraversion": {{
    "score": [integer 1-5],
    "reasoning": "[detailed explanation from transcript]", 
    "confidence": "[high/medium/low]"
  }},
  "agreeableness": {{
    "score": [integer 1-5],
    "reasoning": "[detailed explanation from transcript]",
    "confidence": "[high/medium/low]"
  }},
  "emotional_stability": {{
    "score": [integer 1-5],
    "reasoning": "[detailed explanation from transcript]",
    "confidence": "[high/medium/low]"
  }}
}}

If a dimension is not clearly addressed in the transcript, set score to 0 and note the absence in reasoning.
"""

    # Define structured output schema
    generation_config = {
        "responseMimeType": "application/json",
        "responseSchema": {
            "type": "OBJECT",
            "properties": {
                "openness_to_experience": {
                    "type": "OBJECT",
                    "properties": {
                        "score": {"type": "INTEGER", "minimum": 0, "maximum": 5},
                        "reasoning": {"type": "STRING"},
                        "confidence": {"type": "STRING", "enum": ["high", "medium", "low"]}
                    },
                    "required": ["score", "reasoning", "confidence"]
                },
                "conscientiousness": {
                    "type": "OBJECT", 
                    "properties": {
                        "score": {"type": "INTEGER", "minimum": 0, "maximum": 5},
                        "reasoning": {"type": "STRING"},
                        "confidence": {"type": "STRING", "enum": ["high", "medium", "low"]}
                    },
                    "required": ["score", "reasoning", "confidence"]
                },
                "extraversion": {
                    "type": "OBJECT",
                    "properties": {
                        "score": {"type": "INTEGER", "minimum": 0, "maximum": 5},
                        "reasoning": {"type": "STRING"}, 
                        "confidence": {"type": "STRING", "enum": ["high", "medium", "low"]}
                    },
                    "required": ["score", "reasoning", "confidence"]
                },
                "agreeableness": {
                    "type": "OBJECT",
                    "properties": {
                        "score": {"type": "INTEGER", "minimum": 0, "maximum": 5},
                        "reasoning": {"type": "STRING"},
                        "confidence": {"type": "STRING", "enum": ["high", "medium", "low"]}
                    },
                    "required": ["score", "reasoning", "confidence"]
                },
                "emotional_stability": {
                    "type": "OBJECT",
                    "properties": {
                        "score": {"type": "INTEGER", "minimum": 0, "maximum": 5},
                        "reasoning": {"type": "STRING"},
                        "confidence": {"type": "STRING", "enum": ["high", "medium", "low"]}
                    },
                    "required": ["score", "reasoning", "confidence"]
                }
            },
            "required": ["openness_to_experience", "conscientiousness", "extraversion", "agreeableness", "emotional_stability"]
        }
    }
    
    try:
        print("Extracting scores with Gemini...")
        
        # Use simpler approach without structured output for now
        simple_prompt = f"""
You are an expert personality assessor. Below is a transcript of an AI's personality analysis.
Extract scores (1-5) for these 5 dimensions and provide reasoning:

1. Openness to Experience (imagination, curiosity, novel ideas)
2. Conscientiousness (self-discipline, organization, responsibility)  
3. Extraversion (social energy, assertiveness, talkativeness)
4. Agreeableness (warmth, cooperativeness, empathy)
5. Emotional Stability (calmness, resilience, emotional regulation)

TRANSCRIPT: "{transcript}"

Return JSON format:
{{
  "openness_to_experience": {{"score": X, "reasoning": "..."}},
  "conscientiousness": {{"score": X, "reasoning": "..."}},
  "extraversion": {{"score": X, "reasoning": "..."}}, 
  "agreeableness": {{"score": X, "reasoning": "..."}},
  "emotional_stability": {{"score": X, "reasoning": "..."}}
}}
"""
        
        response = gemini_client.models.generate_content(
            model="models/gemini-2.5-flash-lite-preview-06-17",
            contents=[{"parts": [{"text": simple_prompt}]}]
        )
        
        # Try to parse JSON from response
        response_text = response.text.strip()
        if response_text.startswith('```json'):
            response_text = response_text.replace('```json', '').replace('```', '').strip()
        
        result = json.loads(response_text)
        print("Score extraction completed successfully")
        return result
        
    except Exception as e:
        print(f"Error extracting scores with Gemini: {e}")
        print(f"Raw response: {response.text if 'response' in locals() else 'No response'}")
        return {}

def process_personality_audio(audio_file_path: str, summary_json_path: str = None) -> dict:
    """Main function to process personality analysis audio."""
    
    print(f"Processing personality analysis audio: {audio_file_path}")
    
    # Load existing summary if provided
    metadata = {}
    if summary_json_path and os.path.exists(summary_json_path):
        with open(summary_json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    
    # Step 1: Transcribe audio
    transcript = transcribe_audio(audio_file_path)
    
    if not transcript:
        print("Failed to transcribe audio")
        return {"error": "Transcription failed"}
    
    print(f"Transcript preview: {transcript[:200]}...")
    
    # Step 2: Extract scores with Gemini
    scores = extract_scores_with_gemini(transcript)
    
    if not scores:
        print("Failed to extract scores")
        return {"error": "Score extraction failed"}
    
    # Step 3: Compile final result
    result = {
        "audio_file": audio_file_path,
        "transcript": transcript,
        "personality_scores": scores,
        "metadata": {
            "session_id": metadata.get("session_id", "unknown"),
            "input_file": metadata.get("input_file", "unknown"),
            "timestamp": metadata.get("timestamp", "unknown"),
            "processing_timestamp": datetime.now().isoformat()
        }
    }
    
    return result

def main():
    parser = argparse.ArgumentParser(
        description="Extract personality scores from speaker profiling audio using ASR + LLM"
    )
    parser.add_argument(
        'audio_file',
        type=str,
        help='Path to the personality analysis audio file (WAV format)'
    )
    parser.add_argument(
        '--summary_json',
        type=str,
        help='Optional: Path to the summary JSON file from speaker profiling'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output JSON file path (default: auto-generated)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}")
        exit(1)
    
    # Process the audio
    result = process_personality_audio(args.audio_file, args.summary_json)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        audio_path = Path(args.audio_file)
        output_path = audio_path.parent / f"{audio_path.stem}_personality_scores.json"
    
    # Save results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_path}")
    
    # Print summary
    if "personality_scores" in result:
        print("\nPersonality Scores Summary:")
        print("-" * 50)
        for dimension, data in result["personality_scores"].items():
            dim_name = dimension.replace('_', ' ').title()
            confidence = data.get('confidence', 'N/A')
            print(f"{dim_name}: {data['score']}/5")
            print(f"  Reasoning: {data['reasoning']}")
            print()
    
    return result

if __name__ == "__main__":
    main()