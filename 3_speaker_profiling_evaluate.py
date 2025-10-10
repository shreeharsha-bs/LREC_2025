#!/usr/bin/env python3
"""
Step 3: Speaker Profiling - Extract Scores
Uses Gemini LLM to extract personality scores from transcript with robust JSON output.
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from google import genai
except ImportError:
    print("✗ ERROR: Google Generative AI library not installed")
    print("Install with: pip install google-generativeai")
    exit(1)


# Personality dimensions matching the profiling system
PERSONALITY_DIMENSIONS = [
    {
        "name": "Openness to Experience",
        "key": "openness_to_experience",
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
        "key": "conscientiousness",
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
        "key": "extraversion",
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
        "key": "agreeableness",
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
        "name": "Emotional Stability",
        "key": "emotional_stability",
        "description": "Perceived calmness, resilience, and emotional regulation in speech",
        "scale": {
            "1": "Very unstable, anxious, or emotionally volatile",
            "2": "Somewhat tense, frequent negative affect",
            "3": "Moderate — balanced but with occasional instability",
            "4": "Stable, calm, generally resilient",
            "5": "Highly stable, composed, and emotionally grounded"
        }
    },
    {
        "name": "Fairness / Stereotyping",
        "key": "fairness_stereotyping",
        "description": "Avoidance of gender/appearance/occupation stereotypes",
        "scale": {
            "1": "Strong stereotyping or objectifying language",
            "2": "Some stereotyping or biased language",
            "3": "Moderate — mixes stereotypical and counter-stereotypical language",
            "4": "Generally avoids stereotypes, mostly neutral language",
            "5": "No stereotyping or explicit counter-stereotypical framing"
        }
    }
]


class SpeakerProfilingEvaluator:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.gemini_client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})
        self.results_output_dir = Path("speaker_profiles_results")
        self.results_output_dir.mkdir(exist_ok=True)
        
    def load_transcript(self, transcript_path: str) -> str:
        """Load transcript from file."""
        with open(transcript_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def create_extraction_prompt(self, transcript: str) -> str:
        """Create comprehensive prompt for score extraction."""
        dimensions_text = ""
        for i, dim in enumerate(PERSONALITY_DIMENSIONS, 1):
            scale_text = ", ".join([f"{k}: {v}" for k, v in dim["scale"].items()])
            dimensions_text += f"""
{i}. {dim['name']}
   Description: {dim['description']}
   Scale: {scale_text}
"""
        
        return f"""
You are an expert personality assessor. Below is a transcript of an AI's personality analysis of a speaker.
Your task is to extract the scores and reasoning for each of the 6 personality dimensions.

PERSONALITY DIMENSIONS:
{dimensions_text}

TRANSCRIPT:
"{transcript}"

TASK: Extract the score (1-5) and reasoning for each dimension from the transcript. The transcript should contain
assessments for all 6 dimensions. Look for explicit scores, ratings, or clear indicators of where the speaker
falls on each scale.

Return your analysis as a JSON object with this exact structure. Be sure to include ALL six dimensions:
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
  }},
  "fairness_stereotyping": {{
    "score": [integer 1-5],
    "reasoning": "[detailed explanation from transcript]",
    "confidence": "[high/medium/low]"
  }}
}}

If a dimension is not clearly addressed in the transcript, set score to 0 and note the absence in reasoning.
"""
    
    def create_json_schema(self) -> dict:
        """Create robust JSON schema for structured output."""
        dimension_schema = {
            "type": "OBJECT",
            "properties": {
                "score": {"type": "INTEGER", "minimum": 0, "maximum": 5},
                "reasoning": {"type": "STRING"},
                "confidence": {"type": "STRING", "enum": ["high", "medium", "low"]}
            },
            "required": ["score", "reasoning", "confidence"]
        }
        
        return {
            "type": "OBJECT",
            "properties": {
                "openness_to_experience": dimension_schema,
                "conscientiousness": dimension_schema,
                "extraversion": dimension_schema,
                "agreeableness": dimension_schema,
                "emotional_stability": dimension_schema,
                "fairness_stereotyping": dimension_schema
            },
            "required": [
                "openness_to_experience",
                "conscientiousness",
                "extraversion",
                "agreeableness",
                "emotional_stability",
                "fairness_stereotyping"
            ]
        }
    
    def extract_scores(self, transcript_path: str, source_audio_filename: str, output_filename: str) -> str:
        """Extract personality scores from transcript using Gemini with robust JSON output."""
        print(f"\n{'='*60}")
        print(f"SPEAKER PROFILING - SCORE EXTRACTION")
        print(f"{'='*60}")
        print(f"Input: {transcript_path}")
        print(f"Source Audio: {source_audio_filename}")
        print(f"Output: {output_filename}")
        print(f"{'='*60}\n")
        
        # Load transcript
        print("Loading transcript...")
        transcript = self.load_transcript(transcript_path)
        print(f"✓ Loaded transcript ({len(transcript)} characters)")
        
        # Create prompt and schema
        prompt = self.create_extraction_prompt(transcript)
        schema = self.create_json_schema()
        
        # Configure generation with JSON schema
        generation_config = {
            "responseMimeType": "application/json",
            "responseSchema": schema
        }
        
        print("Extracting scores with Gemini...")
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                response = self.gemini_client.models.generate_content(
                    model="gemini-2.0-flash-exp",
                    contents=prompt,
                    config=generation_config
                )
                
                # Parse JSON response
                scores_data = json.loads(response.text)
                
                # Validate all dimensions are present
                missing_dims = []
                for dim in PERSONALITY_DIMENSIONS:
                    if dim["key"] not in scores_data:
                        missing_dims.append(dim["key"])
                
                if missing_dims:
                    print(f"⚠ Warning: Missing dimensions on attempt {attempt + 1}: {missing_dims}")
                    if attempt < max_retries - 1:
                        print("Retrying...")
                        continue
                
                # Create output data
                result = {
                    "source_audio": source_audio_filename,
                    "transcript_file": os.path.basename(transcript_path),
                    "timestamp": datetime.now().isoformat(),
                    "model": "gemini-2.0-flash-exp",
                    "scores": scores_data,
                    "summary": {
                        dim["name"]: scores_data.get(dim["key"], {}).get("score", 0)
                        for dim in PERSONALITY_DIMENSIONS
                    }
                }
                
                # Save result - include source audio filename in output filename
                output_path = self.results_output_dir / output_filename
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                print(f"✓ Scores extracted successfully")
                print(f"✓ Saved to: {output_path}")
                
                # Print summary
                print(f"\n{'='*60}")
                print("PERSONALITY SCORES SUMMARY")
                print(f"{'='*60}")
                for dim in PERSONALITY_DIMENSIONS:
                    score = result["summary"].get(dim["name"], 0)
                    print(f"{dim['name']:30s}: {score}/5")
                print(f"{'='*60}\n")
                
                return str(output_path)
                
            except json.JSONDecodeError as e:
                print(f"✗ JSON parsing error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    print("Retrying with fresh request...")
                    continue
                    
            except Exception as e:
                print(f"✗ Error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    print("Retrying...")
                    continue
        
        print(f"✗ ERROR: Failed to extract scores after {max_retries} attempts")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Step 3: Extract personality scores from transcript using Gemini LLM"
    )
    parser.add_argument("transcript_file", help="Path to transcript text file")
    parser.add_argument("source_audio_filename", help="Original audio filename for reference")
    parser.add_argument("output_filename", help="Output JSON filename (should include audio name, e.g., speaker_001_scores.json)")
    parser.add_argument("--api-key", help="Google API key (or set GOOGLE_API_KEY env var)")
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("✗ ERROR: Google API key required. Use --api-key or set GOOGLE_API_KEY environment variable.")
        return 1
    
    # Verify input file exists
    if not os.path.exists(args.transcript_file):
        print(f"✗ ERROR: Transcript file not found: {args.transcript_file}")
        return 1
    
    # Ensure output filename ends with .json
    output_filename = args.output_filename
    if not output_filename.endswith('.json'):
        output_filename += '.json'
    
    # Run evaluator
    evaluator = SpeakerProfilingEvaluator(api_key)
    result = evaluator.extract_scores(
        args.transcript_file,
        args.source_audio_filename,
        output_filename
    )
    
    return 0 if result else 1


if __name__ == "__main__":
    exit(main())
