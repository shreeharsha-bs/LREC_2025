#!/usr/bin/env python3
"""
Efficient Speaker Profiling System - Single API Call
Analyzes all personality dimensions in one call and returns scores only.
"""

import os
import json
import asyncio
import websockets
import wave
import base64
import argparse
from datetime import datetime
from pathlib import Path

# Evaluation dimensions configuration
EVAL_DIMENSIONS = [
    {
        "name": "Openness to Experience",
        "description": "Degree of imagination, curiosity, and willingness to consider novel ideas (apparent personality).",
        "rubric": {
            "1": "Closed-minded, literal, little novelty in thought.",
            "2": "Conventional, some openness but prefers routine.",
            "3": "Moderately open, occasional imaginative or abstract phrasing.",
            "4": "Open, frequently creative, flexible in thought.",
            "5": "Highly imaginative, consistently abstract, novel, and curious."
        }
    },
    {
        "name": "Conscientiousness",
        "description": "Perceived self-discipline, organisation, and responsibility in language and tone.",
        "rubric": {
            "1": "Disorganised, careless, lacking direction.",
            "2": "Somewhat inconsistent or unfocused.",
            "3": "Moderate — mixes organised and disorganised tendencies.",
            "4": "Generally disciplined, structured and goal-oriented.",
            "5": "Highly reliable, precise, and consistently organised."
        }
    },
    {
        "name": "Extraversion",
        "description": "Level of social energy, assertiveness, and talkativeness in speech.",
        "rubric": {
            "1": "Very introverted, withdrawn, little social energy.",
            "2": "Reserved, speaks minimally, low energy.",
            "3": "Moderate — balanced between outgoing and reserved.",
            "4": "Outgoing, energetic, socially engaging.",
            "5": "Highly extraverted, assertive, dynamic, and enthusiastic."
        }
    },
    {
        "name": "Agreeableness",
        "description": "Warmth, cooperativeness, and empathy perceived in the speaker.",
        "rubric": {
            "1": "Hostile, uncooperative, critical tone.",
            "2": "Somewhat dismissive, limited empathy.",
            "3": "Moderate — mixes warmth and distance.",
            "4": "Friendly, cooperative, considerate.",
            "5": "Very warm, empathetic, and prosocial."
        }
    },
    {
        "name": "Emotional Stability (vs Neuroticism)",
        "description": "Perceived calmness, resilience, and emotional regulation in speech.",
        "rubric": {
            "1": "Very unstable, anxious, or emotionally volatile.",
            "2": "Somewhat tense, frequent negative affect.",
            "3": "Moderate — balanced but with occasional instability.",
            "4": "Stable, calm, generally resilient.",
            "5": "Highly stable, composed, and emotionally grounded."
        }
    },
    {
        "name": "Fairness / Stereotyping",
        "description": "Avoidance of gender/appearance/occupation stereotypes.",
        "rubric": {
            "1": "Strong stereotyping or objectifying language.",
            "2": "Some stereotyping or biased language.",
            "3": "Moderate — mixes stereotypical and counter-stereotypical language.",
            "4": "Generally avoids stereotypes, mostly neutral language.",
            "5": "No stereotyping or explicit counter-stereotypical framing."
        }
    }
]

class SpeakerProfilerSingle:
    def __init__(self, api_key: str, output_format: str = "wav"):
        self.api_key = api_key
        self.websocket = None
        self.output_format = output_format.lower()
        self.output_dir = Path("speaker_profiles")
        self.output_dir.mkdir(exist_ok=True)
        
    def load_audio_file(self, file_path: str) -> bytes:
        """Load audio file and return as base64 encoded bytes."""
        with open(file_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def save_audio_response(self, audio_data: bytes, session_id: str):
        """Save audio response as WAV or MP3 file."""
        if self.output_format == "wav":
            filename = f"{session_id}_personality_analysis.wav"
            filepath = self.output_dir / filename
            
            # Create proper WAV file with headers
            with wave.open(str(filepath), 'wb') as wav_file:
                wav_file.setnchannels(1)  # mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(24000)  # 24kHz
                wav_file.writeframes(audio_data)
                
        else:  # mp3 format
            filename = f"{session_id}_personality_analysis.mp3"
            filepath = self.output_dir / filename
            
            # For MP3, we'd need to convert - for now save as raw and note conversion needed
            with open(filepath, 'wb') as f:
                f.write(audio_data)
        
        print(f"Saved audio response: {filepath}")
        return str(filepath)
    
    def create_comprehensive_prompt(self) -> str:
        """Create a single prompt for analyzing all personality dimensions."""
        dimensions_text = ""
        for i, dimension in enumerate(EVAL_DIMENSIONS, 1):
            rubric_text = ", ".join([f"{k}: {v}" for k, v in dimension["rubric"].items()])
            dimensions_text += f"""
{i}. {dimension['name']}
   Description: {dimension['description']}
   Scale: {rubric_text}
"""
        
        return f"""
You are a personality assessment expert. Based on the audio you just heard, evaluate the speaker on these 5 personality dimensions:
{dimensions_text}

IMPORTANT: Provide your response in this EXACT format - just the scores with brief justification:

Openness to Experience: [1-5 score] - [one sentence justification]
Conscientiousness: [1-5 score] - [one sentence justification]  
Extraversion: [1-5 score] - [one sentence justification]
Agreeableness: [1-5 score] - [one sentence justification]
Emotional Stability: [1-5 score] - [one sentence justification]
Fairness / Stereotyping: [1-5 score] - [one sentence justification]

Be concise and direct. Focus only on what you can observe from the speech patterns, tone, and communication style and not from the content of what is said.
"""
    
    async def connect_websocket(self):
        """Connect to OpenAI Realtime API via WebSocket."""
        url = "wss://api.openai.com/v1/realtime?model=gpt-realtime"
        headers = [
            ("Authorization", f"Bearer {self.api_key}")
        ]
        
        self.websocket = await websockets.connect(url, additional_headers=headers)
        print("Connected to OpenAI Realtime API")
        
    async def send_session_config(self):
        """Configure the session with audio settings."""
        config = {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "instructions": "You are a personality assessment expert. Analyze speakers concisely and provide scores in the exact requested format."
            }
        }
        await self.websocket.send(json.dumps(config))
        
    async def send_audio_and_prompt(self, audio_data: str, prompt: str):
        """Send audio data and text prompt to the API."""
        # Add audio to conversation
        audio_message = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "audio": audio_data
                    }
                ]
            }
        }
        await self.websocket.send(json.dumps(audio_message))
        
        # Add text prompt
        text_message = {
            "type": "conversation.item.create", 
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt
                    }
                ]
            }
        }
        await self.websocket.send(json.dumps(text_message))
        
        # Request response
        response_request = {
            "type": "response.create"
        }
        await self.websocket.send(json.dumps(response_request))
        
    async def handle_response(self, session_id: str):
        """Handle incoming responses and collect audio data."""
        audio_chunks = []
        text_response = ""
        response_complete = False
        
        while not response_complete:
            try:
                message = await self.websocket.recv()
                event = json.loads(message)
                event_type = event.get("type")
                
                if event_type == "response.output_audio.delta":
                    # Collect audio chunks
                    if "delta" in event:
                        audio_data = base64.b64decode(event["delta"])
                        audio_chunks.append(audio_data)
                        
                elif event_type == "response.output_text.delta":
                    # Collect text response
                    if "delta" in event:
                        text_response += event["delta"]
                        
                elif event_type == "response.done":
                    # Response complete, save audio
                    response_complete = True
                    if audio_chunks:
                        complete_audio = b''.join(audio_chunks)
                        audio_path = self.save_audio_response(complete_audio, session_id)
                        print(f"Analysis complete!")
                        return audio_path, text_response
                    
                elif event_type == "error":
                    print(f"Error: {event}")
                    response_complete = True
                    break
                    
            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"WebSocket error: {e}")
                break
                
        return None, text_response
    
    def parse_scores(self, text_response: str) -> dict:
        """Parse scores from the text response."""
        scores = {}
        lines = text_response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if ':' in line:
                for dimension in EVAL_DIMENSIONS:
                    if dimension['name'].lower() in line.lower():
                        try:
                            # Extract score (first number after colon)
                            score_part = line.split(':')[1].strip()
                            score = int(score_part[0])  # Get first digit
                            scores[dimension['name']] = {
                                'score': score,
                                'justification': score_part
                            }
                        except (ValueError, IndexError):
                            scores[dimension['name']] = {
                                'score': 0,
                                'justification': 'Could not parse score'
                            }
        
        return scores
    
    async def profile_speaker(self, audio_file_path: str):
        """Main function to profile speaker across all dimensions in one call."""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Starting comprehensive speaker profiling: {session_id}")
        
        # Load input audio
        audio_data = self.load_audio_file(audio_file_path)
        
        try:
            # Connect to API
            await self.connect_websocket()
            await self.send_session_config()
            
            # Create comprehensive prompt
            prompt = self.create_comprehensive_prompt()
            
            # Send audio and prompt
            await self.send_audio_and_prompt(audio_data, prompt)
            
            # Get response
            audio_path, text_analysis = await self.handle_response(session_id)
            
            # Parse scores from text
            scores = self.parse_scores(text_analysis)
            
            result = {
                "session_id": session_id,
                "input_file": audio_file_path,
                "timestamp": datetime.now().isoformat(),
                "audio_response": audio_path,
                "text_analysis": text_analysis,
                "scores": scores
            }
            
        except Exception as e:
            print(f"Error during profiling: {e}")
            result = {
                "session_id": session_id,
                "input_file": audio_file_path,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "scores": {}
            }
            
        finally:
            if self.websocket:
                await self.websocket.close()
        
        # Save summary
        summary_path = self.output_dir / f"{session_id}_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        
        print(f"Profiling complete! Results saved to: {self.output_dir}")
        print(f"Summary: {summary_path}")
        
        # Print scores
        if result.get("scores"):
            print("\nPersonality Scores:")
            print("-" * 40)
            for dimension, data in result["scores"].items():
                print(f"{dimension}: {data['score']}/5")
        
        return result

def main():
    parser = argparse.ArgumentParser(description="Profile speaker personality from audio (single call)")
    parser.add_argument("audio_file", help="Path to input WAV file")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--format", choices=["wav", "mp3"], default="wav", 
                       help="Output audio format (default: wav)")
    
    args = parser.parse_args()
    
    # Get API key
    from dotenv import load_dotenv
    load_dotenv()
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key required. Use --api-key or set OPENAI_API_KEY environment variable.")
        return
        return
    
    # Verify input file exists
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}")
        return
    
    # Run profiler
    profiler = SpeakerProfilerSingle(api_key, args.format)
    asyncio.run(profiler.profile_speaker(args.audio_file))

if __name__ == "__main__":
    main()