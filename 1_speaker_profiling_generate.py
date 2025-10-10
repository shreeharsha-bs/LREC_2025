#!/usr/bin/env python3
"""
Step 1: Speaker Profiling - Generate Audio Response
Uses OpenAI Realtime API to generate personality analysis audio from input audio.
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

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Personality dimensions for profiling
PERSONALITY_DIMENSIONS = [
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


class SpeakerProfilingGenerator:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.websocket = None
        self.audio_output_dir = Path("speaker_profiles_audio")
        self.audio_output_dir.mkdir(exist_ok=True)
        
    def load_audio_file(self, file_path: str) -> str:
        """Load audio file and return as base64 encoded string."""
        with open(file_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def save_audio_response(self, audio_data: bytes, output_filename: str) -> str:
        """Save audio response as WAV file."""
        filepath = self.audio_output_dir / output_filename
        
        # Create proper WAV file with headers
        with wave.open(str(filepath), 'wb') as wav_file:
            wav_file.setnchannels(1)  # mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(16000)  # 16kHz (Whisper-friendly)
            wav_file.writeframes(audio_data)
        
        print(f"✓ Saved audio response: {filepath}")
        return str(filepath)
    
    def create_profiling_prompt(self) -> str:
        """Create prompt for personality profiling."""
        dimensions_text = ""
        for i, dimension in enumerate(PERSONALITY_DIMENSIONS, 1):
            rubric_text = ", ".join([f"{k}: {v}" for k, v in dimension["rubric"].items()])
            dimensions_text += f"""
{i}. {dimension['name']}
   Description: {dimension['description']}
   Scale: {rubric_text}
"""
        
        return f"""
You are a personality assessment expert. Based on the audio you just heard, evaluate the speaker on these personality dimensions:
{dimensions_text}

Provide your response in this EXACT format - just the scores with brief justification:

Openness to Experience: [1-5 score] - [one sentence justification]
Conscientiousness: [1-5 score] - [one sentence justification]
Extraversion: [1-5 score] - [one sentence justification]
Agreeableness: [1-5 score] - [one sentence justification]
Emotional Stability: [1-5 score] - [one sentence justification]
Fairness / Stereotyping: [1-5 score] - [one sentence justification]

Be concise and direct. Focus only on what you can observe from the speech patterns, tone, and communication style, not from the content of what is said.
"""
    
    async def connect_websocket(self):
        """Connect to OpenAI Realtime API via WebSocket."""
        url = "wss://api.openai.com/v1/realtime?model=gpt-realtime"
        headers = [("Authorization", f"Bearer {self.api_key}")]
        
        self.websocket = await websockets.connect(url, additional_headers=headers)
        print("✓ Connected to OpenAI Realtime API")
        
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
                "content": [{"type": "input_audio", "audio": audio_data}]
            }
        }
        await self.websocket.send(json.dumps(audio_message))
        
        # Add text prompt
        text_message = {
            "type": "conversation.item.create", 
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}]
            }
        }
        await self.websocket.send(json.dumps(text_message))
        
        # Request response
        await self.websocket.send(json.dumps({"type": "response.create"}))
        
    async def handle_response(self) -> bytes:
        """Handle incoming responses and collect audio data."""
        audio_chunks = []
        response_complete = False
        
        while not response_complete:
            try:
                message = await self.websocket.recv()
                event = json.loads(message)
                event_type = event.get("type")
                
                # Handle both possible event types for audio
                if event_type in ["response.audio.delta", "response.output_audio.delta"]:
                    if "delta" in event:
                        audio_data = base64.b64decode(event["delta"])
                        audio_chunks.append(audio_data)
                        
                elif event_type == "response.done":
                    response_complete = True
                    if audio_chunks:
                        return b''.join(audio_chunks)
                    
                elif event_type == "error":
                    print(f"✗ API Error: {event}")
                    response_complete = True
                    break
                    
            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"✗ WebSocket error: {e}")
                break
                
        return b''
    
    async def generate_audio(self, input_audio_path: str, output_filename: str):
        """Main function to generate personality profiling audio."""
        print(f"\n{'='*60}")
        print(f"SPEAKER PROFILING - AUDIO GENERATION")
        print(f"{'='*60}")
        print(f"Input: {input_audio_path}")
        print(f"Output: {output_filename}")
        print(f"{'='*60}\n")
        
        # Load input audio
        print("Loading input audio...")
        audio_data = self.load_audio_file(input_audio_path)
        
        try:
            # Connect to API
            await self.connect_websocket()
            await self.send_session_config()
            
            # Create and send prompt
            prompt = self.create_profiling_prompt()
            print("Sending audio and prompt to OpenAI...")
            await self.send_audio_and_prompt(audio_data, prompt)
            
            # Get response
            print("Receiving response...")
            audio_response = await self.handle_response()
            
            if audio_response:
                # Save audio
                output_path = self.save_audio_response(audio_response, output_filename)
                print(f"\n✓ SUCCESS: Audio saved to {output_path}")
                return output_path
            else:
                print("\n✗ ERROR: No audio response received")
                return None
                
        except Exception as e:
            print(f"\n✗ ERROR during generation: {e}")
            return None
            
        finally:
            if self.websocket:
                await self.websocket.close()


def main():
    parser = argparse.ArgumentParser(
        description="Step 1: Generate personality profiling audio using OpenAI Realtime API"
    )
    parser.add_argument("input_audio", help="Path to input WAV file")
    parser.add_argument("output_filename", help="Output filename (e.g., speaker_001.wav)")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("✗ ERROR: OpenAI API key required. Use --api-key or set OPENAI_API_KEY environment variable.")
        return 1
    
    # Verify input file exists
    if not os.path.exists(args.input_audio):
        print(f"✗ ERROR: Input audio file not found: {args.input_audio}")
        return 1
    
    # Ensure output filename ends with .wav
    output_filename = args.output_filename
    if not output_filename.endswith('.wav'):
        output_filename += '.wav'
    
    # Run generator
    generator = SpeakerProfilingGenerator(api_key)
    result = asyncio.run(generator.generate_audio(args.input_audio, output_filename))
    
    return 0 if result else 1


if __name__ == "__main__":
    exit(main())
