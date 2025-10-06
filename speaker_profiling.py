#!/usr/bin/env python3
"""
Speaker Profiling System using OpenAI Realtime API
Analyzes personality dimensions from audio input and saves responses as WAV files.
"""

import os
import json
import asyncio
import websockets
import base64
import argparse
import wave
import struct
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
    }
]

class SpeakerProfiler:
    def __init__(self, api_key: str, output_format: str = "wav"):
        self.api_key = api_key
        self.output_format = output_format
        self.websocket = None
        self.audio_chunks = []
        self.current_dimension = 0
        self.output_dir = Path("speaker_profiles")
        self.output_dir.mkdir(exist_ok=True)
        
    def load_audio_file(self, file_path: str) -> bytes:
        """Load audio file and return as base64 encoded bytes."""
        with open(file_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def save_audio_response(self, audio_data: bytes, dimension_name: str, session_id: str, output_format: str = "wav"):
        """Save audio response as proper WAV or MP3 file with headers."""
        base_filename = f"{session_id}_{dimension_name.replace(' ', '_').lower()}"
        
        # Always create WAV first (proper format)
        wav_filepath = self.output_dir / f"{base_filename}.wav"
        
        # OpenAI Realtime API returns PCM16 at 24kHz
        sample_rate = 24000
        channels = 1
        sample_width = 2  # 16-bit = 2 bytes
        
        try:
            # Create proper WAV file with headers
            with wave.open(str(wav_filepath), 'wb') as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(sample_width)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data)
            
            print(f"Saved audio response: {wav_filepath}")
            
            # If MP3 requested, try to convert using ffmpeg if available
            if output_format.lower() == "mp3":
                mp3_filepath = self.output_dir / f"{base_filename}.mp3"
                try:
                    import subprocess
                    # Try to convert to MP3 using ffmpeg
                    result = subprocess.run([
                        'ffmpeg', '-i', str(wav_filepath), '-acodec', 'mp3', 
                        '-ab', '128k', str(mp3_filepath), '-y'
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        print(f"Also saved as MP3: {mp3_filepath}")
                        return str(mp3_filepath)
                    else:
                        print(f"MP3 conversion failed: {result.stderr}")
                        
                except (ImportError, FileNotFoundError):
                    print("ffmpeg not found - install it for MP3 conversion: brew install ffmpeg")
            
            return str(wav_filepath)
            
        except Exception as e:
            print(f"Error saving WAV file: {e}")
            # Fallback: save as raw data with .raw extension
            raw_filepath = wav_filepath.with_suffix('.raw')
            with open(raw_filepath, 'wb') as f:
                f.write(audio_data)
            print(f"Saved as raw audio: {raw_filepath}")
            return str(raw_filepath)
    
    def create_profiling_prompt(self, dimension: dict) -> str:
        """Create a prompt for analyzing a specific personality dimension."""
        rubric_text = "\n".join([f"Score {k}: {v}" for k, v in dimension["rubric"].items()])
        
        return f"""
        You are a personality assessment expert. Based on the audio you just heard, evaluate the speaker on the dimension "{dimension['name']}".
        
        Description: {dimension['description']}
        
        Rating Scale:
        {rubric_text}
        
        Please provide a detailed analysis of the speaker's {dimension['name'].lower()} based on their speech patterns, word choice, tone, and communication style. 
        Give specific examples from what you heard and conclude with a score from 1-5 and brief justification.
        
        Speak naturally as if you're explaining your assessment to a colleague.
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
                "instructions": "You are a personality assessment expert who analyzes speakers objectively."
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
        
    async def handle_response(self, dimension_name: str, session_id: str):
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
                        audio_path = self.save_audio_response(complete_audio, dimension_name, session_id, self.output_format)
                        print(f"Text analysis: {text_response}")
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
    
    async def analyze_single_dimension(self, audio_data: str, dimension: dict, session_id: str):
        """Analyze a single personality dimension with its own WebSocket connection."""
        print(f"\nAnalyzing: {dimension['name']}")
        
        # Create new connection for this dimension
        await self.connect_websocket()
        await self.send_session_config()
        
        try:
            # Create prompt for this dimension
            prompt = self.create_profiling_prompt(dimension)
            
            # Send audio and prompt
            await self.send_audio_and_prompt(audio_data, prompt)
            
            # Get response
            audio_path, text_analysis = await self.handle_response(
                dimension['name'], session_id
            )
            
            return {
                "dimension": dimension['name'],
                "audio_response": audio_path,
                "text_analysis": text_analysis
            }
            
        finally:
            if self.websocket:
                await self.websocket.close()
                self.websocket = None
    
    async def profile_speaker(self, audio_file_path: str):
        """Main function to profile speaker across all dimensions."""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Starting speaker profiling session: {session_id}")
        
        # Load input audio
        audio_data = self.load_audio_file(audio_file_path)
        
        results = []
        
        # Process each dimension (excluding Fairness as specified)
        for dimension in EVAL_DIMENSIONS:
            try:
                result = await self.analyze_single_dimension(audio_data, dimension, session_id)
                if result:
                    results.append(result)
                
                # Small delay between dimensions
                await asyncio.sleep(2)
                
            except Exception as e:
                print(f"Error analyzing {dimension['name']}: {e}")
                results.append({
                    "dimension": dimension['name'],
                    "audio_response": None,
                    "text_analysis": f"Error: {str(e)}"
                })
        
        # Save summary
        summary_path = self.output_dir / f"{session_id}_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                "session_id": session_id,
                "input_file": audio_file_path,
                "timestamp": datetime.now().isoformat(),
                "results": results
            }, f, indent=2)
        
        print(f"\nProfiler complete! Results saved to: {self.output_dir}")
        print(f"Summary: {summary_path}")
        return results

def main():
    parser = argparse.ArgumentParser(description="Profile speaker personality from audio")
    parser.add_argument("audio_file", help="Path to input WAV file")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--format", choices=["wav", "mp3"], default="wav", 
                       help="Output audio format (default: wav)")
    
    args = parser.parse_args()
    
    # Get API key from .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # If dotenv is not installed, skip loading .env

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key required. Use --api-key or set OPENAI_API_KEY environment variable.")
        return
    
    # Verify input file exists
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}")
        return
    
    # Run profiler
    profiler = SpeakerProfiler(api_key, args.format)
    asyncio.run(profiler.profile_speaker(args.audio_file))

if __name__ == "__main__":
    main()


