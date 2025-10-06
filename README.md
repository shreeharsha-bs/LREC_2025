# Speaker Evaluation System

This system provides two evaluation modes for analyzing speech audio:

1. **Personality Profiling Mode**: Evaluates Big Five personality dimensions (OpenAI → Local ASR → Gemini)
2. **Long-form Task Mode**: Evaluates responses across specific task scenarios (OpenAI response → Gemini evaluation)

## Architecture

### Personality Mode
- **OpenAI Realtime**: Generates personality analysis audio
- **Local Whisper**: Transcribes the analysis 
- **Gemini**: Extracts structured personality scores

### Long-form Mode  
- **OpenAI Realtime**: Generates task-specific responses to speech input
- **Gemini**: Evaluates the responses on multiple dimensions

## Setup

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Set environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key"
export GOOGLE_API_KEY="your-google-api-key"  # Only needed for personality mode
```

## Usage

### Mode Selection

Use the main workflow script with different modes:

```bash
# Test both modes
python test_both_modes.py

# Personality Profiling Mode
python test_complete_workflow.py audio_file.wav --mode personality

# Long-form Evaluation Mode (all tasks) - requires both API keys
python test_complete_workflow.py audio_file.wav --mode longform

# Long-form Evaluation Mode (specific task)
python test_complete_workflow.py audio_file.wav --mode longform --task therapy

# With explicit API keys
python test_complete_workflow.py audio_file.wav --mode longform --openai-key YOUR_KEY --google-key YOUR_KEY
```

### Available Long-form Tasks

- `therapy`: Evaluates advice-giving for emotional distress
- `career`: Evaluates career guidance and job recommendations  
- `interview`: Evaluates hiring decisions and salary recommendations
- `story`: Evaluates narrative creation and character agency
- `all`: Runs all tasks sequentially

### Individual Mode Scripts

You can also run each mode separately:

```bash
# Personality profiling only
python speaker_profiling_single.py audio_file.wav

# Long-form evaluation only
python longform_evaluator.py audio_file.wav --task therapy
python longform_evaluator.py audio_file.wav --task all
```

## Output

### Personality Mode
- Audio responses saved to `speaker_profiles/`
- JSON summaries with personality scores (1-5 scale)
- Dimensions: Openness, Conscientiousness, Extraversion, Agreeableness, Emotional Stability, Fairness

### Long-form Mode
- Audio responses saved to `longform_evaluations/`
- JSON results with task-specific scores (1-5 scale)
- Each task has 3 evaluation dimensions with detailed rubrics

## File Structure

- `speaker_profiling_single.py` - Personality profiling system
- `longform_evaluator.py` - Long-form task evaluation system  
- `test_complete_workflow.py` - Main workflow with mode selection
- `personality_score_extractor.py` - Local ASR + score extraction
- `test_both_modes.py` - Comprehensive test script
- `requirements.txt` - Python dependencies

## Testing

Run the comprehensive test to verify both modes work:

```bash
python test_both_modes.py
```

This will test both evaluation modes and provide usage examples.