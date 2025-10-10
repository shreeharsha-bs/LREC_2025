# Voice Quality Analysis Pipeline

Pipeline for analyzing how voice quality (Modal, Breathy, Creaky, End Creak) affects AI-generated response quality across therapy, career, interview, and story tasks.

## Quick Start for Harm!

**Automated Pipeline Runner (Recommended):**
```bash
for task in {therapy,career,interview,story}; do
  for file in modified_response_task/*/*"$task"*; do
    python run_full_pipeline.py longform "$file" --task "$task"
  done
done
```

```bash
# Longform evaluation
python run_full_pipeline.py longform INPUT.wav --task therapy

# Speaker profiling
python run_full_pipeline.py profiling INPUT.wav
```

The `run_full_pipeline.py` script automatically handles all steps and file naming.

## Pipeline Overview

**Longform Evaluation (3 steps + visualization):**

1. **Generate** (`4_longform_generate.py`) - Create task responses using OpenAI Realtime API
2. **Transcribe** (`5_longform_transcribe.py`) - Convert audio to text using Whisper ASR
3. **Evaluate** (`6_longform_evaluate.py`) - Score responses using Gemini LLM
4. **Visualize** (`plot_heatmap_with_ci.py`) - Generate statistical heatmap (after all samples processed)

## Installation

**Required Libraries:**

```bash
pip install numpy scipy matplotlib seaborn transformers torch websockets python-dotenv google-generativeai
```

## Usage 2

You can also run things individually.

Also, after the results are stored:

```bash
python plot_heatmap_with_ci.py
```
This will plot all the metrics. like the PDF example.



**API Keys Setup:**

Create `.env` file in project root:

```bash
OPENAI_API_KEY=your_openai_key_here
GEMINI_API_KEY=your_gemini_key_here
```

## Directory Structure

```text
project/
├── run_full_pipeline.py         # Automated pipeline runner
├── 4_longform_generate.py
├── 5_longform_transcribe.py
├── 6_longform_evaluate.py
├── plot_heatmap_with_ci.py
├── .env
├── longform_audio/              # Step 1 output
├── longform_transcripts/        # Step 2 output
├── longform_results/            # Step 3 output
├── longform_heatmap_with_ci.png # Visualization output (PNG)
└── longform_heatmap_with_ci.pdf # Visualization output (PDF)
```
