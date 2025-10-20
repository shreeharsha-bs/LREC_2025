#!/bin/bash

# Define the base directory for VCTK audio files
VCTK_BASE_DIR="/shared/lameris/vctk_16bit"

# Define voice quality folders
VOICE_QUALITIES=("modal_converted" "modified_breathy" "modified_creaky" "modified_end_creak")

# Define male and female speakers (10 each)
MALE_SPEAKERS=("p226" "p227" "p232" "p237" "p241" "p243" "p245" "p246" "p247" "p251")
FEMALE_SPEAKERS=("p225" "p228" "p229" "p230" "p231" "p233" "p234" "p236" "p238" "p239")

# Combine all speakers
ALL_SPEAKERS=("${MALE_SPEAKERS[@]}" "${FEMALE_SPEAKERS[@]}")

# Tasks to process
TASKS=("therapy" "career_advice" "interview_screening" "story")

# Task mapping for pipeline
declare -A TASK_MAP
TASK_MAP["therapy"]="therapy"
TASK_MAP["career_advice"]="career"
TASK_MAP["interview_screening"]="interview"
TASK_MAP["story"]="story"

echo "========================================================================"
echo "VCTK LONGFORM EVALUATION (Steps 4, 5, 6 - All Voice Qualities)"
echo "========================================================================"

# Counters
TOTAL_FILES=0
SUCCESS=0
FAILED=0
SKIPPED=0

# Process each voice quality folder
for voice_quality in "${VOICE_QUALITIES[@]}"; do
    echo ""
    echo "========================================================================"
    echo "Processing voice quality: $voice_quality"
    echo "========================================================================"
    
    VCTK_DIR="$VCTK_BASE_DIR/$voice_quality"
    
    if [ ! -d "$VCTK_DIR" ]; then
        echo "✗ Directory not found: $VCTK_DIR"
        continue
    fi
    
    # Process each speaker
    for speaker in "${ALL_SPEAKERS[@]}"; do
        echo ""
        echo "Processing speaker: $speaker ($voice_quality)"
        
        SPEAKER_DIR="$VCTK_DIR/$speaker"
        
        if [ ! -d "$SPEAKER_DIR" ]; then
            echo "✗ Directory not found for $speaker"
            continue
        fi
        
        # Process each task
        for task_dir in "${TASKS[@]}"; do
            TASK_PATH="$SPEAKER_DIR/$task_dir"
            
            if [ ! -d "$TASK_PATH" ]; then
                continue
            fi
            
            PIPELINE_TASK="${TASK_MAP[$task_dir]}"
            
            # Process each audio file
            for input_audio in "$TASK_PATH"/*.wav; do
                if [ ! -f "$input_audio" ]; then
                    continue
                fi
                
                TOTAL_FILES=$((TOTAL_FILES + 1))
                BASE_NAME=$(basename "$input_audio" .wav)
                
                # Generate filenames with voice quality suffix
                GENERATED_AUDIO="longform_audio/${BASE_NAME}_${PIPELINE_TASK}_${voice_quality}.wav"
                TRANSCRIPT_FILE="longform_transcripts/${BASE_NAME}_${PIPELINE_TASK}_${voice_quality}_transcript.txt"
                SCORES_FILE="longform_results/${BASE_NAME}_${PIPELINE_TASK}_${voice_quality}_scores.json"
                
                echo "  Processing: $BASE_NAME"
                
                # Step 4: Generate
                python 4_longform_generate.py "$input_audio" "$PIPELINE_TASK" "$GENERATED_AUDIO"
                
                if [ $? -ne 0 ]; then
                    echo "  ✗ Generation failed"
                    FAILED=$((FAILED + 1))
                    continue
                fi
                
                # Step 5: Transcribe
                python 5_longform_transcribe.py "$GENERATED_AUDIO" "$TRANSCRIPT_FILE"
                
                if [ $? -ne 0 ]; then
                    echo "  ✗ Transcription failed"
                    FAILED=$((FAILED + 1))
                    continue
                fi
                
                # Step 6: Evaluate
                AUDIO_NAME=$(basename "$GENERATED_AUDIO")
                SCORES_NAME=$(basename "$SCORES_FILE")
                python 6_longform_evaluate.py "$TRANSCRIPT_FILE" "$PIPELINE_TASK" "$AUDIO_NAME" "$SCORES_NAME"
                
                if [ $? -eq 0 ]; then
                    echo "  ✓ Success"
                    SUCCESS=$((SUCCESS + 1))
                else
                    echo "  ✗ Evaluation failed"
                    FAILED=$((FAILED + 1))
                fi
            done
        done
    done
done

echo ""
echo "========================================================================"
echo "Total: $TOTAL_FILES | Success: $SUCCESS | Failed: $FAILED | Skipped: $SKIPPED"
echo "========================================================================"