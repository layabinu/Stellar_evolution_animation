#!/bin/bash

# Function to display usage/help
usage() {
    echo "Usage: $0 <log|linear> <tulips|default>"
    exit 1
}

# 1. Check if exactly 2 arguments were provided
if [ "$#" -ne 2 ]; then
    echo "Error: Incorrect number of arguments."
    usage
fi

SCALING=$1
IMAGES=$2

# 2. Validate the 'scaling' argument
case "$SCALING" in
    log|linear)
        # Valid choice
        ;;
    *)
        echo "Error: Invalid scaling value '$SCALING'. Must be 'log' or 'linear'."
        usage
        ;;
esac

# 3. Validate the 'images' argument
case "$IMAGES" in
    tulips|default)
        # Valid choice
        ;;
    *)
        echo "Error: Invalid images value '$IMAGES'. Must be 'tulips' or 'default'."
        usage
        ;;
esac

FRAMES="/Users/ronin/animation/Final Animation/frames_data.npz"
PREPROCESS="/Users/ronin/animation/Final Animation/preprocess.py"
ANIMATE="/Users/ronin/animation/Final Animation/animate.py"
COLOR="/Users/ronin/animation/Final Animation/temp_to_color.py"

# Run preprocess only if frames_data.npz does not exist
if [ ! -f "$FRAMES" ]; then
    echo "frames_data.npz not found. Running preprocess.py..."
    python3 "$PREPROCESS"
    python3 "$COLOR"
else
    echo "frames_data.npz found. Skipping preprocess."
fi

# Run animation
echo "Running animate.py: python3 "$ANIMATE" --scaling ${SCALING} --images ${IMAGES}"
python3 "$ANIMATE" --scaling "${SCALING}" --images "${IMAGES}"

