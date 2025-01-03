#!/bin/bash

# File containing the YouTube video IDs
INPUT_FILE="ava_video_list_v1.0.txt"

# Directory to save downloaded audio files
OUTPUT_DIR="files"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Read each line from the input file
while IFS= read -r video_id || [[ -n "$video_id" ]]; do
    # Construct the full YouTube URL
    url="https://www.youtube.com/watch?v=$video_id"

    echo "Processing $url..."

    # Download audio as WAV format and include the video ID in the filename
    yt-dlp -f "bestaudio" --extract-audio --audio-format wav \
        -o "$OUTPUT_DIR/%(title)s_${video_id}.%(ext)s" "$url"

    # Check the exit code of yt-dlp to handle errors
    if [[ $? -ne 0 ]]; then
        echo "Failed to download $url. Skipping..."
    else
        echo "Downloaded $url successfully."
    fi

done < "$INPUT_FILE"

echo "All downloads completed."