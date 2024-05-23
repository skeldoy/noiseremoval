#!/bin/bash

# Source and target directories
SOURCE_DIR="../data-smaller"
TARGET_DIR="../data"

# Find all png files in the source directory
find "$SOURCE_DIR" -type f -name '*.png' | while read file; do
    # Get the dimensions of the image
    dimensions=$(identify -format "%wx%h" "$file")
    
    # Check if dimensions are 640x480
    if [ "$dimensions" = "640x480" ]; then
        # Get the filename
        filename=$(basename "$file")
        # Set the target file path
        target_file="$TARGET_DIR/$filename"
        # Upscale the image to 1024x576 and save to the target directory
        convert "$file" -resize 1024x576 "$target_file"
    fi
done

