#!/bin/bash

# Source and target directories
SOURCE_DIR="../data"
TARGET_DIR="../data-smaller"

# Find all png files in the source directory
find "$SOURCE_DIR" -type f -name '*.png' | while read file; do
    # Get the dimensions of the image
    dimensions=$(identify -format "%wx%h" "$file")
    
    # Check if dimensions are 640x480
    if [ "$dimensions" = "640x480" ]; then
        # Move the file to the target directory
        mv "$file" "$TARGET_DIR"
    fi
done

