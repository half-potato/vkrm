#!/bin/bash

# Exit immediately if a command fails
set -e

# Check for the required command-line argument
if [ -z "$1" ]; then
    echo "Usage: $0 <version_suffix>"
    exit 1
fi

VERSION=$1
REMOTE_HOST="dronelab@67.58.52.188"
REMOTE_PATH="~/delaunay_rasterization/output"

# Create results directory if it doesn't exist
mkdir -p results

# Initialize the results file with a header
echo "Scene,Average FPS" > results/all_fps.csv

## --- Process scenes that use a transform file ---
for scene in bicycle flowers garden stump treehill; do
    echo "--- Processing: $scene ---"

    # Download assets
    scp "${REMOTE_HOST}:${REMOTE_PATH}/${scene}_ifimages_4_${VERSION}/ckpt.ply" .
    scp "${REMOTE_HOST}:${REMOTE_PATH}/${scene}_ifimages_4_${VERSION}/transform.txt" .

    # Run benchmark and redirect output to a scene-specific log
    ./build/bin/benchmark \
        --scene ckpt.ply \
        --colmap "/data/nerf_datasets/360/${scene}/sparse/0/" \
        --auto \
        --transform_file transform.txt > "results/${scene}.txt"

    # Correctly capture FPS from the correct log file
    fps=$(grep "Average FPS:" "results/${scene}.txt" | awk '{print $NF}')

    # Append the result to the main CSV file
    echo "${scene},${fps}" >> results/all_fps.csv
    echo "Result for ${scene}: ${fps} FPS"

    # Clean up assets for the next loop
    rm -f ckpt.ply transform.txt
done

# --- Process scenes that do not use a transform file ---
for scene in counter room kitchen bonsai; do
    echo "--- Processing: $scene ---"

    # Download assets
    scp "${REMOTE_HOST}:${REMOTE_PATH}/${scene}_ifimages_2_${VERSION}/ckpt.ply" .
    scp "${REMOTE_HOST}:${REMOTE_PATH}/${scene}_ifimages_2_${VERSION}/transform.txt" .

    # Run benchmark
    ./build/bin/benchmark \
        --scene ckpt.ply \
        --colmap "/data/nerf_datasets/360/${scene}/sparse/0/" \
        --auto \
        --transform_file transform.txt > "results/${scene}.txt"

    # Correctly capture FPS
    fps=$(grep "Average FPS:" "results/${scene}.txt" | awk '{print $NF}')

    # Append the result
    echo "${scene},${fps}" >> results/all_fps.csv
    echo "Result for ${scene}: ${fps} FPS"

    # Clean up
    rm -f ckpt.ply
done

echo "--- Script finished. All results are in results/all_fps.csv ---"
