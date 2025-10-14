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
RESULTS_DIR="results"

# --- Script Setup ---
# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Define the resolutions to test
# RESOLUTIONS=("test" "1080p" "2k" "4k")
RESOLUTIONS=("test")
CSV_FILE="${RESULTS_DIR}/all_fps_${VERSION}.csv"

# Initialize the results file with a new header, including resolution
echo "Scene,Resolution,Average FPS" > "$CSV_FILE"

# --- Scene Definitions ---
scenes_4=("bicycle" "flowers" "garden" "stump" "treehill")
scenes_2=("counter" "room" "kitchen" "bonsai")
scenes_1=("truck" "train" "drjohnson" "playroom")
scenes_4=("bicycle" "garden")
scenes_2=("bonsai")
scenes_1=("truck" "train" "drjohnson" "playroom")


# --- Function to run benchmark for a scene ---
# Reduces code duplication
run_benchmark() {
    local scene=$1
    local downsample_factor=$2
    local use_transform=$3
    local log_file="${RESULTS_DIR}/${scene}.txt"
    local colmap_path="/data/nerf_datasets/360/${scene}/sparse/0/"
    # Set the colmap path based on the scene name
    case "$scene" in
        "truck" | "train")
            colmap_path="/data/nerf_datasets/tandt/${scene}/sparse/0/"
            ;;
        "playroom" | "drjohnson")
            colmap_path="/data/nerf_datasets/db/${scene}/sparse/0/"
            ;;
        *)
            # Default path for other scenes
            colmap_path="/data/nerf_datasets/360/${scene}/sparse/0/"
            ;;
    esac

    echo "--- Processing Scene: $scene ---"

    # Download assets ONCE per scene
    echo "Downloading assets for $scene..."
    if [ "$downsample_factor" = 1 ]; then
        scp "${REMOTE_HOST}:${REMOTE_PATH}/${scene}_ifimages_${VERSION}/ckpt.ply" .
        if [ "$use_transform" = true ]; then
            scp "${REMOTE_HOST}:${REMOTE_PATH}/${scene}_ifimages_${VERSION}/transform.txt" .
        fi
    else
        scp "${REMOTE_HOST}:${REMOTE_PATH}/${scene}_ifimages_${downsample_factor}_${VERSION}/ckpt.ply" .
        if [ "$use_transform" = true ]; then
            scp "${REMOTE_HOST}:${REMOTE_PATH}/${scene}_ifimages_${downsample_factor}_${VERSION}/transform.txt" .
        fi
    fi

    # Loop through each resolution
    for res in "${RESOLUTIONS[@]}"; do
        echo "--> Running benchmark for ${scene} at ${res}"

        # Build the command
        cmd="./build/bin/benchmark \
            --scene ckpt.ply \
            --colmap \"${colmap_path}\" \
            --auto \
            -d ${downsample_factor} \
            -r ${res}" # Use the new resolution flag

        # Add transform file if needed
        if [ "$use_transform" = true ]; then
            cmd+=" --transform_file transform.txt"
        fi

        # Run benchmark and redirect output to a log file (overwrite for each resolution)
        # The '|| true' prevents the script from exiting if the benchmark fails
        eval "$cmd" > "$log_file" || true

        # Capture FPS from the log file
        fps=$(grep "Average FPS:" "$log_file" | awk '{print $NF}')
        if [ -z "$fps" ]; then
            fps="N/A" # Handle cases where FPS is not found
        fi

        # Append the result to the main CSV file
        echo "${scene},${res},${fps}" >> "$CSV_FILE"
        echo "Result for ${scene} at ${res}: ${fps} FPS"
    done

    # Clean up assets after all resolutions for this scene are done
    rm -f ckpt.ply transform.txt
    echo "--- Finished: $scene ---"
    echo "" # Add a blank line for readability
}

# --- Main Execution ---

for scene in "${scenes_4[@]}"; do
    run_benchmark "$scene" 4 true
done

for scene in "${scenes_2[@]}"; do
    run_benchmark "$scene" 2 true
done

# for scene in "${scenes_1[@]}"; do
#     run_benchmark "$scene" 1 true
# done

echo "--- Script finished. All results are in ${CSV_FILE} ---"
