#!/bin/bash

# Whether or not to use Snellius (project space)
SNELLIUS=0
if [ $SNELLIUS -eq 1 ]; then
  # Load the Snellius module
  export output_path="/projects/0/prjs1537/projects/data-efficient-analysis"
  cd $output_path
fi


# Parameter indicating which dataset to download
PARAM=$1

# Define the destination directory and download links based on the parameter
case "$PARAM" in
  tinystories)
    DEST_DIR="./data/tinystories"
    declare -A FILE_URLS=(
      ["train/TinyStoriesV2-GPT4.txt"]="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
      ["dev/TinyStoriesV2-GPT4.txt"]="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
    )
    ;;
  babylm)
    DEST_DIR="./data/babylm"
    declare -A FILE_URLS=(
      ["10M_train.zip"]="https://osf.io/download/5mk3x/"   # 10M train
      ["100M_train.zip"]="https://osf.io/download/rduj2/"  # 100M train
      ["dev.zip"]="https://osf.io/download/m48ed/"         # dev
    )
    ;;
    babylm3)
    DEST_DIR="./data/babylm3"
    declare -A FILE_URLS=(
      ["100M_train.zip"]="https://osf.io/download/6819fdbfbecda878d4c61566/"   # 10M train
      ["10M_train.zip"]="https://osf.io/download/6819fd8d91a6c4b6d4159848/"  # 100M train
      ["dev.zip"]="https://osf.io/download/6819fd86ff83c3467ac0dcb0/"         # dev
    )
    ;;
  *)
    echo "Error: Not implemented data support for '$PARAM'"
    exit 1
    ;;
esac

# Create destination directory and necessary subdirectories
mkdir -p "$DEST_DIR"

# For tinystories, also make train/ and dev/ subdirectories
if [ "$PARAM" == "tinystories" ]; then
  mkdir -p "$DEST_DIR/train"
  mkdir -p "$DEST_DIR/dev"
fi

# Download all of the dataset files to the destination dir
for FILENAME in "${!FILE_URLS[@]}"; do
  URL="${FILE_URLS[$FILENAME]}"
  wget -O "$DEST_DIR/$FILENAME" "$URL"
done


# Unzip the files if the parameter is babylm or babylm3
if [[ "$PARAM" == "babylm" || "$PARAM" == "babylm3" ]]; then
  for ZIP_FILE in "$DEST_DIR"/*.zip; do
    if [ -f "$ZIP_FILE" ]; then
      echo "Unzipping $ZIP_FILE..."
      unzip "$ZIP_FILE" -d "$DEST_DIR"
      echo "Done."
      rm "$ZIP_FILE"
    fi
  done
fi