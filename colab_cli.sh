#!/bin/bash

############
# Automatically run using google colab cli
############

SESS="kplus"
FILE="/tmp/kplus.tar.gz"
REMOTE_DIR="/content/KaraokePlus"

ARGS=""
INPUT_FILE=""
FILENAME=""
STOP="False"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--input)
      INPUT_FILE="$2"
      FILENAME=$(basename "$INPUT_FILE")
      
      # Rewrite the argument to use the remote Colab path instead of the local one
      ARGS="$ARGS $1 $REMOTE_DIR/$FILENAME"
      
      shift 2
      ;;
    --stop)
      STOP="True"
      shift
      ;;
    *)
      # If it's any other argument, keep it exactly as it is
      ARGS="$ARGS $1"
      shift
      ;;
  esac
done

echo "Packaging local directory..."
tar --exclude="$FILE" -czf "$FILE" .

echo "Starting Colab session ($SESS)..."
colab new -s "$SESS" --gpu t4

echo "Uploading project to Colab..."
colab upload "$FILE" "/content/kplus.tar.gz" -s "$SESS"
echo "Creating remote working directory..."
echo "mkdir -p $REMOTE_DIR" | colab exec -s "$SESS"

if [ -n "$INPUT_FILE" ]; then
    if [ -f "$INPUT_FILE" ]; then
        echo "Intercepted input file! Uploading: $INPUT_FILE..."
        colab upload "$INPUT_FILE" "$REMOTE_DIR/$FILENAME" -s "$SESS"
    else
        echo "!!! Error: Input file '$INPUT_FILE' not found locally!"
        rm "$FILE"
        colab stop -s "$SESS"
        exit 1
    fi
fi

echo "⚙️ Extracting, installing, and running script..."
echo "!tar -xzf /content/kplus.tar.gz -C $REMOTE_DIR && \
    cd /content/KaraokePlus && \
    python kplus-bin $ARGS \
    " | colab exec -s "$SESS" --timeout 120.0

echo "Cleaning up local files..."
rm "$FILE"

if [ "$STOP" == "True" ]; then
  echo "Stopping Colab session ($SESS)..."
  colab stop -s "$SESS"
else
  colab repl -s "$SESS"
fi

echo "Stopping Colab session ($SESS)..."
colab stop -s "$SESS"

echo "Done!"