#!/bin/bash
#
# Upload datasets to GCS for TPU v4-64 training
#
# This script:
# 1. Checks if gsutil is available
# 2. Creates GCS bucket if needed
# 3. Generates required datasets if they don't exist locally
# 4. Uploads datasets to GCS
#

set -euo pipefail

# Configuration
GCS_BUCKET="${GCS_BUCKET:-gs://sculptor-tpu-experiments}"
GCS_DATA_PATH="${GCS_BUCKET}/data"
REGION="${REGION:-us-central2}"
LOCAL_DATA_DIR="data"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "================================================================"
echo "GCS Dataset Upload for TPU v4-64"
echo "================================================================"
echo "GCS Bucket:     ${GCS_BUCKET}"
echo "Data Path:      ${GCS_DATA_PATH}"
echo "Local Data:     ${LOCAL_DATA_DIR}"
echo "Region:         ${REGION}"
echo "================================================================"
echo ""

# Check gsutil
if ! command -v gsutil &> /dev/null; then
    echo -e "${RED}✗ Error: gsutil not found${NC}"
    echo "  Install Google Cloud SDK: https://cloud.google.com/sdk/docs/install"
    exit 1
fi
echo -e "${GREEN}✓ gsutil found${NC}"

# Check GCS authentication
if ! gsutil ls &> /dev/null; then
    echo -e "${RED}✗ Error: Not authenticated with GCS${NC}"
    echo "  Run: gcloud auth login"
    exit 1
fi
echo -e "${GREEN}✓ GCS authentication OK${NC}"

# Create bucket if it doesn't exist
echo ""
echo "Checking GCS bucket..."
if ! gsutil ls "${GCS_BUCKET}/" &> /dev/null; then
    echo -e "${YELLOW}Creating GCS bucket: ${GCS_BUCKET}${NC}"
    if gsutil mb -l "${REGION}" "${GCS_BUCKET}/"; then
        echo -e "${GREEN}✓ Bucket created${NC}"
    else
        echo -e "${RED}✗ Failed to create bucket${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ Bucket exists${NC}"
fi

# Define all datasets to upload
# Format: "local_name:subsample_size:num_aug"
DATASETS=(
    "sudoku-extreme-1k-aug-1000:1000:1000"
    "sudoku-extreme-100-aug-1000:100:1000"
    "sudoku-extreme-250-aug-1000:250:1000"
    "sudoku-extreme-500-aug-1000:500:1000"
    "sudoku-extreme-1000-aug-1000:1000:1000"
    "sudoku-extreme-2000-aug-1000:2000:1000"
    "sudoku-extreme-5000-aug-1000:5000:1000"
    "sudoku-extreme-1k-aug-10:1000:10"
    "sudoku-extreme-1k-aug-100:1000:100"
    "sudoku-extreme-1k-aug-500:1000:500"
    "sudoku-extreme-1k-aug-2000:1000:2000"
)

echo ""
echo "================================================================"
echo "Processing ${#DATASETS[@]} datasets"
echo "================================================================"

uploaded=0
skipped=0
generated=0
failed=0

for dataset_spec in "${DATASETS[@]}"; do
    # Parse dataset specification
    IFS=':' read -r dataset_name subsample_size num_aug <<< "${dataset_spec}"

    local_path="${LOCAL_DATA_DIR}/${dataset_name}"
    gcs_path="${GCS_DATA_PATH}/${dataset_name}"

    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Dataset: ${dataset_name}${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo "  Subsample:  ${subsample_size}"
    echo "  Augment:    ${num_aug}"
    echo "  Local:      ${local_path}"
    echo "  GCS:        ${gcs_path}"

    # Check if already exists in GCS
    if gsutil -q stat "${gcs_path}/train/dataset.json" 2>/dev/null; then
        echo -e "${GREEN}✓ Already exists in GCS, skipping${NC}"
        ((skipped++))
        continue
    fi

    # Check if exists locally
    if [[ ! -d "${local_path}" ]] || [[ ! -f "${local_path}/train/dataset.json" ]]; then
        echo -e "${YELLOW}⚠ Not found locally, generating...${NC}"

        # Generate dataset
        if python dataset/build_sudoku_dataset.py \
            --output-dir "${local_path}" \
            --subsample-size "${subsample_size}" \
            --num-aug "${num_aug}"; then
            echo -e "${GREEN}✓ Generated successfully${NC}"
            ((generated++))
        else
            echo -e "${RED}✗ Generation failed${NC}"
            ((failed++))
            continue
        fi
    else
        echo -e "${GREEN}✓ Found locally${NC}"
    fi

    # Upload to GCS
    echo "Uploading to GCS..."
    if gsutil -m cp -r "${local_path}" "${GCS_DATA_PATH}/"; then
        echo -e "${GREEN}✓ Upload complete${NC}"
        ((uploaded++))

        # Verify upload
        if gsutil -q stat "${gcs_path}/train/dataset.json"; then
            echo -e "${GREEN}✓ Verified in GCS${NC}"
        else
            echo -e "${RED}✗ Verification failed${NC}"
            ((failed++))
        fi
    else
        echo -e "${RED}✗ Upload failed${NC}"
        ((failed++))
    fi
done

# Summary
echo ""
echo "================================================================"
echo "SUMMARY"
echo "================================================================"
echo "Total datasets:      ${#DATASETS[@]}"
echo -e "${GREEN}Uploaded:            ${uploaded}${NC}"
echo -e "${YELLOW}Skipped (in GCS):    ${skipped}${NC}"
echo -e "${BLUE}Generated locally:   ${generated}${NC}"
echo -e "${RED}Failed:              ${failed}${NC}"
echo "================================================================"

# List uploaded datasets
echo ""
echo "Datasets in GCS:"
if gsutil ls "${GCS_DATA_PATH}/" 2>/dev/null; then
    echo -e "${GREEN}✓ GCS data directory accessible${NC}"
else
    echo -e "${RED}✗ Could not list GCS data directory${NC}"
fi

# Final status
echo ""
if [[ ${failed} -gt 0 ]]; then
    echo -e "${RED}✗ Upload completed with ${failed} failures${NC}"
    exit 1
elif [[ ${uploaded} -gt 0 ]] || [[ ${skipped} -gt 0 ]]; then
    echo -e "${GREEN}✓ All datasets ready in GCS!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Update experiment configs:  python kellen/update_data_paths_to_gcs.py"
    echo "  2. Test GCS data loader:       python test_gcs_dataloader.py"
    echo "  3. Launch training:            ./launch_tpu_training.sh baseline"
    exit 0
else
    echo -e "${YELLOW}⚠ No datasets processed${NC}"
    exit 1
fi
