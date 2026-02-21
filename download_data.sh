DEST_DIR=./data
mkdir -p $DEST_DIR
URL=https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip?dl=0
ZIP_FILE=./data/afhq.zip
mkdir -p ./data
wget -N $URL -O $ZIP_FILE
unzip $ZIP_FILE -d $DEST_DIR
rm $ZIP_FILE
mv ./data/afhq ./data/afhq_cat
mv ./data/afhq_cat/val ./data/afhq_cat/test
bash scripts/afhq_validation_images.sh

DEST_DIR=./data/celeba
ZIP_FILE="$DEST_DIR/celeba-dataset.zip"
mkdir -p $DEST_DIR
echo "Downloading CelebA dataset..."
kaggle datasets download jessicali9530/celeba-dataset -p "$DEST_DIR"
# Ensure the ZIP file exists before extracting
if [ -f "$ZIP_FILE" ]; then
    echo "Dataset downloaded. Extracting..."
    unzip -q "$ZIP_FILE" -d "$DEST_DIR"
    rm "$ZIP_FILE"
    echo "Extraction completed!"
else
    echo "Error: ZIP file not found after download!"
    exit 1
mv ./data/celeba/img_align_celeba/img_align_celeba/* ./data/celeba/img_align_celeba
fi
