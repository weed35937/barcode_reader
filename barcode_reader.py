import cv2
import os
from pyzbar.pyzbar import decode

IMAGE_FOLDER = "barcodes"   
OUTPUT_FILE = "output.txt" 

CROP_X = 313    
CROP_Y = 435   
CROP_WIDTH = 642  
CROP_HEIGHT = 615 

def read_barcode(image_path):
    """Reads and extracts Barcode39 from a cropped image."""
    image = cv2.imread(image_path)

    if image is None:
        return "ERROR_READING_IMAGE"

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cropped = gray[CROP_Y:CROP_Y + CROP_HEIGHT, CROP_X:CROP_X + CROP_WIDTH]

    _, cropped = cv2.threshold(cropped, 127, 255, cv2.THRESH_BINARY)

    decoded_objects = decode(cropped)
    if decoded_objects:
        return decoded_objects[0].data.decode("utf-8")  
    else:
        return "NO_BARCODE"

def process_images(folder):
    """Processes all images in the folder and writes results to a text file."""
    results = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(".jpg"):
            image_path = os.path.join(folder, filename)
            barcode = read_barcode(image_path)
            results.append(f"{barcode}, {filename}")

    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(results))
    
    print(f"✅ Processed {len(results)} images. Results saved in '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    process_images(IMAGE_FOLDER)
