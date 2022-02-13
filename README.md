# Scanbage

## What it does
- Classifies the waste material in the image into 6 categories :
  1. Cardboard
  2. Glass
  3. Metal
  4. Paper
  5. Plastic
  6. Trash
- The results of tells if the item should be recycled, composted, put into the general trash or needs special treatment (hazardous materials)
- Tells the information on how much CO2 emission they avoided

## How to use
- Pre requisite: You should already have flask, tensflow and torchvision libraries installed
- Steps:
  1. Download the github zip file.
  2. Extract the file.
  3. Go to your terminal and change the directory to where you extracted the zip file.
  4. Then type in, python app.py.
  5. Follow the link provided by the terminal.    
  6. Upload the desired image (preferably from the dataset-resized.zip)

## How was it built
- Pre-Trained image detection model **DenseNet-121**
- DenseNet-121 is pre-trained on **ImageNet** to distinguish 1000 classes of objects
- Dataset is trained using **VGG16 Transfer Learning Technique** of CNN for the classification
