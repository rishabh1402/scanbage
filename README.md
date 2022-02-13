# Scanbage

# Scanbage
## What it does
- A Web App to sort your waste via scanning the uploaded images
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
* Click **Login** to start with a free account

![homePageLogin](/Screenshots/homePageLogin.png)
![loginPage](/Screenshots/loginPage.png)

* Click **Explore** to start

![homePageExplore](/Screenshots/homePageExplore.PNG)

* **Choose** an image to be uploaded

![uploadImage](/Screenshots/uploadImage.PNG)

* **Upload** the file

![uploadedPic](/Screenshots/uploadedPic.PNG)

* Get the result within seconds with atmost accuracy

![resultPage](/Screenshots/resultPage.jpeg)

## How was it built
- Pre-Trained image detection model **DenseNet-121**
- DenseNet-121 is pre-trained on **ImageNet** to distinguish 1000 classes of objects
- Dataset is trained using **VGG16 Transfer Learning Technique** of CNN for the classification

## Try the Web App :
  https://scanbage.herokuapp.com/
  
## Model :
  https://github.com/rishabh1402/scanbage
