# SemanticSegmentationSegNet
semantic segmentation of magnetic resonance (MR) images using deep learning. In digital image processing and computer vision, semantic image segmentation is the process of assigning a label to every pixel in an image such that pixels with the same label share certain characteristics


The aim of this work is to create a neural network that segments cardiac MR images. The dataset is comprised of 200 96x96 greyscale cardiac MR images (CMRI) along the short axis. The goal is to segment them into four different categories: myocardium, left ventricle, right ventricle, and background. The data has been pulled from the Automated Caridiac Diagnosis Challenge (ACDC).

his is the ideal problem for a convolutional neural network - indeed as technology and deep learning have developed, the area of medical imagery has benefited greatly. Applying a CNN to pixel-wise classification allows for a much faster and less error-prone method of segmentation.

The dataset has been split into the following three subsets: 50% for training, 10% for validation, and the remaining 40% for testing. The goal of this work is to implement a known segmentation architecture to this dataset then perform hyper-parameter optimisation to maximise the dice score on the test set. The dice score between two generic masks A, B is defined as: [dice(A, B) = \frac{2|A \cap B|}{|A|+|B|}] The dice score ranges between 0 and 1 with 0 being completely wrong and 1 being a perfect match. Since there are 4 classes to be segmented then even if the theoretical minimum is 0, by just classiying each pixel at random will achieve a dice score of 0.25 (that is if each class occurs with equal frequency).

e opted to use the SegNet architecture (Badrinarayanan, Kendall and Cipolla, 2015) which is an encoder-decoder model for image segmentation. The idea is that the model applies a sequence of increasingly large number of convolutional filters and max-pooling layers to map the image to a low dimensional space (but with a large number of channels) and then to reverse the process by up-sampling and convolutional layers to return the image back to the same dimensionality of the input. Below is the architecture of SegNet taken from their paper.

![image](https://user-images.githubusercontent.com/12696541/158090653-7b4540eb-25c6-4a10-ba6b-646ec90e3edd.png)

Training History

![image](https://user-images.githubusercontent.com/12696541/158090913-328ef37b-7f60-425d-a729-8414632568e0.png)

Classification Accuracy

![image](https://user-images.githubusercontent.com/12696541/158090954-d9ad18a4-6285-4486-8358-d8118ef459c1.png)


Best Performance:

  Epoch 99/100
  
  Train loss:  0.7521338015794754
  
  Validation loss:  0.7804208397865295
  
  Dice score:  0.877760516852967
