# LICENSE PLATE RECOGNITION USING DEEP LEARNING METHODS

1. Convolutional Neural Networks (CNNs) are inspired by the biological visual cortex, which utilizes local receptive fields to detect patterns in visual inputs. They have been extensively used in computer vision tasks such as object detection and recognition, optical character recognition, and face detection.
       
2. CNNs utilize raw pixels as input, reducing the need for extensive preprocessing and simplifying the machine learning pipeline. However, this results in greater computational intensity, especially for larger datasets.
       
3. Key features of CNNs include local receptive fields, shared weights, and pooling. These elements distinguish CNNs from standard feed-forward neural networks.
       
4. Local receptive fields allow neurons to be connected only to a window of n-by-n adjacent pixels surrounding it, reducing computational complexity and capitalizing on the correlation of adjacent pixels in an image.
       
5. Shared weights in CNNs allow for all weights and biases to be shared across neurons, regardless of which n-by-n pixel window is evaluated. This sharing of weights and biases leads to translation invariance, which helps in detecting the same features across different parts of the image.
       
6. Pooling is a function that samples sub-regions of the convolutional response map, resulting in a smaller response map. This process reduces the number of parameters and regularizes the CNN, aiding in avoiding overfitting and improving computational efficiency.
       
7. The Faster R-CNN Inception V2 model, which is pre-trained on the COCO (Common Objects in Context) dataset, has two main modules: a deep fully convolutional network that proposes regions, and the Fast R-CNN detector that uses these proposed regions for detection.
       
8. The Region Proposal Network (RPN) uses a sliding window on the convolutional feature maps, generating 9 anchors of different scales and aspect ratios for each window. It computes a value indicating how much these anchors overlap with the ground-truth bounding boxes.
       
9. The Region of Interest Pooling layer (ROIP) regularizes proposed regions into the same size to simplify computation. This step increases the system's processing speed by allowing the same input feature map to be used for multiple object proposals.
       
10. The final step involves the application of the Region-based Convolutional Neural Network (R-CNN) for classifying proposals into classes and refining the bounding boxes for the proposal according to the predicted class. The final output indicates the detected license plates in the image, marked by bounding boxes and associated probabilities.
