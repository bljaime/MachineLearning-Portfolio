<!--- Futuro: Pasar de markdown a html para poder meterle un css y que no sea tan soso -->
# Machine Learning Portfolio

**Author**: Jaime Blanco Linares.


This repository gathers some of my public Machine Learning work. Each project is constructed on a *Jupyter Notebook*, which contains **Python** code, text explainations and figures.  **NOTE**: I am doing my best to decorate and upload the missing notebooks to the repository as soon as possible, as I want it to have a didactic purpose.

## Projects

- ### Linear Regression

  - [Simple Linear Regression - *Boston Housing* dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/P1_SimpleLinearRegression_Boston.ipynb): Predicting the median value of a house based on the *number of rooms*. **Exploratory Data Analysis**,  **Ordinary Least Squares regression** and **Scikit-learn’s** implementation.
  
  - [Multiple Linear Regression - *Boston Housing* dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/P2_MultipleLinearRegression_Boston.ipynb): Predicting the median value of a house according to the *number of rooms*, *percentage of lower status population* and *pupil-teacher ratio*. **Exploratory Data Analysis**, **Ordinary Least Squares regression** and **Scikit-learn’s** implementation.
  
- ### Polynomial Regression

  - [Polynomial Regression - Insurance Claims dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/P3_PolynomialRegression_Insurance.ipynb): Predicting the *number of complaints* in an insurance company by year. A comparison between Polynomial Regression and Linear Regression, **Ordinary Least Squares in Polynomial Regression** and **Scikit-learn’s** implementation.

- ### Logistic Regression

  - [Logistic Regression - Wisconsin Breast Cancer dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/P4_LogisticRegression_BreastCancer.ipynb): Predicting whether the identified cells are *Benign* or *Malignant*. **Exploratory Data Analysis**, several models **Scikit-learn’s** implementation and selection of the fittest classifier.

- ### Classification - Supervised Learning

  - [K-Nearest Neighbours - Wine dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/P5_kNearestNeighbor_Wine.ipynb): Classifying wines according to their characteristic attributes, such as their *alcohol content*, *acidity* or *magnesium levels*, among others. **In-house implementation based on the k-NN algorithm criteria** (in which wines classification boundaries are represented by colors), and **Scikit-learn’s** implementation with **k-Fold Cross-Validation**.

- ### Clustering - Unsupervised Learning

  - [K-Means - Iris Flower dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/P6_K_Means_IrisFlower.ipynb): Dividing unlabeled flowers into clusters or groups. Extensive **in-house implementation based on the k-Means algorithm theory**, in which the involved centroids position is updated and graphically shown in each iteration. Within Cluster Sum of Squares (WCSS) is calculated for different k-Means executions, varying the number of clusters, and the *Elbow method* is applied.
  
- ### Semi-Supervised Classifier

  - [K-Means - MNIST dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/P7_SemiSupervised_MNIST.ipynb): Classifying handwritten numbers using only a very small portion of the labels. **Exploratory Data Analysis**, **Semi-Supervised** approach (**K-Means algorithm with Scikit-learn**, assignment of labels to clusters according to the most frequent label of each cluster elements, checking only 0.25% of the labels) and implementation of a **k-NN** classifier to discuss both models performance.
  -----
  ## Neural Networks and Deep Learning:

- ### Multi-Layer Perceptron

  - [Multi-Layer Perceptron - MNIST dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/P8_MultiLayerPerceptron_MNIST.ipynb): Classifying handwritten numbers with multiple fully-connected layers. Designing and training of a Multi-Layer Perceptron based on ***tensorflow.keras***, with the justification of each decision taken in the selection of architecture, techniques and parameters, and the subsequent analysis of its **loss** and **accuracy** (With demonstrative purposes, since MLP is not suitable for image classification).
 
  - [Data Augmentation. Multi-Layer Perceptron - MNIST dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/P9_DataAugmentation_MLP_MNIST.ipynb): **Data Augmentation** in MNIST dataset images, concretely **translating**, **rotating**, and **adding noise**, to check how robust is the Multi-Layer Perceptron model from the previous notebook, trained only with non-distorted images.
  
  - [Multi-Layer Perceptron - Wisconsin Breast Cancer dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/P10_MultiLayerPerceptron_BreastCancer.ipynb): Classifying whether the identified cells present in digitalized images of breast mass are *Benign* or *Malignant* with a Multi-Layer Perceptron based on ***tensorflow.keras***.

- ### Convolutional Neural Networks

  - [Convolutional Neural Networks - MNIST dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/P11_ConvolutionalNN_MNIST.ipynb): Classifying original and distorted handwritten numbers in MNIST with Convolutional Neural Networks. Designing, training and results discusion achieved by a Convolutional Neural Network with ***tensorflow.keras***.
  
  - [Convolutional Neural Networks - CIFAR-10 dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/P12_ConvolutionalNN_CIFAR10.ipynb): Classifying 60000 tiny colored images with Convolutional Neural Networks. Designing, training and results discusion achieved by a Convolutional Neural Network with ***tensorflow.keras***.
  
- ### Autoencoders

  - [Convolutional Autoencoders - MNIST dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/P13_ConvolutionalAutoencoders_MNIST.ipynb): Reconstruction of handwritten **MNIST** numbers with Convolutional Autoencoders, (*Denoiser Autoencoder*) and linear interpolation in latent space. Designing, training and results discusion achieved by a Convolutional Autoencoder with ***tensorflow.keras***.
  
  - [Convolutional Autoencoders - *Tiny ImageNet* dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/P14_ConvolutionalAutoencoders_Tiny_imagenet.ipynb): Reconstruction of colored noisy images with Convolutional Autoencoders, (*Denoiser Autoencoder*) and linear interpolation in latent space. Designing, training and results discusion achieved by a Convolutional Autoencoder with ***tensorflow.keras***.

  ## Personal Project

  - [Bank Customer Clustering and Churn Prediction](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/Customer_Clustering_and_Churn_Prediction.ipynb): Clustering the different types of customers of a bank and predicting their abandonment (*churn rate* prediction). The implementation of this project would have a great impact on the economy of any company with a large number of customers, since the models we have designed can classify between loyal customers and customers with a high likelihood of leaving the bank. In this notebook you can see: An **Extensive Exploratory Data Analysis (Visualizing the Distributions, Valuable Information about Customers and Correlation between the Variables)**, **Customer Clusterization (Visualizing the New Space, Finding Clusters and Visualizing and Understanding Clusters)**, **How to Handle Imbalanced Classes**, **Classifiers based on Neural network models (Multi-Layer Perceptron)** and **Receiver Operator Characteristic (ROC) Curve as an alternative to accuracy metric**.
  
## Acknowledgements

The development of this material would not have been possible without the help of [Dot CSV](https://www.youtube.com/channel/UCy5znSnfMsDwaLlROnZ7Qbg).
  
## About me

Jaime Blanco Linares.
Computer Engineer.
24 y/o, living in Madrid (Spain).

Any comment, suggestion about the portfolio or collaboration proposal, will be thankfully received and replied at blancolinares.jaime@gmail.com
