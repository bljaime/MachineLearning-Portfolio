<!--- Futuro: Pasar de markdown a html para poder meterle un css y que no sea tan soso -->
# Machine Learning Portfolio

**Author**: Jaime Blanco Linares.

This repository gathers some of my public Machine Learning work. Each project is constructed on a *Jupyter Notebook*, which contains **Python** code, text explainations and figures.

**NOTE**: Often you get the message "*Sorry, something went wrong. Reload?*" when trying to visualize an *.ipynb* (*Jupyter Notebook* file) on a GitHub blob page. This probably concerns the GitHub notebook viewing tool, as sometimes GitHub fails to render the *.ipynb* notebooks. As it seems to be a temporary issue with their backend, please refresh the page until you can view the entire file in case you have this problem, or, in exceptional circumstances, use [nbviewer](https://nbviewer.jupyter.org/).

## Projects

- ### Linear Regression

  - [Simple Linear Regression - *Boston Housing* dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/notebook/SimpleLinearRegression_Boston.ipynb): Predicting the median value of a house based on the *number of rooms*. **Exploratory Data Analysis**,  **Ordinary Least Squares regression** and **Scikit-learn’s** implementation.
  
  - [Multiple Linear Regression - *Boston Housing* dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/notebook/MultipleLinearRegression_Boston.ipynb): Predicting the median value of a house according to the *number of rooms*, *percentage of lower status population* and *pupil-teacher ratio*. **Exploratory Data Analysis**, **Ordinary Least Squares regression** and **Scikit-learn’s** implementation.
  
- ### Polynomial Regression

  - [Polynomial Regression - Insurance Claims dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/notebook/PolynomialRegression_Insurance.ipynb): Predicting the *number of complaints* in an insurance company by year. Comparison between Polynomial Regression and Linear Regression, **Ordinary Least Squares in Polynomial Regression** and **Scikit-learn’s** implementation.

- ### Logistic Regression

  - [Logistic Regression - Wisconsin Breast Cancer dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/notebook/LogisticRegression_BreastCancer.ipynb): Predicting whether the identified cells are *Benign* or *Malignant*. **Exploratory Data Analysis**, several models **Scikit-learn’s** implementation and selection of the fittest classifier.

- ### Classification - Supervised Learning

  - [K-Nearest Neighbours - Wine dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/notebook/kNearestNeighbor_Wine.ipynb): Classifying wines according to their characteristic attributes, such as their *alcohol content*, *acidity* or *magnesium levels*, among others. **In-house implementation based on the k-NN algorithm criteria**, in which wines classification boundaries are delimited by colors. **Scikit-learn’s** implementation with **k-Fold Cross-Validation**.

- ### Clustering - Unsupervised Learning

  - [K-Means - Iris Flower dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/notebook/kMeans_IrisFlower.ipynb): Dividing unlabeled flowers into clusters or groups. Extensive **in-house implementation based on the k-Means algorithm theory**, in which the positions of the centroids involved are updated and graphically shown in each iteration. **Within Cluster Sum of Squares** (WCSS) is calculated for different k-Means executions, varying the number of clusters, and the **Elbow method** is applied.
  
- ### Semi-Supervised Classifier

  - [K-Means - MNIST dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/notebook/SemiSupervised_MNIST.ipynb): Classifying handwritten numbers using only a very small portion of the labels. **Exploratory Data Analysis**, **Semi-Supervised** approach (**K-Means algorithm with Scikit-learn**, assignment of labels to clusters according to the most frequent label of each cluster elements, checking only 0.25% of the labels) and implementation of a **k-NN** classifier to discuss both models performance.
  -----
  ## Neural Networks and Deep Learning:

- ### Multi-Layer Perceptron (MLP)

  - [Multi-Layer Perceptron - MNIST dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/notebook/MultiLayerPerceptron_MNIST.ipynb): **Classifying handwritten numbers** with multiple fully-connected layers. Designing and training a Multi-Layer Perceptron based on ***tensorflow.keras*** (only with demonstrative purposes, since MLP is not suitable for image classification).
 
  - [Data Augmentation. Multi-Layer Perceptron - MNIST dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/notebook/DataAugmentation_MLP_MNIST.ipynb): **Data Augmentation** in MNIST dataset images, concretely **translating**, **rotating**, and **adding noise**, to check how robust is the Multi-Layer Perceptron model from the previous notebook when **classifying distorted images**.
  
  - [Multi-Layer Perceptron - Wisconsin Breast Cancer dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/notebook/MultiLayerPerceptron_BreastCancer.ipynb): **Classifying** whether the identified **cells** present in digitalized images of breast mass are ***Benign*** or ***Malignant*** with a Multi-Layer Perceptron based on ***tensorflow.keras***.

- ### Convolutional Neural Networks (CNN)

  - [Convolutional Neural Networks - MNIST dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/notebook/ConvolutionalNN_MNIST.ipynb): **Classifying original and distorted handwritten numbers** in MNIST with Convolutional Neural Networks. Designing, training and results discusion achieved by a Convolutional Neural Network with ***tensorflow.keras***.
  
  - [Convolutional Neural Networks - CIFAR-10 dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/notebook/ConvolutionalNN_CIFAR10.ipynb): **Classifying** 60000 tiny **colored images** with Convolutional Neural Networks. Designing, training and results discusion achieved by a Convolutional Neural Network with ***tensorflow.keras***.
  
- ### Autoencoders

  - [Convolutional Autoencoders - MNIST dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/notebook/ConvolutionalAutoencoders_MNIST.ipynb): **Reconstruction** of handwritten **MNIST** numbers with Convolutional Autoencoders and **linear interpolations in latent space**. Architecture designing, training and results discusion achieved by a Convolutional Autoencoder with ***tensorflow.keras***.
  
  - [Convolutional Autoencoders - *Tiny ImageNet* dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/notebook/ConvolutionalAutoencoders_Tiny_imagenet.ipynb): **Reconstruction** of colored noisy images with Convolutional Autoencoders, ***Denoiser Autoencoder*** and **linear interpolation in latent space**. Designing, training and results discusion achieved by a Convolutional Autoencoder with ***tensorflow.keras***.

  ## Personal Project

  - [Bank Customer Clustering and Churn Prediction](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/notebook/Customer_Clustering_and_Churn_Prediction.ipynb): Clustering the different types of customers of a bank and predicting their abandonment (*churn rate* prediction). The implementation of this project would have a great impact on the economy of any company with a large number of customers, since the models we have designed can classify between loyal customers and customers with a high likelihood of leaving the bank. In this notebook you can see: An **Extensive Exploratory Data Analysis (Visualizing the Distributions, Valuable Information about Customers and Correlation between the Variables)**, **Customer Clusterization (Visualizing the New Space, Finding Clusters with k-Means and Visualizing and Understanding Clusters)**, **How to Handle Imbalanced Classes**, **Classifiers based on Neural network models (Multi-Layer Perceptron)** and **Receiver Operator Characteristic (ROC) Curve as an alternative to accuracy metric**.
  
## Acknowledgements

The development of this material would not have been possible without [Dot CSV](https://www.youtube.com/channel/UCy5znSnfMsDwaLlROnZ7Qbg).
  
## About me

Jaime Blanco Linares.
Computer Engineer.
24 y/o, living in Madrid (Spain).

Any comment, suggestion about the portfolio or collaboration proposal, will be thankfully received and replied at blancolinares.jaime@gmail.com
