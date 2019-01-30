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

  - [Logistic Regression - Wisconsin Breast Cancer dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/maintenance): (UNDER MAINTENANCE) Predicting whether the identified cells are *Benign* or *Malignant*. **Exploratory Data Analysis** and **Scikit-learn’s** implementation.

- ### Classification - Supervised Learning

  - [K-Nearest Neighbours - Wine dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/maintenance): (UNDER MAINTENANCE) Classifying wines according to their characteristic attributes, such as their *alcohol content*, *acidity* or *magnesium levels*, among others. An **in-house implementation based on the k-NN algorithm theory** (in which dataset wines are identified and represented by colors), and **Scikit-learn’s** implementation with **k-Fold Cross-Validation**.

- ### Clustering - Unsupervised Learning

  - [K-Means - Iris Flower dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/maintenance): (UNDER MAINTENANCE) Dividing unlabeled flowers into clusters or groups. An extensive **in-house implementation based on the k-Means algorithm theory**, in which the involved centroids are updated in each iteration and shown graphically. The *Elbow method* is applied.
  
- ### Semi-Supervised Classifier

  - [K-Means - MNIST dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/maintenance): (UNDER MAINTENANCE) Classifying handwritten numbers using only a very small portion of the labels. **Exploratory Data Analysis** and a semi-supervised approach to the problem: **K-Means algorithm with Scikit-learn**, assignment of labels to clusters according to the most frequent label of each cluster elements, and use of several metrics to evaluate the quality of the models. Further, the effectiveness of **k-NN** and **Logistic Regression** for the resolution of this problem is checked.
  -----
  ## Neural Networks and Deep Learning:

- ### Multi-Layer Perceptron

  - [Multi-Layer Perceptron - MNIST dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/P8_MultiLayerPerceptron_MNIST.ipynb): Classifying handwritten numbers with multiple fully-connected layers. Designing and training of a Multi-Layer Perceptron based on the **Keras** library (in particular ***tensorflow.keras***), with the justification of each decision taken in the selection of architecture, techniques and parameters, and the subsequent analysis of its **loss** and **accuracy**.
 
  - [Data Augmentation. Multi-Layer Perceptron - MNIST dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/P9_DataAugmentation_MLP_MNIST.ipynb): **Data Augmentation** in MNIST dataset images, concretely **translating**, **rotating**, and **adding noise**, to check how robust is the Multi-Layer Perceptron model from the previous notebook, trained only with non-distorted images.
  
  - [Multi-Layer Perceptron - Wisconsin Breast Cancer dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/P10_MultiLayerPerceptron_BreastCancer.ipynb): Classifying whether the identified cells present in digitalized images of breast mass are *Benign* or *Malignant* with a Multi-Layer Perceptron model.

- ### Convolutional Neural Networks

  - [Convolutional Neural Networks - MNIST dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/P11_ConvolutionalNN_MNIST.ipynb): Classifying original and distorted handwritten numbers in MNIST with Convolutional Neural Networks. Designing, training and results discusion achieved by a Convolutional Neural Network with ***tensorflow.keras***.
  
  - [Convolutional Neural Networks - CIFAR-10 dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/maintenance): (UNDER MAINTENANCE) Classifying 60000 tiny coloured images with Convolutional Neural Networks. Designing, training and results discusion achieved by a Convolutional Neural Network with ***tensorflow.keras***.
  
- ### Autoencoders

  - [Autoencoders - MNIST dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/maintenance): (UNDER MAINTENANCE) Encoding of handwritten numbers and representation in their latent space. Designing, training and results discusion achieved by an Autoencoder with ***tensorflow.keras***.
  
  - [Convolutional Autoencoders - *Tiny ImageNet* dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/maintenance): (UNDER MAINTENANCE) Reconstruction of coloured noisy images with Convolutional Autoencoders (*Denoiser Autoencoder*). Designing, training and results discusion achieved by a Convolutional Autoencoder with ***tensorflow.keras***.
  
  -----
  
    - [**BONUS**: How to save your already trained models with Keras](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/maintenance): (UNDER MAINTENANCE) Re-training your model whenever you want to use it is not necessary. This notebook describes how to **save your trained models**, so you can reload them in the future when you want to use them.
  
  -----
  ## Personal Project

  - [Bank Customer Clustering and Churn Prediction](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/Customer_Clustering_and_Churn_Prediction.ipynb): Clustering the different types of customers of a bank and predicting their abandonment (*churn rate* prediction). The implementation of this project would have a great impact on the economy of any company with a large number of customers, since the models we have designed can classify between loyal customers and customers with a high likelihood of leaving the bank. In this notebook you can see: An **Extensive Exploratory Data Analysis (Visualizing the Distributions, Valuable Information about Customers and Correlation between the Variables)**, **Customer Clusterization (Visualizing the New Space, Finding Clusters and Visualizing and Understanding Clusters)**, **How to Handle Imbalanced Classes**, **Classifiers based on Neural network models (Multi-Layer Perceptron)** and **Receiver Operator Characteristic (ROC) Curve as an alternative to accuracy metric**.
  
  
## About me

Jaime Blanco Linares.
Computer Engineer.
24 y/o, living in Madrid (Spain).

Any comment, suggestion about the portfolio or collaboration proposal, will be thankfully received and replied at blancolinares.jaime@gmail.com
