<!--- Futuro: Pasar de markdown a html para poder meterle un css y que no sea tan soso -->
# Machine Learning Portfolio

**Author**: Jaime Blanco Linares.


This repository gathers some of my public Machine Learning work. Each project is constructed on a Jupyter Notebook, which contains **Python** code, text explainations and figures.

## Projects

- ### Linear Regression

  - [Simple Linear Regression - *Boston Housing* dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/P1_SimpleLinearRegression.ipynb): Predicting the median value of a house based on the *number of rooms*. **Exploratory Data Analysis**,  **Ordinary Least Squares regression** and **Scikit-learn’s** implementation.
  
  - [Multiple Linear Regression - *Boston Housing* dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/P2_MultipleLinearRegression.ipynb): Predicting the median value of a house according to the *number of rooms*, *percentage of lower status population* and *pupil-teacher ratio*. **Exploratory Data Analysis**, **Ordinary Least Squares regression** and **Scikit-learn’s** implementation.
  
- ### Polynomial Regression

  - [Polynomial Regression - Insurance Claims dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/P3_PolynomialRegression.ipynb): Predicting the *number of complaints* in an insurance company by year. A comparison between Polynomial Regression and Linear Regression, **Ordinary Least Squares in Polynomial Regression** and **Scikit-learn’s** implementation.

- ### Logistic Regression

  - [Logistic Regression - Wisconsin Breast Cancer dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/maintenance): (UNDER MAINTENANCE) Predicting whether the identified cells are *Benign* or *Malignant*. **Exploratory Data Analysis** and **Scikit-learn’s** implementation.

- ### Classification - Supervised Learning

  - [K-Nearest Neighbours - Wine dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/maintenance): (UNDER MAINTENANCE) Classifying wines according to their characteristic attributes, such as their *alcohol content*, *acidity* or *magnesium levels*, among others. An **in-house implementation based on the k-NN algorithm theory** (in which dataset wines are identified and represented by colors) and **Scikit-learn’s** implementation with **k-Fold Cross-Validation**.

- ### Clustering - Unsupervised Learning

  - [K-Means - Iris Flower dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/maintenance): (UNDER MAINTENANCE) Dividing unlabeled flowers into clusters or groups. An extensive **in-house implementation based on the k-Means algorithm theory**, in which the involved centroids are updated in each iteration and shown graphically. The *Elbow method* is applied.
  
- ### Semi-Supervised Classifier

  - [K-Means - MNIST Dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/maintenance): (UNDER MAINTENANCE) Classifying handwritten numbers using only a very small portion of the labels. **Exploratory Data Analysis** and a semi-supervised approach to the problem: **K-Means algorithm with Scikit-learn**, assignment of labels to clusters according to the most frequent label of each cluster elements, and use of several metrics to evaluate the quality of the models. Further, the effectiveness of **k-NN** and **Logistic Regression** for the resolution of this problem is checked.
  -----
  ## Neural Networks and Deep Learning:

- ### Multi-Layer Perceptron

  - [Multi-Layer Perceptron - MNIST Dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/maintenance): (UNDER MAINTENANCE) Classifying handwritten numbers with multiple fully-connected layers. Designing and training of a Multi-Layer Perceptron based on the **Keras** library (in particular ***tf.keras***), with the justification of each decision taken in the selection of architecture, techniques and parameters, and the subsequent analysis of its **loss** and **accuracy**.
 
  - [**BONUS**: Data Augmentation. Multi-Layer Perceptron - MNIST Dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/maintenance): (UNDER MAINTENANCE) Data Augmentation in MNIST Dataset to check how robust is the model based on an MLP from the previous notebook: **translations**, **rotations**, and **noise**.
  
  - [Multi-Layer Perceptron - Wisconsin Breast Cancer dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/maintenance): (UNDER MAINTENANCE) Predicting whether the identified cells are *Benign* or *Malignant* with Deep Neural Networks (MLP).

- ### Convolutional Neural Networks

  - [Convolutional Neural Networks - MNIST Dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/maintenance): (UNDER MAINTENANCE) Classifying handwritten numbers with Convolutional Neural Networks. Designing, training and results discusion achieved by a Convolutional Neural Network with ***tf.keras***.
  
  - [Convolutional Neural Networks - CIFAR-100 dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/maintenance): (UNDER MAINTENANCE) Classifying 60000 tiny coloured images with Convolutional Neural Networks. Designing, training and results discusion achieved by a Convolutional Neural Network with ***tf.keras***.
  
- ### Autoencoders

  - [Autoencoders - MNIST Dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/maintenance): (UNDER MAINTENANCE) Encoding of handwritten numbers and representation in their latent space. Designing, training and results discusion achieved by an Autoencoder with ***tf.keras***.
  
  - [Convolutional Autoencoders - *Tiny ImageNet* dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/maintenance): (UNDER MAINTENANCE) Reconstruction of coloured noisy images with Convolutional Autoencoders (*Denoiser Autoencoder*). Designing, training and results discusion achieved by a Convolutional Autoencoder with ***tf.keras***.
  
  -----
  ## Personal Project

  - [Multi-Layer Perceptron - MNIST Dataset](https://github.com/bljaime/MachineLearning-Portfolio/blob/master/maintenance): (UNDER MAINTENANCE) Clustering the different types of customers of a bank and predicting their abandonment (*churn rate* prediction). The implementation of this project would have a great impact on the economy of any company with a large number of clients, since the models we have designed predict with a high accuracy whether a client will eventually leave the bank, and in that case would allow the entity to carry out a series of actions that guarantee their loyalty. In this notebook you can see: **Data Preprocessing**, **study of the Correlation between the attributes**, **Data Visualization*, **K-Means Clustering**, **techniques to handle Imbalanced Data**, **Logistic Regression** and **Multi-Layer Perceptron** based models, as well as a detailed justification of the choice of the different hyperparameters of the Neural Network model.
  
  
## About me

Jaime Blanco Linares.
Computer Engineer.
24 y/o, living in Madrid (Spain).

Any comment or suggestion about the portfolio, job offers or collaboration proposals, will be thankfully received and replied at blancolinares.jaime@gmail.com
