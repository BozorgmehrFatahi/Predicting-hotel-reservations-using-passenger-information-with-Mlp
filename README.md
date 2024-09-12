# Predicting-hotel-reservations-using-passenger-information-with-Mlp
# Hotel Booking Prediction Model

This repository contains a machine learning model designed to predict hotel booking behavior using a neural network. The model leverages TensorFlow's Keras API to classify whether a booking will be made based on various features extracted from the dataset.
## Project Overview

The goal of this project is to analyze hotel booking data to determine the likelihood of a booking being made by a user. The model is trained on a dataset that includes several features related to user behavior and booking characteristics. By predicting bookings, hotels can optimize their marketing strategies and improve customer engagement.
Probably, when you want to travel to a city for a vacation, you enter a hotel reservation site through a mobile phone or laptop, through Google search, advertising SMS or directly, and after creating an account, search and select the destination. And you enter your check-in/check-out date. As a result, a list of hotels will be shown to you, and finally, by clicking on different hotels, you will compare them and maybe finally book one of them.
In this project, after pre-processing the data and solving the related challenges, we train a model that can predict whether a user will book the observed hotel or not, based on user search information and other related features. In this way, it is possible to make a suitable decision for each user in the moment, according to the reservation forecast, such as offering a discount or offering other hotels.
## Features

- Data preprocessing and cleaning
- Exploratory data analysis (EDA) with visualizations
- Machine learning model training using Keras
- Evaluation of model performance using ROC AUC score

## Requirements

To run this project, you will need the following Python libraries:

- NumPy
- Pandas
- Keras (with TensorFlow as backend)
- Plotly
- Scikit-learn

You can install these libraries using pip.
## Dataset

The dataset used in this project can be downloaded from Google Drive. The code automatically retrieves and extracts the necessary data files for analysis.
## Key Features

Analyze and process hotel reservation data to predict booking behavior. Preprocessing steps include handling missing values, converting date columns, and creating new attributes that receive relevant information about reservations.
- **Data Cleaning**: 
  - Removal of unnecessary columns (e.g., user identifiers).
  - Handling missing values in critical columns like check-in and check-out dates.

- **Feature Engineering**: 
  - Creation of new features such as:
    - Duration of stay (in days).
    - Days between search date and check-in date.
    - Hour of the search date.
    - Day of the week for search and check-in dates.
    - Month of the search and check-in dates.
  
- **Data Normalization**: 
  - Normalization of booking counts to understand patterns in booking behavior.
## Visualizations

The following visualizations are created:

1. **Search Hour Frequency**:
   - Displays the frequency of bookings based on the hour of the search.
   - The results are saved in a JSON file (search_hour.json).

2. **Day of the Week Frequency**:
   - Compares booking frequencies between booked and not booked records across the days of the week.
   - The results are saved in a JSON file (checkIn_day.json).

3. **Month Frequency**:
   - Analyzes booking frequencies by month for both booked and not booked records.
   - The results are saved in a JSON file (checkIn_date_month.json).

4. **Days Between Search and Check-In**:
   - Visualizes the distribution of days between the search date and the check-in date for both categories.
   - The results are saved in a JSON file (days_between.json).

5. **Length of Stay**:
   - Compares the length of stay for booked and not booked records.
   - The results are saved in a JSON file (los.json).
## Data splitting

The first step involves splitting the dataset into training and testing sets. The target variable is is_booking, which indicates whether a booking was made. The training set consists of 95% of the data, while 5% is reserved for testing. 

## Model Architecture

The model is constructed using a Sequential architecture, which is suitable for a linear stack of layers:

1. **Input Layer**: Accepts input features corresponding to the number of columns in the training data.
2. **Batch Normalization**: This layer normalizes the activations of the previous layer at each batch, which helps in stabilizing and accelerating training.
3. **Hidden Layers**:
   - The first hidden layer consists of 100 neurons with ReLU (Rectified Linear Unit) activation function, which introduces non-linearity into the model.
   - A Dropout layer is included to prevent overfitting by randomly setting a fraction (10%) of input units to zero during training.
   - The second hidden layer consists of 50 neurons, also using ReLU activation, followed by another Batch Normalization layer.
4. **Output Layer**: A single neuron with a sigmoid activation function outputs the probability of a booking being made.

## Model Compilation

The model is compiled with the following configurations:
- **Optimizer**: Adam optimizer, which adapts the learning rate during training.
- **Loss Function**: Binary Crossentropy, suitable for binary classification tasks.
- **Metrics**: Area Under the Curve (AUC) is used to evaluate the model's performance.

## Training Process

The model is trained over 100 epochs with a batch size of 2000. The training process includes:
- **Validation Split**: 20% of the training data is used for validation to monitor the model's performance on unseen data.
- **Callbacks**:
  - **Model Checkpoint**: Saves the best model based on validation AUC during training.
  - **Early Stopping**: Stops training if there’s no improvement in validation AUC for 20 consecutive epochs, restoring the best weights.

## Model Evaluation

After training, the model's performance is evaluated on the test set using the ROC AUC score. This metric provides insight into how well the model distinguishes between bookings and non-bookings.