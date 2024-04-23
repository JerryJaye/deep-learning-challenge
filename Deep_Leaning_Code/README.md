X Repository created on 17 April 2024

### Module 21 - Challenge - Select Candidates For Funding

## Create a binary classifier using Machine Learning and Neural Networks to predict the success of applicants funded by Alphabet Soup.

Introduction
------------

The nonprofit foundation Alphabet Soup wants a tool to help it select the applicants for funding with the best chance of success in their ventures. They have provided csv file containing more than 34,299 organisations that have applied for funding over the years. My task is to use my knowledge of Machine Learning and Neural Networks to build a model that can predict whether the fund applicant's out will be successful.

Files Provided
--------------
Deep_Learning.ipynb              Initial pre-processing and Neural Network model 
Deep_Learing-optimisation.ipynb  Four additional Neural Network Models
Deep_Learning-pca.ipynb          Comparison between Neural Netword with and with PCA function
Deep_Learning-remove_STATUS-SPECIAL_CONSIDERATION.iynb  Neural Network after dropping two Features
Deep_Learning-RandomForest.ipyn   Comparing results with those of the RandomForest model.

Visualisation - PCA and Original Models.png  Comparing an original NN model with a PCA model.
Visualisation - RandomForest.png  How much influence does each feature have on the result?
		
AlphabetSoup.Charity.h5
AlphabetSoup_pca.Charity.h5


1. Pre-processing for Original NN Model.

The Jupiter Notebook for this section of the work is Deap_Leaning.ipynb.

Pre-Process the Data
--------------------

. Alphabet Soup provided the URL for their csv file 'charity_data.csv' to enable the data to be read directly into the program. On reading the data into a dataframe 'application_df', I viewed the first 5 rows to view the data.

.  I investigated the data by using the 'shape' function to view the number of rows and columns and then the '.info()' function to view datatypes and see whether there were any null values to clean. There were not any null values. There were 12 columns with datatypes of int64 and object.

.  I noted that the TARGET for the Model was IS_SUCCESSFUL, while the FEATURES were EIN, NAME, APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMOUNT, SPECIAL_CONSIDERATIONS, and ASK_AMT.

.  I noted that 'EIN' and 'NAME' were non-beneficial ID columns and dropped them from the dataframe. I confirmed their removal using the '.info()' function.

'  I determined the number of values for each feature using the '.nunique' function.

.  I used the .value_counts() function for the APPLICATION_TYPES feature to calculate the number of each APPLICATION_TYPE and binned as 'other' all APPLICTION_TYPES under 250, and then confirmed the success of the binning, and replaced in the 'application_df' dataframe. I did the same with the feature 'CLASSIFICATION'.

.  I encoded the Categorical variables using the pd.get_dummies function. This increased the number of columns to 44, with all columns now being either Boolean or Numerical.

.  I split the pre-processed data into a Target array (IS_SUCCESSFUL) and the Features array, noting that the Target Array had 34,299 rows and 1 column, and the Features Array had 34,299 and 43 columns.

.  I further split the data into train and test datasets, with 20% of the dataset for training and 80% of the data set for testing. The features training dataset comprised 6,860 rows and 43 columns, and the target dataset 6,860 rows, while the testing dataset features were 27,439 rows and 43 columns, and the target dataset 27,439 rows.

. To complete the prepossessing, I scaled the features dataset using the StandardScaler() function.

Define, Compile, Train and Evaluate the Neural Network Model
------------------------------------------------------------

. In this program, I created, compiled, trained, and evaluated and ran two NN Models to create a binary classification using TensorFlow and Keras functions to run as a comparison.

.  Model - Two hidden layer models with 50 epochs. The two hidden layers had 32 and 16 neurons, respectively. 
I used the 'relu' activation function for each hidden layer and 'sigmoid' for the output layer.
The loss and accuracy of this Model were: 

Final training loss: 0.5400
Final training accuracy: 0.7364      73.6%
Final validation loss: 0.5456
Final validation accuracy: 0.7367    73.6%        

The result indicates that the Model performed consistently across the training and testing data sets. The Model correctly predicts about 73.6% of the time. The training and validation loss rates represent how far off the predictions are on average.


Optimise the Model (1) - Compare different HyperParameters with NN models.
--------------------------------------------------

AlphabetSoupCharity instructed me to optimise my Model to achieve a target predictive accuracy highter than 75%.

The Jupiter Notebook for this section of the work is Deep_Leaning-optimising.ipynb. In this section, I attempted to create models using the 'keras.Sequential() function to optimise a Sequential Model with Hyperparameter Options. After creating a working program, a bug got into it, which I couldn't recover, even with expert help.

I prepared a notebook with the same pre-processing, similar to Deep_Learing.ipynb and created 4 different models with varying numbers of hidden layers, numbers of neurons, and activation functions. Their accuracy and loss results are reported here.

Model 1 - 2 Hidden Layers (32 and 16 neurons), Activation functions 'hanh', 'hanh', 'relu', and 'relu' for each layer, sigmoid for the output layer, and 100 Epochs: 

Final training loss: 0.5371
Final training accuracy: 0.7377  73.7%
Final validation loss: 0.5444
Final validation accuracy: 0.7391  73.9%


Model 2 - 4 hidden layer model, with 50 epochs. The four hidden layers had 128, 64, 32, and 16 neurons, respectively, activation function 'relu' for each of the 4 hidden layers, and 'sigmoid' for the output layer and 100 Epochs.

Final training loss: 0.5320
Final training accuracy: 0.7399  74%
Final validation loss: 0.5649
Final validation accuracy: 0.7385  73.9%

Model 3 - 1 Hidden layer model, with 64 neurons, activation function 'relu', and 10 epochs.

Final training loss: 0.5506
Final training accuracy: 0.7288   72.9%
Final validation loss: 0.5517
Final validation accuracy: 0.7362   73.6%

Model 4 - 2 hidden layers (16 and 32 neurons), activation functions 'relu' and 'tanh' and 'sigmoid' out layer.

Final training loss: 0.5453
Final training accuracy: 0.7327  73.4%
Final validation loss: 0.5437
Final validation accuracy: 0.7383  73.8%

I could not create a model with a 75% accuracy rate. All accuracy rates are similar, and you can say essentially the same.


Optimise the Model (2) - Compare Neural Network and PCA Models
-------------------------------------------------------------

The Jupiter Notebook for this work section is Deap_Leaning - pcs.ipynb. In this section, I used the pca function on the data to determine whether it would improve the accuracy.

Pre-Process the Data
-------------------------

. Data pre-processing for this section was the same as for the neural network section (Deep_Learning.ipynb). I commented on many of the outputs, as you can see from the code. 

.I fed scaled data into the Model.

Neural Network Model

. I defined the Neural Network Model with 4 Hidden Layers using the 'relu' Activation function, with the output layer using the 'sigmoid' activation function, over 50 Epochs. The loss and accuracy results were:

Final training loss: 0.5377
Final training accuracy: 0.7356   73.5%
Final validation loss: 0.5480
Final validation accuracy: 0.7394  73.9%

PCA Model

. I imported the PCA from sklearn.decomposition with the Model retaining 95% of the variance. After applying the function I trained the data through the same 4 layer model over 50 Epochs. The loss and accuracy results were:

Final training loss: 0.5399
Final training accuracy: 0.7361   73.6%
Final validation loss: 0.5475
Final validation accuracy: 0.7372   73.7%

I also used Matplotlib to visualise the the loss and accuracy results, both results were essentially the same. 


Optimise Model (3) - Drop STATUS and SPECIAL_CONSIDERATIONS FEATURES
--------------------------------------------------------------------

I noted that the Features STATUS and SPECIAL_CONSIDERATIONS probably had little influence on the accuracy of the results. I dropped these columns and ran the NN Model with 4 hidden layers (128, 64, 32, 16 Neurons) and relu activation function.

There results were:

Final training loss: 0.5381
Final training accuracy: 0.7381     73.8%
Final validation loss: 0.5497
Final validation accuracy: 0.7385   73.8%	

There is was no significant difference to those obtained previously.


Optimise Model (4) - RandomForest Model	
---------------------------------------
I created and ran a RandomForest Model because I wanted to run the 'importances' function to see what features were influencing the result and whether dropping further Features may increase the accuracy of the result.

The Jupiter Notebook is 'Deep_Learaning-RandomForest.

RandomForest Results

Accuracy: 0.7087463556851312  70.9%
              precision    recall  f1-score   support

           0       0.69      0.69      0.69      3196
           1       0.73      0.72      0.73      3664

    accuracy                           0.71      6860
   macro avg       0.71      0.71      0.71      6860
weighted avg       0.71      0.71      0.71      6860

The Model predicted the corrent answer 70.8% of the time. Of all the instances predicted as class 0, 69% were actually class 0. All the instances predicted as class 1, 73% were actuall class 1.

Visualisation

My program printed out the Importances of each of the Features. We can see that the outcome is heavily dependent upon the ASK_AMT, and AFFILIATION_Independent Features.

Results
-------
1. I have commented on all my coding which I believe makes everything clear.

2. I have explained that the purpose of the challenge is to create a model so that Alphabet Soup can predict the outcome of a request for funding from the features provided in their csv file database.

3. The TARGET for the model was IS_SUCCESSFUL, while the FEATURES were EIN, NAME, APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMOUNT, SPECIAL_CONSIDERATIONS, and ASK_AMT.

4. The Features 'EIN" and 'NAME' were removed from the dataset as they did not benefit the application.

5. I selected a combination of 128, 64, 32, and 16 neurons. I created 5 models, each with a different combination of Neurons, Hidden Layers, Activations functions and Epochs. I am curious to see what effect these have on the results. In this challenge, the neural network models perform very similarly. I was unable to reach a successful prediction rate of 75%. The RandomForest model accuracy rates were 2-3% lower than the other.

6. I was not able to achieve the target performance. There are two ways I would try and improve the performance. The RandomForest model Importances visualisations show many features, particularly after applying the pd.get-dummies function, that does not appear to have any significance on the model performance. The other thing I would look at would be to change the collected data so that the features influencing the predictive success are supported with additional data.



