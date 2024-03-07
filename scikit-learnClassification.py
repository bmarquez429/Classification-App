"""
         File: scikit-learnClassification.py
 Date Created: February 6, 2024
Date Modified: March 6, 2024
----------------------------------------------------------------------------------------
Walk the user through the steps in training and testing one or more classifiers using a 
selection of algorithms that are implemented in scikit-learn."
----------------------------------------------------------------------------------------
"""

from helperFunctions import displayData, printTrainingResults, setAllOptions, \
                            setOptions, setStage
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import modelParams
import numpy as np
import os
import pandas as pd
import streamlit as st

st.markdown("## Binary Classification Using `scikit-learn`")
             
#------
# Begin
#------

if "stage" not in st.session_state:
    st.session_state.stage = 0
    
if "case1" not in st.session_state:
   st.session_state.case1 = ""
   
if "case2" not in st.session_state:
   st.session_state.case2 = ""
   
if "case3" not in st.session_state:
   st.session_state.case3 = ""
    
if "datasetAvailability" not in st.session_state:
   st.session_state.datasetAvailability = ""
   
if "variables" not in st.session_state:
   st.session_state.variables = []
      
if "trainingSucceeded" not in st.session_state:   
   st.session_state.trainingSucceeded = []
   
if "trainingFailed" not in st.session_state:   
   st.session_state.trainingFailed = []  
   
toSave = False   
toSplit = False
         
if st.session_state.stage == 0 or st.session_state.stage == 1:
    
   st.markdown("This web app will walk you through the steps in training and testing one or more \
                binary classifiers using a selection of algorithms that are implemented in \
                `scikit-learn`.")
   st.markdown("Click the button below to begin.")
   st.button(label = 'Begin', on_click = setStage, args = [1])
   
#----------------------------------------------
# Select the kind of dataset that will be used.
#----------------------------------------------

if st.session_state.stage == 1:
    
   st.write(" ")
   
   case1 = "I have a dataset that I would like to split into a training set and a test set." 
   case2 = "I already have a training set and a test set."
   case3 = "I would like to use one of the standard datasets that comes with scikit-learn."
   datasetAvailability = st.radio(label = "Please select the kind of dataset that you will use.",
                                  options = [case1, case2, case3],
                                  index = None)
   
   st.session_state.case1 = case1
   st.session_state.case2 = case2
   st.session_state.case3 = case3
   st.session_state.datasetAvailability = datasetAvailability
   
   st.button(label = "Next", 
             on_click = setStage,
             args = [2])
   
#--------------------------
# Get the selected dataset.
#--------------------------

if st.session_state.stage >= 2:
    
   st.sidebar.button(label = "Reset", 
                     help = "Clicking this button at any time will bring you back to the intro page",
                     on_click = setStage, 
                     args = [0])
   
   case1 = st.session_state.case1
   case2 = st.session_state.case2
   case3 = st.session_state.case3
   datasetAvailability = st.session_state.datasetAvailability
   variables = st.session_state.variables
   
   if datasetAvailability == case1:
      
      uploadedDataset = st.sidebar.file_uploader(label = "Upload the dataset in `csv` file format.",
                                                 on_change = setStage, 
                                                 args = [3])
 
      if uploadedDataset is not None:
  
         dataset = pd.read_csv(uploadedDataset)
         displayData(data = dataset, header = "**Dataset**")
         variables = dataset.columns.tolist()
         datasetHolder = dataset
         
      else:
         setStage(2)
   
   elif datasetAvailability == case2:
    
        uploadedTrainSet = st.sidebar.file_uploader(label = "Upload the training set in `csv` file format.",
                                                    on_change = setStage, 
                                                    args = [3])
   
        if uploadedTrainSet is not None:
    
           trainSet = pd.read_csv(uploadedTrainSet)
           displayData(data = trainSet, header = "**Training Set**")
           variables = trainSet.columns.tolist()
           datasetHolder = trainSet
           
        else:
           setStage(2)
           
   elif datasetAvailability == case3:
       
        datasetName = st.sidebar.selectbox(label = "Select a dataset.",
                                           options = ["Breast Cancer", "Digits", "Iris", "Wine"],
                                           index = None,
                                           on_change = setStage,
                                           args = [3],
                                           placeholder = "Select a dataset")
        
        st.sidebar.write("You can find some information about these datasets in \
                         [scikit-learn](https://scikit-learn.org/stable/datasets/toy_dataset.html).")
        
        if datasetName is not None:
            
           if datasetName == "Breast Cancer":
              dataset = load_breast_cancer(return_X_y = True, as_frame = True)
           elif datasetName == "Digits":
                dataset = load_digits(return_X_y = True, as_frame = True)
           elif datasetName == "Iris":
                dataset = load_iris(return_X_y = True, as_frame = True)   
           elif datasetName == "Wine":
                dataset = load_wine(return_X_y = True, as_frame = True)
            
           dataset = pd.concat([dataset[0], dataset[1]], axis = 1)
           displayData(data = dataset, header = "**Dataset**")
           variables = dataset.columns.tolist()
           datasetHolder = dataset
           
        else:
          setStage(2)
          
#----------------------------
# Select the target variable.
#----------------------------

if st.session_state.stage >= 3:
    
   if datasetAvailability in [case1, case2]:
       
      targetVariable = st.selectbox(label = "Select the target variable.",
                                    options = variables,
                                    index = None,
                                    on_change = setStage,
                                    args = [4],
                                    placeholder = "Select a variable")
      
      if targetVariable is None:
         setStage(3)
      
   else:
       
      targetVariable = "target"
      messagePart = "For the purpose of this binary classification, you can binarize the target variable \
                     by identifying the classes that will be regarded as positive."
      
      if datasetName == "Breast Cancer":
          
         st.write("The target variable has 2 classes. The 2 classes are: malignant (0) and benign (1).")  
         toSplit = True
         
      elif datasetName == "Digits":
           st.write("The target variable has 10 classes where each class refers to a digit. " + messagePart) 
      elif datasetName == "Iris":
           st.write("The target variable has 3 classes. The 3 classes are the 3 different types \
                     of the Iris flowering plant: setosa (0), versicolor (1), and virginica (3). " + \
                     messagePart) 
      elif datasetName == "Wine":
           st.write("The target variable has 3 classes. The 3 classes are 3 different types of wine. " + \
                     messagePart) 
          
   if targetVariable is not None:
      
      classes = datasetHolder[targetVariable].unique()
      nUniqueValues = len(classes)
       
      if nUniqueValues > 2:
         
         if datasetAvailability in [case1, case2]:
             
            st.markdown(":red[`" + targetVariable + "` is not a binary variable. Please change \
                         your selection.]")
            setStage(3)
            
         else:
             
            positiveClasses = st.multiselect(label = "Select the classes that will be regarded as positive.",
                                             options = classes,
                                             on_change = setStage,
                                             args = [3],
                                             placeholder = "Select class(es)")
            
            if len(positiveClasses) > 0:
               
               if len(positiveClasses) == nUniqueValues:
                   
                  st.markdown(":red[You cannot select all the classes since at least one class should be \
                               regarded as negative. Please edit your selection.]")
                  setStage(3)
                  
               else:
                  
                  st.write("Click the button below to complete the selection of the classes that will be \
                            regarded as positive.")         
                         
                  st.button(label = "Complete selection", key = "positiveClassesSelection", on_click = setStage, 
                            args = [5])
                  
                  dataset["binarized target"] = dataset[targetVariable].apply(lambda x: 1 if x in positiveClasses else 0)
    
            else:
               setStage(3)
            
   features = variables.copy()
   
   if targetVariable in features:
      features.remove(targetVariable)

#-----------------------------------------------------------------------------
# Select features that are categorical or that will be treated as categorical.
#-----------------------------------------------------------------------------
      
if st.session_state.stage >= 4:
   
   if datasetAvailability in [case1, case2]:
       
      st.write(" ") 
    
      label1 = "Select features that are categorical or that will be treated as categorical. "
      label2 = "Features that are not selected are continuous or will be treated as continuous. "
      label3 = "Select `None` if no features are categorical."
      categoricalFeatures = st.multiselect(label = label1 + label2 + label3,
                                           options = features + ["None"],
                                           on_change = setStage,
                                           args = [4],
                                           placeholder = "Select feature(s) or select None")
   
      if len(categoricalFeatures) > 0:
      
         if len(categoricalFeatures) >= 2 and "None" in categoricalFeatures:
          
            st.markdown(":red[You cannot select None together with one or more features. Please edit \
                         your selection.]")
            setStage(4)
         
         else:
         
            st.write("Click the button below to complete the selection of categorical features \
                      or if no features are categorical.")         
         
            if datasetAvailability in [case1, case3]:          
               st.button(label = "Complete selection", key = "catFeaturesSelection", on_click = setStage, 
                         args = [5])
            else:
               st.button(label = "Complete selection", key = "catFeaturesSelection", on_click = setStage, 
                         args = [6]) 
             
      else:
          setStage(4)
     
#--------------------------------------------------------------------------------------
# If case1 or case3 is selected, split the dataset into a training set and a test set .
#--------------------------------------------------------------------------------------         
 
if st.session_state.stage >= 5 or toSplit:
   
   if datasetAvailability in [case1, case3]:
       
      st.write(" ")
      st.write("Use the slider below to set the train-test split ratio.")
   
      cols = st.columns(2)
   
      with cols[0]:
           trainingSetSize = st.slider(label = "train-test split ratio", 
                                       min_value = 0.0, 
                                       max_value = 1.0, 
                                       value = 0.7,
                                       on_change = setStage,
                                       args = [5])
        
      testSetSize = 1 - trainingSetSize
   
      st.write("training set size = %.2f" % trainingSetSize)
      st.write("    test set size = %.2f" % testSetSize)
      
      if nUniqueValues > 2:
          
         X = dataset[features + [targetVariable]]
         y = dataset[["binarized target"]]
      
      else:
          
         X = dataset[features]
         y = dataset[[targetVariable]]
         
      xTrain, xTest, yTrain, yTest = train_test_split(X, y, train_size = trainingSetSize)
      
      st.write(" ")
      st.write("Click the button below to confirm the selected train-test split ratio and \
                to display the randomly formed training set.")
      
      st.button(label = "Display training set", on_click = setStage, args = [6])
   
#---------------------------------------------------------
# If case1 or case3 is selected, display the training set. 
#---------------------------------------------------------

if st.session_state.stage >= 6: 
    
   if datasetAvailability in [case1, case3]:
              
      if "trainTestSplit" not in st.session_state:
          
         st.session_state.trainTestSplit = {}
         st.session_state.trainTestSplit["xTrain"] = xTrain
         st.session_state.trainTestSplit["xTest"] = xTest
         st.session_state.trainTestSplit["yTrain"] = yTrain
         st.session_state.trainTestSplit["yTest"] = yTest
      
      xTrain = st.session_state.trainTestSplit["xTrain"]
      xTest = st.session_state.trainTestSplit["xTest"]
      yTrain = st.session_state.trainTestSplit["yTrain"]
      yTest = st.session_state.trainTestSplit["yTest"]
         
      trainSet = pd.concat([xTrain, yTrain], axis = 1)
      testSet = pd.concat([xTest, yTest], axis = 1)
      
      displayData(data = trainSet, header = "**Training Set**")
      
      st.write("Train Set = ", trainSet.shape)
   
   st.write(" ")
   st.write("Click the button below to select at least one model to train.")     
    
   st.button(label = "Select one or more models", on_click = setStage, args = [7]) 
    
#----------------------------------------------------------
# Transform columns and select at least one model to train.
#----------------------------------------------------------

if st.session_state.stage >= 7: 
   
   if datasetAvailability in [case1, case2]:
       
      if categoricalFeatures[0] == "None":
         colTransformer = ColumnTransformer(transformers = [("continuous", StandardScaler(), features)])
      else:
         colTransformer = ColumnTransformer(transformers = [("categorical", OneHotEncoder(drop = "first"), 
                                                          categoricalFeatures)],
                                         remainder = StandardScaler())
         
   else:
       colTransformer = ColumnTransformer(transformers = [("continuous", StandardScaler(), features)])
       
   colTransformer.fit(trainSet[features])
      
   xTrain = colTransformer.transform(trainSet[features])
   
   if nUniqueValues > 2:
      yTrain = trainSet["binarized target"].to_numpy()
   else:
      yTrain = trainSet[targetVariable].to_numpy()
   
   model1 = "Decision Tree Classifier"
   model2 = "Gaussian Process Classifier"
   model3 = "k-Nearest Neighbors Classifier"
   model4 = "Logistic Regression"
   model5 = "Random Forest Classifier"
   model6 = "Support Vector Classifier"
   
   st.checkbox(label = model1, key = model1, on_change = setAllOptions)
   st.checkbox(label = model2, key = model2, on_change = setAllOptions)
   st.checkbox(label = model3, key = model3, on_change = setAllOptions)
   st.checkbox(label = model4, key = model4, on_change = setAllOptions)
   st.checkbox(label = model5, key = model5, on_change = setAllOptions)
   st.checkbox(label = model6, key = model6, on_change = setAllOptions)
   st.checkbox("Select all models", key = 'allOptions', on_change = setOptions)
   
   options = [model1, model2, model3, model4, model5, model6]
   selected = 0
   for option in options:
       selected += st.session_state[option]
   
   st.write("Click the button below to complete the selection of models.")
   clicked = st.button(label = "Complete selection", key = "modelSelection", on_click = setStage, args = [8])
   
   if clicked and selected == 0:
       
      st.markdown(":red[Please select at least one model to train.]")
      setStage(7)
   
#-----------------------------------------------------------------
# Click to set the values of the parameters of the selected model.
#-----------------------------------------------------------------
    
if st.session_state.stage >= 8:
    
    trueOptions = []
    for option in options:
        
        if st.session_state[option] is True:
            trueOptions.append(option)
            
    m = len(trueOptions)
    
    st.write(" ")
            
    if m == 1:
       st.write("You selected " + trueOptions[0] + ". Click the button \
                 below to set the values of the model parameters.")
    elif m == 2:
         st.write("You selected " + trueOptions[0] + " and " + trueOptions[1] + ". \
                   Click the button below to set the values of the model parameters.") 
    elif m >= 3 and m != len(options):
        
         st.write("You selected the following models:")
         
         for i in range(m):
             st.write("(" + str(i + 1) + ") " + trueOptions[i])
             
         st.write("Click the button below to set the values of the model parameters.")
         
    elif m == len(options):
         st.write("You selected all the available models. Click the button below to set the values \
                   of the model parameters.")
             
    st.button(label = "Set model parameters", on_click = setStage, args = [9])

#--------------------------------------------------------
# Set the values of the parameters of the selected model.
#--------------------------------------------------------
    
if st.session_state.stage >= 9:
    
   st.write("")
    
   paramsValues = {}
    
   if m == 1:
      paramsValues[trueOptions[0]], disabled = modelParams.setModelParams(trueOptions[0])
   else:
      
      if m >= 5:
         st.write("To scroll through the tabs, click on a tab and use the left-arrow and right-arrow keys.")
       
      tabs = st.tabs([trueOptions[i] for i in range(m)])
    
      for i in range(m):
    
          with tabs[i]:
        
               paramsValues_i, disabled = modelParams.setModelParams(trueOptions[i])
        
          paramsValues[trueOptions[i]] = paramsValues_i
     
   if disabled == 0:     
      st.write("Click the `Reset model parameters` button to set the model parameters \
                again or click the `Confirm` button to confirm the \
                assigned values of the model parameters.")
   elif disabled == 1:
        st.markdown("Please correct the input above as indicated by the message in :red[red] to enable the \
                    `Confirm model parameters` button.")    
   else:
        st.markdown("Please correct the inputs above as indicated by the messages in :red[red] to enable the \
                    `Confirm model parameters` button.") 
   
   cols = st.columns(3)
   
   with cols[0]:
        st.button(label = "Reset model parameters", on_click = setStage, args = [8])   
        
   with cols[1]:
        st.button(label = "Confirm model parameters", on_click = setStage, args = [10], disabled = disabled)

#-------------
# Train model.
#-------------

if st.session_state.stage >= 10:
    
   if "models" not in st.session_state:
      st.session_state.models = {}
      
   if m == 1:
      messagePart = "model"
   else:
      messagePart = "models"
      
   st.write(" ") 
   st.write("Click the button below to train the selected " + messagePart + ".")

   if st.button(label = "Train " + messagePart, on_click = setStage, args = [11]): 
      
      trainingSucceeded = []
      trainingFailed = []
      for option in trueOptions:
          
          if option == model1:
             model = DecisionTreeClassifier(random_state = 1, **paramsValues[option])
          elif option == model2:
               model = GaussianProcessClassifier(random_state = 1, **paramsValues[option])
          elif option == model3:  
               model = KNeighborsClassifier(**paramsValues[option])  
          elif option == model4:
               model = LogisticRegression(random_state = 1, **paramsValues[option])
          elif option == model5:
               model = RandomForestClassifier(random_state = 1, **paramsValues[option])
          elif option == model6:
               model = SVC(random_state = 1, **paramsValues[option])
          
          try:
              
              model.fit(xTrain, yTrain)
              trainingSucceeded.append(option)
              st.session_state.models[option] = model
              
          except:
              trainingFailed.append(option)
      
      st.session_state.trainingSucceeded = trainingSucceeded
      st.session_state.trainingFailed = trainingFailed
             
#--------------------------------
# Display or upload the test set.
#--------------------------------  

if st.session_state.stage >= 11:
    
   trainingSucceeded = st.session_state.trainingSucceeded
   trainingFailed = st.session_state.trainingFailed
   
   ts = len(trainingSucceeded)
   tf = len(trainingFailed)
   
   if datasetAvailability in [case1, case3]:
       
       messagePart = "Click the button below to display the randomly formed test set."
       printTrainingResults(trainingSucceeded = trainingSucceeded, 
                            trainingFailed = trainingFailed,
                            messagePart = messagePart) 
       
       if ts == 0:
          setStage(11)
       else:
                    
          st.button(label = "Display test set", on_click = setStage, args = [12])
       
          if st.session_state.stage >= 12:
             displayData(data = testSet, header = "**Test Set**")
             st.write("Test Set = ", testSet.shape)
       
   else:
      
      messagePart = "Upload the test set at the sidebar."
      printTrainingResults(trainingSucceeded = trainingSucceeded, 
                           trainingFailed = trainingFailed,
                           messagePart = messagePart) 
   
      uploadedTestSet = st.sidebar.file_uploader(label = "Upload the test set in `csv` file format.",
                                                 on_change = setStage, 
                                                 args = [12])   
   
      if uploadedTestSet is not None:
    
         testSet = pd.read_csv(uploadedTestSet)
         displayData(data = testSet, header = "**Test Set**")
      
      else:
         setStage(11)
      
#--------------------------------
# Click to display the ROC curve.
#--------------------------------

if st.session_state.stage >= 12:
    
   if ts == 1:
      
      if tf > 0: 
         messagePart1 = "successfully trained model"
      else:
         messagePart1 = "model"
         
      messagePart2 = "curve"
       
   else:
      
      if tf > 0:
         messagePart1 = "successfully trained models"
      else:
         messagePart1 = "models"
         
      messagePart2 = "curves"
   
   st.write("Click the button below to display the ROC " + messagePart2 + " of the " + messagePart1 + " on the test set.")
     
   st.button(label = "Display the ROC " + messagePart2, on_click = setStage, args = [13])
      
   st.write(" ")
   st.write(" ")

#-----------------------
# Display the ROC curve.
#-----------------------

if st.session_state.stage >= 13:
   
   try:
       
       xTest = colTransformer.transform(testSet[features])
       
       if nUniqueValues > 2: 
          yTest = testSet["binarized target"].to_numpy() 
       else:    
          yTest = testSet[targetVariable].to_numpy()
        
       n = 0
    
       if ts == 1:
    
          plt.rcParams["axes.linewidth"] = 0.3
          plt.rc('legend', fontsize = 2.7)
          model = st.session_state.models[trainingSucceeded[0]]
          RocCurveDisplay.from_estimator(model, xTest, yTest, lw = 0.3)
          fig = plt.gcf()
          fig.set_size_inches(3, 2)
          ax = plt.gca()
          ax.set_aspect("equal")
          ax.xaxis.set_tick_params(length = 1.5, width = 0.3, labelsize = 2.7)
          ax.yaxis.set_tick_params(length = 1.5, width = 0.3, labelsize = 2.7)
          ax.xaxis.label.set_size(3.2)
          ax.yaxis.label.set_size(3.2)
         
       else:
          
          remainder = ts % 2
      
          if remainder == 0:
             nRows = ts//2
          else:
             nRows = ts//2 + 1
       
          if nRows == 1:
             figsize = (10, 5)
          elif nRows == 2:
               figsize = (9, 9)
          else:
               figsize = (9, 14)
      
          fig, axes = plt.subplots(nRows, 2, sharey = True, figsize = figsize)
          plt.rc('legend', fontsize = 9)
      
          rocAucScores = []
         
          for i in range(ts):
             
              if nRows == 1:
                 j = i
              else:
                 j = (i//2, i % 2)
            
              model = st.session_state.models[trainingSucceeded[i]]
              RocCurveDisplay.from_estimator(model, xTest, yTest, ax = axes[j], lw = 1.2)
              axes[j].set_aspect("equal")
              axes[j].set_title(trainingSucceeded[i], fontsize = 12)
              axes[j].xaxis.set_tick_params(labelsize = 9)
              axes[j].yaxis.set_tick_params(labelsize = 9)
              axes[j].xaxis.label.set_size(10)
              
              if i % 2 == 0:
                 axes[j].yaxis.label.set_size(10)
              else:
                 axes[j].set_ylabel("")
             
              rocAucScore = np.round(roc_auc_score(yTest, model.predict_proba(xTest)[:, 1]), 2)
              rocAucScores.append(rocAucScore)
                
          if remainder == 1:
             fig.delaxes(axes[ts//2, 1])
   
          fig.tight_layout()  
   
          if nRows == 2:
             plt.subplots_adjust(wspace = 0.05, hspace = 0.3)
          elif nRows > 2:
               plt.subplots_adjust(wspace = 0.1, hspace = 0.3)
                
          indicesMaxRocAucScore = np.argwhere(rocAucScores == np.amax(rocAucScores)).flatten().tolist()
          n = len(indicesMaxRocAucScore)
             
       st.pyplot(fig)
       
       if ts > 1:
           
          st.write(" ")
          st.write(" ")
          
          if tf > 0:
             messagePart = " successfully trained "
          else:
             messagePart = " selected "
             
          if n == 1 and ts == 2:
             
             betterModel = trainingSucceeded[indicesMaxRocAucScore[0]]
             otherModel = list(set(trainingSucceeded).difference({betterModel}))[0]
             st.write(betterModel + " is better than " + otherModel + " on the basis of their ROC AUC \
                      scores on the test set.")
                      
          elif n == 1 and ts >= 3:
               st.write("The best model among the " + str(ts) + messagePart + "models on the basis of \
                         their ROC AUC scores on the test set is " + trainingSucceeded[indicesMaxRocAucScore[0]] + ".")  
          elif n == 2 and ts >= 3:
               st.write("The best models among the " + str(ts) + messagePart + "models on the basis of \
                         their ROC AUC scores on the test set are " + trainingSucceeded[indicesMaxRocAucScore[0]] + \
                         " and " + trainingSucceeded[indicesMaxRocAucScore[1]] + ".")
          elif n >= 3 and ts >=3 and n != ts:
              
               st.write("The best models among the " + str(ts) + messagePart + "models on the basis of \
                         their ROC AUC scores on the test set are:")
                       
               for i in range(n):
                   st.write("(" + str(i + 1) + ") " + trainingSucceeded[indicesMaxRocAucScore[i]])
                       
          elif n == ts:
               st.write("The" + messagePart + "models have the same ROC AUC score on the test set.")
   
       toSave = True            
   
   except:
       st.markdown(":red[An error occured in transforming one or more columns of the test set. The error may be \
                    avoided by increasing the train-test split ratio.]")
       
#------------
# Save model.
#------------

if toSave:
   
   st.write(" ")
   st.write(" ") 
   
   if datasetAvailability in [case1, case3]:
      fullDataset = dataset.copy()
   else:
      fullDataset = pd.concat([trainSet, testSet], ignore_index = True)
   
   colTransformer.fit(fullDataset[features])     
   x = colTransformer.transform(fullDataset[features])
   
   if nUniqueValues > 2: 
      y = fullDataset["binarized target"].to_numpy()
   else:
      y = fullDataset[targetVariable].to_numpy()
   
   if ts == 1 or n == 1:
       
      if n == 1 and ts == 2:
         messagePart = " the better "
      elif n == 1 and ts >= 3:
           messagePart = " the best "
      else:
           messagePart = " "
      
      st.write("Click the button below to retrain" + messagePart + "model using the full \
                dataset (the union of the training and test sets) and save the retrained \
                model.")
    
      if ts == 1:
         modelName = trainingSucceeded[0]
      elif n == 1:
           modelName = trainingSucceeded[indicesMaxRocAucScore[0]]
           
      model = st.session_state.models[modelName]
    
      modelNameParts = modelName.split() 
      filename = modelNameParts[0].lower()
   
      for i in range(1, len(modelNameParts)):
          filename += modelNameParts[i]
   
      filename += ".joblib"

      if st.button(label = "Retrain and save" + messagePart + "model", on_click = setStage, args = [14]):
                 
         model.fit(x, y)
         dump(model, filename) 
         st.write("Model saved as a joblib file in " + os.getcwd() + ". Click the `Start Over` or \
                   `Reset` buttons to go back to the intro page or close the browser's tab displaying \
                   this web app to exit.")
         
   else:
      
      if n == ts and tf == 0:
         messagePart = " "
      elif n == ts and tf > 0:
           messagePart = " successfully trained "
      else:
         messagePart = " best " 
       
      st.write("Click the button below to retrain the" + messagePart + "models using the full \
                dataset (the union of the training and test sets) and save the retrained \
                models.")
   
      if st.button(label = "Retrain and save the" + messagePart + "models", on_click = setStage, args = [14]):
       
         for i in range(n):
          
             modelName = trainingSucceeded[indicesMaxRocAucScore[i]]
             model = st.session_state.models[modelName]
        
             modelNameParts = modelName.split() 
             filename = modelNameParts[0].lower()
       
             for j in range(1, len(modelNameParts)):
                 filename += modelNameParts[j]
       
             filename += ".joblib"
          
             model.fit(x, y)
             dump(model, filename) 
             
         st.write("Models saved as joblib files in " + os.getcwd() + ". Click the `Start Over` or \
                   `Reset` buttons to go back to the intro page or close the browser's tab displaying \
                   this web app to exit.")
          
#-----------
# Start over
#-----------

if st.session_state.stage >= 14:
   st.sidebar.button(label = 'Start Over', on_click = setStage, args = [0])
