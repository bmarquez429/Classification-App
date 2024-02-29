"""
         File: scikit-learnClassification.py
 Date Created: February 6, 2024
Date Modified: February 29, 2024
----------------------------------------------------------------------------------------
Take the user through the steps to train and test classification models in scikit-learn.
----------------------------------------------------------------------------------------
"""

from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay
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

st.markdown("## Classification Using `scikit-learn`")

def setStage(i):
    "Change the value of the stage key."
    
    st.session_state.stage = i
    
def setAllOptions():
    "Change the value of the allOptions key."
    
    condition = st.session_state[model1] and \
                st.session_state[model2] and \
                st.session_state[model3] and \
                st.session_state[model4] and \
                st.session_state[model5] and \
                st.session_state[model6]     

    if condition:
       st.session_state.allOptions = True
    else:
       st.session_state.allOptions = False
       
    setStage(4)
        
    return

def setOptions():
    "Change the values of the model keys."

    if st.session_state.allOptions:
        
       st.session_state[model1] = True
       st.session_state[model2] = True
       st.session_state[model3] = True
       st.session_state[model4] = True
       st.session_state[model5] = True
       st.session_state[model6] = True
       
    else:
         
       st.session_state[model1] = False
       st.session_state[model2] = False
       st.session_state[model3] = False
       st.session_state[model4] = False
       st.session_state[model5] = False
       st.session_state[model6] = False
       
    setStage(4)
        
    return

#------
# Begin
#------

if 'stage' not in st.session_state:
    st.session_state.stage = 0

if st.session_state.stage == 0:
    
   st.markdown("This web app will walk you through the steps in training a \
                classifier using a selection of algorithms that are implemented \
                in `scikit-learn`.")
   st.markdown("Click the button below to begin.")
   st.button(label = 'Begin', on_click = setStage, args = [1])
   
if st.session_state.stage >= 1:
   st.sidebar.button(label = "Reset", 
                     help = "Clicking this button at any time will bring you back to the intro page",
                     on_click = setStage, 
                     args = [0])

#-------------------------
# Upload the training set.
#-------------------------

if st.session_state.stage >= 1:
    
   uploadedTrainSet = st.sidebar.file_uploader(label = "Upload the training set.",
                                               on_change = setStage, 
                                               args = [2])
   
   if uploadedTrainSet is not None:
       
      if "trainSet" not in st.session_state:
          st.session_state.trainSet = ""
    
      trainSet = pd.read_csv(uploadedTrainSet)
      st.session_state.trainSet = trainSet
      st.markdown("**Training Set**")
      st.dataframe(trainSet, height = 215)
      
   else:
      setStage(1)
      
#-----------------------------------------------------------------------------
# Select features that are categorical or that will be treated as categorical.
#-----------------------------------------------------------------------------

if st.session_state.stage >= 2:
    
   trainSet = st.session_state.trainSet
   features = trainSet.columns.tolist()
   features.remove("target")
   
   label1 = "Select features that are categorical or that will be treated as categorical. "
   label2 = "Features that are not selected are continuous or will be treated as continuous. "
   label3 = "Select `None` if no features are categorical."
   categoricalFeatures = st.multiselect(label = label1 + label2 + label3,
                                        options = features + ["None"],
                                        on_change = setStage,
                                        args = [3],
                                        placeholder = "Select feature(s) or select None")

#------------------------------------------------
# Complete the selection of categorical features.
#------------------------------------------------

if st.session_state.stage >= 3:
    
   if len(categoricalFeatures) > 0:
      
      if len(categoricalFeatures) >= 2 and "None" in categoricalFeatures:
          
         st.write("You cannot select None together with one or more features. Please edit \
                   your selection.")
         setStage(2)
         
      else:
         
         st.write("Click the button below to complete the selection of categorical features \
                   or if no features are categorical.")
         st.button(label = "Complete selection", key = "catFeaturesSelection", on_click = setStage, args = [4])
        
   else:
      setStage(2)
   
#-------------------------------------------------
# Transform columns and select the model to train.
#-------------------------------------------------

if st.session_state.stage >= 4: 

   # categoricalFeatures = ["sex", "cp", "fbs", "restecg", "exang", "thal"]
   
   if categoricalFeatures[0] == "None":
      colTransformer = ColumnTransformer(transformers = [("continuous", StandardScaler(), features)])
   else:
      colTransformer = ColumnTransformer(transformers = [("categorical", OneHotEncoder(drop = "first"), 
                                                          categoricalFeatures)],
                                         remainder = StandardScaler())
      
   colTransformer.fit(trainSet[features])
   
   xTrain = colTransformer.transform(trainSet[features])
   yTrain = trainSet["target"].to_numpy()
   
   st.write(" ")
   st.write("Select the model to train.")
   
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
   
   st.write("Click the button below to complete the selection of models.")
   st.button(label = "Complete selection", key = "modelSelection", on_click = setStage, args = [5])
   
#-----------------------------------------------------------------
# Click to set the values of the parameters of the selected model.
#-----------------------------------------------------------------
    
if st.session_state.stage >= 5:
    
    options = [model1, model2, model3, model4, model5, model6]
    trueOptions = []
    
    for option in options:
        
        if st.session_state[option] is True:
            trueOptions.append(option)
            
    n = len(trueOptions)
    
    st.write(" ")
            
    if n == 1:
       st.write("You selected " + trueOptions[0] + ". Click the button \
                 below to set the values of the model parameters.")
    elif n == 2:
         st.write("You selected " + trueOptions[0] + " and " + trueOptions[1] + ". \
                   Click the button below to set the values of the model parameters.") 
    elif n >= 3 and n != len(options):
        
         st.write("You selected the following models:")
         
         for i in range(n):
             st.write("(" + str(i + 1) + ") " + trueOptions[i])
             
         st.write("Click the button below to set the values of the model parameters.")
         
    elif n == len(options):
         st.write("You selected all the available models. Click the button below to set the values \
                   of the model parameters.")
             
    st.button(label = "Set model parameters", on_click = setStage, args = [6])

#--------------------------------------------------------
# Set the values of the parameters of the selected model.
#--------------------------------------------------------
    
if st.session_state.stage >= 6:
    
   st.write("")
    
   paramsValues = {}
    
   if n == 1:
      paramsValues[trueOptions[0]], disabled = modelParams.setModelParams(trueOptions[0])
   else:
      
      if n >= 5:
         st.write("To scroll through the tabs, click on a tab and use the left-arrow and right-arrow keys.")
       
      tabs = st.tabs([trueOptions[i] for i in range(n)])
    
      for i in range(n):
    
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
        st.button(label = "Reset model parameters", on_click = setStage, args = [5])   
        
   with cols[1]:
        st.button(label = "Confirm model parameters", on_click = setStage, args = [7], disabled = disabled)

#-------------
# Train model.
#-------------

if st.session_state.stage >= 7:
    
   if "models" not in st.session_state:
      st.session_state.models = {}
      
   if n == 1:
       
      stringPart1 = "model"
      stringPart2 = "curve"
      
   else:
       
      stringPart1 = "models"
      stringPart2 = "curves"
      
   st.write(" ") 
   st.write("Click the button below to train the selected " + stringPart1 + ".")

   if st.button(label = "Train " + stringPart1, on_click = setStage, args = [8]): 
          
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
                 
          model.fit(xTrain, yTrain)
          st.session_state.models[option] = model

#---------------------
# Upload the test set.
#---------------------   

if st.session_state.stage >= 8:
    
   st.write("Training completed. Upload the test set at the sidebar.")
   
   uploadedTestSet = st.sidebar.file_uploader(label = "Upload the test set.",
                                              on_change = setStage, 
                                              args = [9])   
   
   if uploadedTestSet is not None:
    
      testSet = pd.read_csv(uploadedTestSet)
      st.markdown("**Test Set**")
      st.dataframe(testSet, height = 215)
      
   else:
      setStage(8)

#--------------------------------
# Click to display the ROC curve.
#--------------------------------

if st.session_state.stage >= 9:
   
   st.write("Click the button below to display the ROC " + stringPart2 + " of the " + stringPart1 + " on the test set.")
     
   st.button(label = "Display the ROC " + stringPart2, on_click = setStage, args = [10])
      
   st.write(" ")
   st.write(" ")

#-----------------------
# Display the ROC curve.
#-----------------------

if st.session_state.stage >= 10:
      
   xTest = colTransformer.transform(testSet[features])
   yTest = testSet["target"].to_numpy()
   
   m = 0
    
   if n == 1:
    
      plt.rcParams["axes.linewidth"] = 0.3
      plt.rc('legend', fontsize = 2.7)
      model = st.session_state.models[trueOptions[0]]
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
          
      remainder = n % 2
      
      if remainder == 0:
         nRows = n//2
      else:
         nRows = n//2 + 1
       
      if nRows == 1:
         figsize = (10, 5)
      elif nRows == 2:
           figsize = (9, 9)
      else:
           figsize = (9, 14)
      
      fig, axes = plt.subplots(nRows, 2, sharey = True, figsize = figsize)
      plt.rc('legend', fontsize = 9)
      
      rocAucScores = []
         
      for i in range(n):
             
          if nRows == 1:
             j = i
          else:
             j = (i//2, i % 2)
            
          model = st.session_state.models[trueOptions[i]]
          RocCurveDisplay.from_estimator(model, xTest, yTest, ax = axes[j], lw = 1.2)
          axes[j].set_aspect("equal")
          axes[j].set_title(trueOptions[i], fontsize = 12)
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
         fig.delaxes(axes[n//2, 1])
   
      fig.tight_layout()  
   
      if nRows == 2:
         plt.subplots_adjust(wspace = 0.05, hspace = 0.3)
      elif nRows > 2:
           plt.subplots_adjust(wspace = 0.1, hspace = 0.3)
                
      indicesMaxRocAucScore = np.argwhere(rocAucScores == np.amax(rocAucScores)).flatten().tolist()
      m = len(indicesMaxRocAucScore)
             
   st.pyplot(fig)
   
   if n > 1:
       
      st.write(" ")
      st.write(" ")
         
      if m == 1 and n == 2:
         
         betterModel = trueOptions[indicesMaxRocAucScore[0]]
         otherModel = list(set(trueOptions).difference({betterModel}))[0]
         st.write(betterModel + " is better than " + otherModel + " on the basis of their ROC AUC \
                  scores on the test set.")
                  
      elif m == 1 and n >= 3:
           st.write("The best model among the " + str(n) + " selected trained models on the basis of \
                     their ROC AUC scores on the test set is " + trueOptions[indicesMaxRocAucScore[0]] + ".")            
      elif m == 2 and n >= 3:
           st.write("The best models among the " + str(n) + " selected trained models on the basis of \
                     their ROC AUC scores on the test set are " + trueOptions[indicesMaxRocAucScore[0]] + \
                     " and " + trueOptions[indicesMaxRocAucScore[1]] + ".")
      elif m >= 3 and n >=3 and m != n:
          
           st.write("The best models among the " + str(n) + " selected trained models on the basis of \
                     their ROC AUC scores on the test set are:")
                   
           for i in range(m):
               st.write("(" + str(i + 1) + ") " + trueOptions[indicesMaxRocAucScore[i]])
                   
      elif m == n:
           st.write("The selected trained models have the same ROC AUC score on the test set.")
       
   setStage(11)
      
#------------
# Save model.
#------------

if st.session_state.stage >= 11:
   
   st.write(" ")
   st.write(" ") 
   
   if n == 1 or m == 1:
       
      if m == 1 and n == 2:
         stringPart = " the better "
      elif m == 1 and n >= 3:
           stringPart = " the best "
      else:
           stringPart = " "
      
      st.write("Click the button below to save" + stringPart + "model.")
    
      if n == 1:
         modelName = trueOptions[0]
      elif m == 1:
           modelName = trueOptions[indicesMaxRocAucScore[0]]
           
      model = st.session_state.models[modelName]
    
      modelNameParts = modelName.split() 
      filename = modelNameParts[0].lower()
   
      for i in range(1, len(modelNameParts)):
          filename += modelNameParts[i]
   
      filename += ".joblib"

      if st.button(label = "Save" + stringPart + "model"):
       
         dump(model, filename) 
         st.write("Model saved as a joblib file in " + os.getcwd())
         
   else:
      
      if m == n:
         stringPart = " "
      else:
         stringPart = " best " 
       
      st.write("Click the button below to save the" + stringPart + "models.")
   
      if st.button(label = "Save the" + stringPart + "models"):
       
         for i in range(m):
          
             modelName = trueOptions[indicesMaxRocAucScore[i]]
          
             model = st.session_state.models[modelName]
        
             modelNameParts = modelName.split() 
             filename = modelNameParts[0].lower()
       
             for j in range(1, len(modelNameParts)):
                 filename += modelNameParts[j]
       
             filename += ".joblib"
          
             dump(model, filename) 
             
         st.write("Models saved as joblib files in " + os.getcwd())
         
   setStage(12)
          
#-----------
# Start over
#-----------

if st.session_state.stage >= 12:
   st.sidebar.button(label = 'Start Over', on_click = setStage, args = [0])
