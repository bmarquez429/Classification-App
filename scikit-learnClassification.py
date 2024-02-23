"""
         File: scikit-learnClassification.py
 Date Created: February 6, 2024
Date Modified: February 23, 2024
----------------------------------------------------------------------------------------
Take the user through the steps to train and test classification models in scikit-learn.
----------------------------------------------------------------------------------------
"""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
import modelParams
import pandas as pd
import streamlit as st

st.markdown("## Classification Using `scikit-learn`")

def setStage(i):
    "Change the value of the stage key."
    
    st.session_state.stage = i
    
def setAllOptions():
    "Change the value of the allOptions key."
    
    condition = st.session_state[option1] and \
                st.session_state[option2] and \
                st.session_state[option3]

    if condition:
       st.session_state.allOptions = True
    else:
       st.session_state.allOptions = False
       
    setStage(4)
        
    return

def setOptions():
    "Change the values of the option1 and option2 keys."

    if st.session_state.allOptions:
        
       st.session_state[option1] = True
       st.session_state[option2] = True
       st.session_state[option3] = True
       
    else:
         
       st.session_state[option1] = False
       st.session_state[option2] = False
       st.session_state[option3] = False
       
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
   
   option1 = "Logistic Regression"
   option2 = "k-Nearest Neighbors Classifier"
   option3 = "Random Forest Classifier"
   
   st.checkbox(label = option1, key = option1, on_change = setAllOptions)
   st.checkbox(label = option2, key = option2, on_change = setAllOptions)
   st.checkbox(label = option3, key = option3, on_change = setAllOptions)
   st.checkbox("Select all models", key = 'allOptions', on_change = setOptions)
   
   st.write("Click the button below to complete the selection of models.")
   st.button(label = "Complete selection", key = "modelSelection", on_click = setStage, args = [5])
   
#-----------------------------------------------------------------
# Click to set the values of the parameters of the selected model.
#-----------------------------------------------------------------
    
if st.session_state.stage >= 5:
    
    options = [option1, option2, option3]
    trueOptions = []
    
    for option in options:
        
        if st.session_state[option] is True:
            trueOptions.append(option)
            
    n = len(trueOptions)
            
    if n == 1:
       st.write("You selected " + trueOptions[0].lower() + ". Click the button \
                 below to set the values of the model parameters.")
    elif n == 2:
         st.write("You selected " + trueOptions[0].lower() + " and " + trueOptions[1].lower() + ". \
                   Click the button below to set the values of the model parameters.") 
    else:
         st.write("You selected all the models. Click the button below to set the values \
                   of the model parameters.")
             
    st.button(label = "Set model parameters", on_click = setStage, args = [6])

#--------------------------------------------------------
# Set the values of the parameters of the selected model.
#--------------------------------------------------------
    
if st.session_state.stage >= 6:
    
   paramsValues = {}
    
   if n == 1:
      paramsValues[trueOptions[0]] = modelParams.setModelParams(trueOptions[0])
   else:
       
        tabs = st.tabs([trueOptions[i] for i in range(n)])
        
        for i in range(n):
        
            with tabs[i]:
            
                 paramsValues_i = modelParams.setModelParams(trueOptions[i])
            
            paramsValues[trueOptions[i]] = paramsValues_i
       
   st.write("Click the `Reset model parameters` button to set the model parameters \
             again or click the `Confirm` button to confirm the \
             assigned values of the model parameters.")
   st.button(label = "Reset model parameters", on_click = setStage, args = [5])          
   st.button(label = "Confirm model parameters", on_click = setStage, args = [7])

#-------------
# Train model.
#-------------

if st.session_state.stage >= 7:
    
   if "models" not in st.session_state:
      st.session_state.models = {}
      
   if n == 1:
      modelString = "model"
   else:
      modelString = "models"
      
   st.write(" ") 
   st.write("Click the button below to train the selected " + modelString + ".")

   if st.button(label = "Train " + modelString, on_click = setStage, args = [8]): 
          
      for option in trueOptions:
          
          if option == option1:  
             model = LogisticRegression(random_state = 1, **paramsValues[option])   
          elif option == option2:
               model = KNeighborsClassifier(**paramsValues[option])   
          elif option == option3:
               model = RandomForestClassifier(random_state = 1, **paramsValues[option])
                 
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

#-----------------------
# Display the ROC curve.
#-----------------------

if st.session_state.stage >= 9:
   
   st.write("Click the button below to display the ROC curve of the model on the test set.")
     
   if st.button(label = "Display the ROC curve", on_click = setStage, args = [10]):
      
      xTest = colTransformer.transform(testSet[features])
      yTest = testSet["target"].to_numpy()
      
      st.write(" ")
      st.write(" ")
      
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
         else:
            figsize = (9, 9)
           
         fig, axes = plt.subplots(nRows, 2, sharey = True, figsize = figsize)
         plt.rc('legend', fontsize = 9)
         
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
                
         if remainder == 1:
            fig.delaxes(axes[n//2, 1])
         
         if nRows > 1:
             
            fig.tight_layout()
            plt.subplots_adjust(wspace = 0.05, hspace = 0.3)

      st.pyplot(fig)

#-----------
# Start over
#-----------

if st.session_state.stage >= 10:
   st.sidebar.button(label = 'Start Over', on_click = setStage, args = [0])
