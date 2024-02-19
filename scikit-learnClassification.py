"""
         File: scikit-learnClassification.py
 Date Created: February 6, 2024
Date Modified: February 20, 2024
----------------------------------------------------------------------------------------
Take the user through the steps to train and test classification models in scikit-learn.
----------------------------------------------------------------------------------------
"""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import lrParams
import matplotlib.pyplot as plt
import pandas as pd
import rfcParams
import streamlit as st

st.markdown("## Classification Using `scikit-learn`")

def setState(i):
    "Set the session state."
    
    st.session_state.stage = i

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
   st.button(label = 'Begin', on_click = setState, args = [1])
   
if st.session_state.stage >= 1:
   st.sidebar.button(label = "Reset", 
                     help = "Clicking this button at any time will bring you back to the intro page",
                     on_click = setState, 
                     args = [0])

#-------------------------
# Upload the training set.
#-------------------------

if st.session_state.stage >= 1:
    
   uploadedTrainSet = st.sidebar.file_uploader(label = "Upload the training set.",
                                               on_change = setState, 
                                               args = [2])
   
   if uploadedTrainSet is not None:
       
      if "trainSet" not in st.session_state:
          st.session_state.trainSet = ""
    
      trainSet = pd.read_csv(uploadedTrainSet)
      st.session_state.trainSet = trainSet
      st.markdown("**Training Set**")
      st.dataframe(trainSet, height = 215)
      
   else:
      setState(1)
      
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
                                        on_change = setState,
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
         setState(2)
         
      else:
         
         st.write("Click the button below to complete the selection of categorical features \
                   or if no features are categorical.")
         st.button(label = "Complete selection", on_click = setState, args = [4])
        
   else:
      setState(2)
   
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
   
   modelName = st.radio(label = "Select the model to train.",
                        options = ["Logistic Regression", "Random Forest"],
                        index = None,
                        on_change = setState, 
                        args = [5])

#-----------------------------------------------------------------
# Click to set the values of the parameters of the selected model.
#-----------------------------------------------------------------
    
if st.session_state.stage >= 5:
    
   if modelName == "Logistic Regression":
      st.write("You selected logistic regression. Click the button below to set \
                the values of the model parameters.")           
   elif modelName == "Random Forest":
        st.write("You selected random forest. Click the button below to set \
                  the values of the model parameters.")               
        
   st.button(label = "Set model parameters", on_click = setState, args = [6])

#--------------------------------------------------------
# Set the values of the parameters of the selected model.
#--------------------------------------------------------
    
if st.session_state.stage >= 6:
    
   if modelName == "Logistic Regression":
      lrParamsValues = lrParams.setLRParams()
   elif modelName == "Random Forest":       
        rfcParamsValues = rfcParams.setRFCParams()
              
   st.write("Click the `Reset model parameters` button to reset the model parameters \
             to its default values or click the `Confirm` button to confirm the \
             assigned values of the model parameters.")
   st.button(label = "Reset model parameters", on_click = setState, args = [5])          
   st.button(label = "Confirm model parameters", on_click = setState, args = [7])

#-------------
# Train model.
#-------------

if st.session_state.stage >= 7:
    
   if "model" not in st.session_state:
      st.session_state.model = ""

   st.write(" ")
   st.write("Click the button below to train the selected model.")
    
   if modelName == "Logistic Regression":
        
      if st.button(label = "Train model", on_click = setState, args = [8]):
           
         model = LogisticRegression(random_state = 1, **lrParamsValues)
         model.fit(xTrain, yTrain)
         st.session_state.model = model
          
   elif modelName == "Random Forest":
            
        if st.button(label = "Train model", on_click = setState, args = [8]):
             
           model = RandomForestClassifier(random_state = 1, **rfcParamsValues)
           model.fit(xTrain, yTrain) 
           st.session_state.model = model

#---------------------
# Upload the test set.
#---------------------   

if st.session_state.stage >= 8:
    
   st.write("Training completed. Upload the test set at the sidebar.")
   
   uploadedTestSet = st.sidebar.file_uploader(label = "Upload the test set.",
                                              on_change = setState, 
                                              args = [9])   
   
   if uploadedTestSet is not None:
    
      testSet = pd.read_csv(uploadedTestSet)
      st.markdown("**Test Set**")
      st.dataframe(testSet, height = 215)
      
   else:
      setState(8)

#-----------------------
# Display the ROC curve.
#-----------------------

if st.session_state.stage >= 9:
   
   st.write("Click the button below to display the ROC curve of the model on the test set.")
     
   if st.button(label = "Display the ROC curve", on_click = setState, args = [10]):
      
      xTest = colTransformer.transform(testSet[features])
      yTest = testSet["target"].to_numpy()
       
      fig, ax = plt.subplots(figsize = (3, 2))
      plt.rc('legend', fontsize = 4)
      model = st.session_state.model
      RocCurveDisplay.from_estimator(model, xTest, yTest, ax = ax, lw = 0.7)
      ax.set_title("ROC Curve", fontsize = 4.5)
      ax.xaxis.set_tick_params(labelsize = 3)
      ax.yaxis.set_tick_params(labelsize = 3)
      ax.xaxis.label.set_size(4)
      ax.yaxis.label.set_size(4)
      
      st.pyplot(fig)

#-----------
# Start over
#-----------

if st.session_state.stage >= 10:
   st.sidebar.button(label = 'Start Over', on_click = setState, args = [0])



