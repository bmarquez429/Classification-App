"""
         File: helperFunctions.py
 Date Created: March 4, 2024
Date Modified: March 9, 2024
------------------------------------------------------------------------------------------------------
The functions defined in this script are imported by modelParams.py and scikit-learnClassification.py.
------------------------------------------------------------------------------------------------------
"""

from sklearn.preprocessing import Binarizer
import streamlit as st

model1 = "Decision Tree Classifier"
model2 = "Gaussian Process Classifier"
model3 = "k-Nearest Neighbors Classifier"
model4 = "Logistic Regression"
model5 = "Random Forest Classifier"
model6 = "Support Vector Classifier"

def binarizeTarget(dataset, classes, targetVariable, variableType):
    '''Binarize a target variable.'''
    
    if variableType == "categorical":
        
       positiveClasses = st.multiselect(label = "Select the classes that will be map to 1.",
                                        options = classes,
                                        on_change = setStage,
                                        args = [3],
                                        placeholder = "Select class(es)")
    
       nUniqueValues = len(classes)
  
       if len(positiveClasses) > 0:
     
          if len(positiveClasses) == nUniqueValues:
         
             st.markdown(":red[You cannot select all the classes since at least one class should be \
                          map to 0. Please edit your selection.]")
             setStage(3)
        
          else:
        
             st.write("Click the button below to complete the selection of the classes that will be \
                       map to 1.")         
               
             st.button(label = "Complete selection", key = "positiveClassesSelection", on_click = setStage, 
                       args = [5])
        
             dataset["binarized target"] = dataset[targetVariable].apply(lambda x: 1 if x in positiveClasses else 0)

       else:
          setStage(3)
          
    elif variableType == "continuous": 
        
         threshold = st.number_input(label = "Input a threshold.",
                                     value = 0.0,
                                     step = 1.0,
                                     format = "%.1f",
                                     on_change = setStage,
                                     args = [3])
        
         st.write("Click the button below to complete the mapping of values to 1 and 0 according to the selected \
                   threshold.")         
           
         st.button(label = "Complete mapping", key = "positiveClassesMapping", on_click = setStage, 
                   args = [5])
         
        
         y = dataset[[targetVariable]]
         binarizer = Binarizer(threshold = threshold)
         binarizer.fit(y)
         dataset["binarized target"] = binarizer.transform(y).astype(int)
         
         
         
    
def displayData(data, header, displayInstancesCount = True):
    '''Display data.'''
    
    st.markdown(header)
    st.dataframe(data, height = 215)
    
    if displayInstancesCount:
       st.write("Number of Instances = ", data.shape[0])
    
def printTrainingResults(trainingSucceeded, trainingFailed, messagePart):
    '''Print a message regarding the result(s) of model training.'''
    
    p = len(trainingSucceeded)
    q = len(trainingFailed)
    
    if q == 1:
        
       messagePart1 = "Training of " + trainingFailed[0] + " has failed."
       messagePart2 = "You may try changing some values of its parameters before training it again."
       
    elif q == 2:
        
         messagePart1 = "Training of " + trainingFailed[0] + " and " + trainingFailed[1] + \
                       " have failed."
         messagePart2 = "You may try changing some values of their parameters before training them again."
                       
                       
    elif q >= 3:
        
         messagePart1 = "Training of the following models have failed."
         messagePart2 = "You may try changing some values of their parameters before training them again."
         
    if p == 1:
       messagePart3 = "On the other hand, training of " + trainingSucceeded[0] + " has succeeded."
    elif p == 2:
         messagePart3 = "On the other hand, training of " + trainingSucceeded[0] + " and " + \
                         trainingSucceeded[1] + " have succeeded."
    elif p >= 3:
         messagePart3 = "On the other hand, training of the following models have succeeded."
         
    messagePart4 = messagePart
    
    if q == 0:
       st.write("Training completed. " + messagePart4)
    elif p == 0:
        
         if q <= 2:
            st.write(messagePart1 + " " + messagePart2 + " You may also try a different model to train.")
         elif q >= 3 and q < 6: 
            st.write("Training of the selected models have failed. " + messagePart2 + " You may also try a \
                      different model to train.")
         else:
            st.write("Training of the selected models have failed. " + messagePart2) 
         
    else:
    
        if q < 3 and p < 3:
           st.write(messagePart1 + " " + messagePart2 + " " + messagePart3 + " " + messagePart4)
        elif q < 3 and p >= 3:
         
           st.write(messagePart1 + " " + messagePart2 + " " + messagePart3)
         
           for i in range(p):
               st.write("(" + str(i + 1) + ") " + trainingSucceeded[i])
             
           st.write(messagePart4)
         
        elif q >= 3 and p < 3:
        
             st.write(messagePart1)
       
             for i in range(q):
                 st.write("(" + str(i + 1) + ") " + trainingFailed[i])
           
             st.write(messagePart2 + " " + messagePart3 + " " + messagePart4)
         
        else:
        
             st.write(messagePart1)
       
             for i in range(q):
                 st.write("(" + str(i + 1) + ") " + trainingFailed[i])
             
             st.write(messagePart2 + " " + messagePart3)
         
             for i in range(p):
                 st.write("(" + str(i + 1) + ") " + trainingSucceeded[i])
             
             st.write(messagePart4)

def setAllOptions():
    '''Change the value of the allOptions key.'''
    
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
       
    setStage(8)

def setOptions():
    '''Change the values of the model keys.'''

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
       
    setStage(8)
    
def setStage(i):
    '''Change the value of the stage key.'''
    
    st.session_state.stage = i
    
    if i == 0 or i == 6:
       
       if "trainTestSplit" in st.session_state:
          del st.session_state["trainTestSplit"]
