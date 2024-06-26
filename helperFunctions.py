"""
         File: helperFunctions.py
 Date Created: March 4, 2024
Date Modified: May 8, 2024
------------------------------------------------------------------------------------------------------
The functions defined in this script are imported by modelParams.py and scikit-learnClassification.py.
------------------------------------------------------------------------------------------------------
"""

from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import Binarizer
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

model1 = "Decision Tree Classifier"
model2 = "Gaussian Naive Bayes Classifier"
model3 = "Gaussian Process Classifier"
model4 = "k-Nearest Neighbors Classifier"
model5 = "Logistic Regression"
model6 = "Multi-layer Perceptron Classifier"
model7 = "Quadratic Discriminant Analysis"
model8 = "Random Forest Classifier"
model9 = "Support Vector Classifier"

def actOnClassImbalance(classDistribution, nUniqueValues, trainSet, features, targetVariable, currentStage,
                        nextStage):
    '''Select on how to handle an imbalanced class distribution.'''
    
    if classDistribution[0] < 0.4 or classDistribution[1] < 0.4:
       
       st.write(" ")
       st.write(" ")
       label = "Please select on how to handle the imbalanced class distribution." 
       bOption1 = "Balance out the class distribution. (Recommended)"
       bOption2 = "Retain the class distribution."
       balancing = st.radio(label = label, 
                            options = [bOption1, bOption2], 
                            index = 0,
                            on_change = selectBalancingOption,
                            args = [currentStage])
       
       st.button(label = "Next", key = "balancingNext", on_click = selectBalancingMethod, 
                 args = [balancing, currentStage, nextStage])
    
       if st.session_state.selectBalancingMethod:
               
          balancingMethods = ["Random Under-sampling", "Random Over-sampling", "SMOTE", "ADASYN"] 
          balancingMethod = st.radio(label = "Please select the method to balance out the class distribution.",
                                     options = balancingMethods,
                                     index = None,
                                     on_change = selectBalancingMethod,
                                     args = [balancing, currentStage, nextStage, True])
         
          if balancingMethod is not None:
             
             st.write("Click the button below to run the selected balancing method to the training set.")      
             st.button(label = "Run balancing method", 
                       on_click = runBalancingMethod,
                       args = [balancingMethod, currentStage])
       
       if st.session_state.runBalancingMethod:
       
          if balancingMethod == balancingMethods[0]: 
             method = RandomUnderSampler()
          elif balancingMethod == balancingMethods[1]:
               method = RandomOverSampler()
          elif balancingMethod == balancingMethods[2]:
               method = SMOTE()
          elif balancingMethod == balancingMethods[3]:
               method = ADASYN()   
               
          st.session_state.method = method
          
          try:
              
             if nUniqueValues > 2:
                xTrainResampled, yTrainResampled = method.fit_resample(trainSet[features + [targetVariable]], 
                                                                       trainSet[["binarized " + targetVariable]]) 
             else:
                xTrainResampled, yTrainResampled = method.fit_resample(trainSet[features], 
                                                                       trainSet[[targetVariable]])
                
             if "resampling" not in st.session_state:
                        
                st.session_state.resampling = {}
                st.session_state.resampling["xTrainResampled"] = xTrainResampled
                st.session_state.resampling["yTrainResampled"] = yTrainResampled   
                    
             xTrainResampled = st.session_state.resampling["xTrainResampled"]
             yTrainResampled = st.session_state.resampling["yTrainResampled"]
                 
             trainSet = pd.concat([xTrainResampled, yTrainResampled], axis = 1)
             
             st.write(" ")
             displayDataset(dataset = trainSet, header = "**Balanced Training Set**")
             classDistribution = displayClassDistribution(datasetHolder = trainSet, 
                                                          targetVariable = targetVariable, 
                                                          nUniqueValues = nUniqueValues,
                                                          titlePart = " Balanced ")
               
             st.button(label = "Next", key = "balancingMethodNext", on_click = setStage, args = [nextStage])
          
          except:
              
             st.markdown(":red[The selected balancing method did not run successfully. Please select a different \
                         balancing method.]") 
                     
       if balancing == bOption1:
            
          method = st.session_state.method
          output = (trainSet, method)
            
       else:
          output = trainSet
       
    else:
        
       st.button(label = "Next", key = "classDistributionNext", on_click = setStage, args = [nextStage]) 
       output = trainSet
           
    return output
       
def binarizeTarget(dataset, classes, targetVariable, variableType, currentStage, nextStage, positiveClassesKey = 0):
    '''Binarize a target variable.'''
    
    if variableType == "categorical":
       
       label = "`" + targetVariable + "` has " + str(len(classes)) + " classes. Select the classes that will be map to 1."
       positiveClasses = st.multiselect(label = label,
                                        options = classes,
                                        default = None, 
                                        key = positiveClassesKey,
                                        on_change = setStage,
                                        args = [currentStage],
                                        placeholder = "Select class(es)")
    
       nUniqueValues = len(classes)
  
       if len(positiveClasses) > 0:
     
          if len(positiveClasses) == nUniqueValues:
         
             st.markdown(":red[You cannot select all the classes since at least one class should be \
                          map to 0. Please edit your selection.]")
             setStage(currentStage)
        
          else:
        
             st.write("Click the button below to complete the selection of the classes that will be \
                       map to 1.")         
               
             st.button(label = "Complete selection", key = "positiveClassesSelection", on_click = setStage, 
                       args = [nextStage])
        
             dataset["binarized " + targetVariable] = dataset[targetVariable].apply(lambda x: 1 if x in positiveClasses else 0)

       else:
          setStage(currentStage)
          
       return positiveClasses
          
    elif variableType == "continuous": 
        
         minValue = min(dataset[targetVariable])
         maxValue = max(dataset[targetVariable])
         medianValue = dataset[targetVariable].median()
         label = "Input a threshold. The input should strictly be between " + str(minValue) + " and " + str(maxValue) + \
                 " since these two numbers are respectively the minimum and maximum of `" + targetVariable + "`. The \
                  default threshold is the median of `" + targetVariable + "`."
        
         threshold = st.number_input(label = label,
                                     value = medianValue,
                                     step = 1.0,
                                     format = "%.1f",
                                     on_change = setStage,
                                     args = [currentStage])
         
         if threshold <= minValue or threshold >= maxValue:
            
            st.markdown(":red[Your input is not in the required range. Please input a number that is strictly between " + \
                         str(minValue) + " and " + str(maxValue) + ".]")
            setStage(currentStage) 
            
         else:
        
            st.write("Click the button below to complete the mapping of values to 1 and 0 according to the selected \
                      threshold.")         
           
            st.button(label = "Complete mapping", key = "positiveClassesMapping", on_click = setStage, 
                      args = [nextStage])
         
        
         y = dataset[[targetVariable]]
         binarizer = Binarizer(threshold = threshold)
         binarizer.fit(y)
         dataset["binarized " + targetVariable] = binarizer.transform(y).astype(int)
         
         return binarizer
         
def changeTargetVariable():
    '''Delete the confirmTargetVariable key.'''
    
    setStage(4)
    del st.session_state["confirmTargetVariable"]
    del st.session_state["toCategorical"]
    
def checkUploadedTestSet(testSet, features, targetVariable, nFeaturesToUse, nFeatures):
    '''Check the columns of the uploaded test set.'''
    
    setDifference = set(features + [targetVariable]).difference(set(testSet.columns.tolist()))
    listDifference = list(setDifference)
    
    if len(listDifference) == 0:
    
       if nFeaturesToUse < nFeatures:
          header = "**Test Set with Selected Features**"
       else:
          header = "**Test Set**"
       
       testSet = testSet[features + [targetVariable]]
       toDisplay = True
       
       return toDisplay, testSet, header
          
    else:
        
       if targetVariable in listDifference:
           
          messagePart1 = "The selected target variable is not in the test set."
          listDifference.remove(targetVariable)
          
       else:
          messagePart1 = ""
          
       listLength = len(listDifference)
       
       if listLength == 0:
          messagePart2 = ""
       elif listLength == 1:
            messagePart2 = "feature `" + listDifference[0] + "` is not in the test set."
       elif listLength == 2:
            messagePart2 = "features `" + listDifference[0] + "` and `" + listDifference[1] + \
                           "` are not in the test set."
       else:
           
           featuresString = ""
           for i in range(listLength):
               
               if i == listLength - 1:
                  featuresString += "and `" + listDifference[i] + "`"
               else:
                  featuresString += "`" + listDifference[i] + "`, "
                  
           messagePart2 = "features " + featuresString + " are not in the test set." 
       
       if messagePart1 == "":
          messagePart = "The " + messagePart2
       else:
           
          if messagePart2 == "":
             messagePart = messagePart1
          else:
             messagePart = messagePart = messagePart1 + " Furthermore, the " + messagePart2
          
       st.markdown(":red[" + messagePart + " Please upload a valid test set.]")
       toDisplay = False
       setStage(16)
                   
       return toDisplay
    
def confirmTargetVariable(targetVariable = None):
    '''Confirm a selected target variable.'''
    
    st.session_state.stage = 3
    st.session_state.targetVariable = targetVariable 
    st.session_state.toTargetVariable = True
    
    if "confirmTargetVariable" not in st.session_state:
       st.session_state.confirmTargetVariable = False  
    
    st.session_state.confirmTargetVariable = True
    
    if "toCategorical" not in st.session_state:
       st.session_state.toCategorical = False
        
    st.session_state.toCategorical = True

def displayClassDistribution(datasetHolder, targetVariable, nUniqueValues, titlePart = " "):
    '''Display the class distribution of the target variable in the training set.'''
    
    st.write(" ")
    st.write(" ")
    
    if nUniqueValues > 2:
       namePart = "binarized "
    else:
       namePart = ""
        
    classDistribution = datasetHolder[namePart + targetVariable].value_counts(normalize = True)
    classDistribution.sort_index(inplace = True)
    
    fig, ax = plt.subplots()
    classDistribution.plot(kind = "barh", xlabel = "Proportion", ylabel = "Class", width = 0.4)
    
    plt.rcParams["axes.linewidth"] = 0.2
    fig.set_size_inches(5, 1.3)
    title = "Class Distribution of " + namePart + targetVariable + " in the" + titlePart + "Training Set"
    ax.set_title(title, fontsize = 5.5)
    ax.set_xlim(0, 1)
    ax.xaxis.set_tick_params(length = 1.5, width = 0.3, labelsize = 4.5)
    ax.yaxis.set_tick_params(length = 1.5, width = 0.3, labelsize = 4.5)
    ax.xaxis.label.set_size(5)
    ax.yaxis.label.set_size(5)
    st.pyplot(fig)
    
    return classDistribution
    
def displayDataset(dataset, header, targetVariable = None, displayInstancesCount = True):
    '''Display dataset.'''
    
    st.markdown(header)
    
    if targetVariable is not None:
       
       columns = dataset.columns.tolist()
       columns.remove(targetVariable)
       columns = columns + [targetVariable]
       dataset = dataset[columns]

    st.dataframe(dataset, height = 215)
    
    if displayInstancesCount:
       st.write("Number of Instances = ", dataset.shape[0])
       
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

def runBalancingMethod(balancingMethod, currentStage):
    '''Change the values of the stage, runBalancingMethod, and randomUnderSampling keys
       and delete the resampling key if it exists.'''
    
    st.session_state.stage = currentStage
    st.session_state.runBalancingMethod = True
    
    if balancingMethod == "Random Under-sampling":
       st.session_state.randomUnderSampling = True
    
    if "resampling" in st.session_state:
       del st.session_state["resampling"]
    
def selectBalancingMethod(balancing, currentStage, nextStage, changeMethod = False):
    '''Change the values of the selectBalancingMethod, stage, or runBalancingMethod keys.'''
    
    bOption1 = "Balance out the class distribution. (Recommended)"
    
    if balancing == bOption1:
        st.session_state.selectBalancingMethod = True
    else:
        st.session_state.stage = nextStage
        
    if changeMethod:
        
       st.session_state.stage = currentStage
       st.session_state.runBalancingMethod = False
       
def selectBalancingOption(currentStage):
    
    st.session_state.stage = currentStage
    st.session_state.selectBalancingMethod = False
    st.session_state.runBalancingMethod = False

def setAllOptions():
    '''Change the value of the allOptions key.'''
    
    condition = st.session_state[model1] and \
                st.session_state[model2] and \
                st.session_state[model3] and \
                st.session_state[model4] and \
                st.session_state[model5] and \
                st.session_state[model6] and \
                st.session_state[model7] and \
                st.session_state[model8] and \
                st.session_state[model9]    

    if condition:
       st.session_state.allOptions = True
    else:
       st.session_state.allOptions = False
       
    setStage(12)

def setOptions():
    '''Change the values of the model keys.'''

    if st.session_state.allOptions:
        
       st.session_state[model1] = True
       st.session_state[model2] = True
       st.session_state[model3] = True
       st.session_state[model4] = True
       st.session_state[model5] = True
       st.session_state[model6] = True
       st.session_state[model7] = True
       st.session_state[model8] = True
       st.session_state[model9] = True
       
    else:
         
       st.session_state[model1] = False
       st.session_state[model2] = False
       st.session_state[model3] = False
       st.session_state[model4] = False
       st.session_state[model5] = False
       st.session_state[model6] = False
       st.session_state[model7] = False
       st.session_state[model8] = False
       st.session_state[model9] = False
       
    setStage(12)
    
def setStage(i):
    '''Change the value of the stage key and perform certain actions according
       to the value of the stage key.'''
    
    st.session_state.stage = i
    
    if i == 0:
       
       if "targetVariable" in st.session_state:
          del st.session_state["targetVariable"]
          
       if "toTargetVariable" in st.session_state:
          del st.session_state["toTargetVariable"]
          
       if "confirmTargetVariable" in st.session_state:
          del st.session_state["confirmTargetVariable"]
          
       if "randomUnderSampling" in st.session_state:
          del st.session_state["randomUnderSampling"]
          
       if "resampling" in st.session_state:
          del st.session_state["resampling"]
          
       if "toCategorical" in st.session_state:
          del st.session_state["toCategorical"]
          
        
    if i <= 3:
       st.session_state.toTargetVariable = False
       
    if i <= 5:
        
       st.session_state.selectBalancingMethod = False
       st.session_state.runBalancingMethod = False
       
    if i <= 9:
       
       if "trainTestSplit" in st.session_state:
          del st.session_state["trainTestSplit"]
          
    
                    