"""
         File: scikit-learnClassification.py
 Date Created: February 6, 2024
Date Modified: June 18, 2024
----------------------------------------------------------------------------------------------
Walk the user through the steps in training and testing one or more binary classifiers using a 
selection of algorithms that are implemented in scikit-learn.
----------------------------------------------------------------------------------------------
"""

from helperFunctions import actOnClassImbalance, binarizeTarget, changeTargetVariable, \
                            checkUploadedTestSet, confirmTargetVariable, displayClassDistribution, \
                            displayDataset, printTrainingResults, setAllOptions, setOptions, setStage
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits, load_iris, load_wine
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import modelParams
import numpy as np
import pandas as pd
import pickle
import scipy
import streamlit as st

#-----------------------------------------------------------------
# Create and initialize session state keys and display intro page.
#-----------------------------------------------------------------

st.markdown("## Binary Classification Using `scikit-learn`")

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
   
if "targetVariable" not in st.session_state:
   st.session_state.targetVariable = None
      
if "toTargetVariable" not in st.session_state:
   st.session_state.toTargetVariable = False
   
if "confirmTargetVariable" not in st.session_state:
   st.session_state.confirmTargetVariable = False  
   
if "nUniqueValues" not in st.session_state:
   st.session_state.nUniqueValues = 0
   
# if "binarize" not in st.session_state:
#    st.session_state.binarize = False

if "binarizer" not in st.session_state:
   st.session_state.binarizer = None
   
if "positiveClasses" not in st.session_state:
   st.session_state.positiveClasses = None
   
if "selectBalancingMethod" not in st.session_state:
   st.session_state.selectBalancingMethod = False
   
if "runBalancingMethod" not in st.session_state:
   st.session_state.runBalancingMethod = False
   
if "randomUnderSampling" not in st.session_state:
   st.session_state.randomUnderSampling = False
   
if "method" not in st.session_state:
   st.session_state.method = None
   
if "toCategorical" not in st.session_state:
   st.session_state.toCategorical = False
               
if "trainingSucceeded" not in st.session_state:   
   st.session_state.trainingSucceeded = []
   
if "trainingFailed" not in st.session_state:   
   st.session_state.trainingFailed = []  
   
if "retrainedModel" not in st.session_state:
   st.session_state.retrainedModel = {}

nUniqueValues = st.session_state.nUniqueValues
toSplit = False
toRetrain = False   

st.sidebar.write("state = ", st.session_state.stage)
      
if st.session_state.stage == 0 or st.session_state.stage == 1:
    
   st.markdown("This web app will walk you through the steps in training and testing one or more \
                binary classifiers using a selection of algorithms that are implemented in \
                `scikit-learn`.")  
   st.write(" ")             
   st.image(image = "classifiers.jpg")    
   st.write(" ")         
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

   if datasetAvailability is not None:
    
      if datasetAvailability == case1:
         st.write("Please ensure that the dataset has been cleaned and is ready for model training and testing.")
      elif datasetAvailability == case2:
           st.write("Please ensure that the training set and test set have been cleaned and are ready for \
                     model training and testing.")
        
      st.button(label = "Next", key = "datasetAvailabilityNext", on_click = setStage, args = [2])
        
   else:
      setStage(1)
   
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
   
   @st.cache_data
   def readCsv(uploadedDataset):
       
       dataset = pd.read_csv(uploadedDataset)
       
       return dataset
       
   if datasetAvailability == case1:
      
      uploadedDataset = st.sidebar.file_uploader(label = "Upload the dataset in `csv` file format.",
                                                 type = "csv",
                                                 on_change = setStage, 
                                                 args = [3])
 
      if uploadedDataset is not None:
  
         dataset = readCsv(uploadedDataset = uploadedDataset)
         variables = dataset.columns.tolist()
         datasetHolder = dataset
         
      else:
         setStage(2)
   
   elif datasetAvailability == case2:
    
        uploadedTrainSet = st.sidebar.file_uploader(label = "Upload the training set in `csv` file format.",
                                                    on_change = setStage, 
                                                    args = [3])
   
        if uploadedTrainSet is not None:
    
           trainSet = readCsv(uploadedDataset = uploadedTrainSet)
           variables = trainSet.columns.tolist()
           datasetHolder = trainSet
           
        else:
           setStage(2)
           
   elif datasetAvailability == case3:
       
        datasetName = st.sidebar.selectbox(label = "Select a dataset.",
                                           options = ["Breast Cancer", "Diabetes", "Digits", "Iris", "Wine"],
                                           index = None,
                                           on_change = setStage,
                                           args = [3],
                                           placeholder = "Select a dataset")
        
        st.sidebar.write("You can find some information about these datasets in \
                         [scikit-learn](https://scikit-learn.org/stable/datasets/toy_dataset.html).")
        
        if datasetName is not None:
            
           if datasetName == "Breast Cancer":
              dataset = load_breast_cancer(return_X_y = True, as_frame = True)
           elif datasetName == "Diabetes":
                dataset = load_diabetes(return_X_y = True, as_frame = True)
           elif datasetName == "Digits":
                dataset = load_digits(return_X_y = True, as_frame = True)
           elif datasetName == "Iris":
                dataset = load_iris(return_X_y = True, as_frame = True)   
           elif datasetName == "Wine":
                dataset = load_wine(return_X_y = True, as_frame = True)
            
           dataset = pd.concat([dataset[0], dataset[1]], axis = 1)
           variables = dataset.columns.tolist()
           datasetHolder = dataset
           
        else:
          setStage(2)
 
#---------------------
# Display the dataset.
#---------------------

if st.session_state.stage >= 3:
    
   targetVariable = st.session_state.targetVariable
   
   if datasetAvailability in [case1, case3]:
      displayDataset(dataset = dataset, header = "**Dataset**", targetVariable = targetVariable)
   else:
      displayDataset(dataset = trainSet, header = "**Training Set**", targetVariable = targetVariable)
   
   if datasetAvailability in [case1, case2]: 
      st.button(label = "Next", key = "displayDatasetNext1", on_click = setStage, args = [4])
   else:
      st.session_state.toTargetVariable = True
    
#----------------------------
# Select the target variable.
#----------------------------

if st.session_state.stage >= 4 or st.session_state.toTargetVariable:
    
   if datasetAvailability in [case1, case2]:
      
      @st.experimental_fragment 
      def selectTargetVariable():
          
          targetVariable = st.selectbox(label = "Select the target variable.",
                                        options = variables,
                                        index = None,
                                        on_change = changeTargetVariable,
                                        placeholder = "Select a variable")
      
          if targetVariable is not None:
          
             if targetVariable == variables[-1]:
                messagePart = ""
             else:
                messagePart = "Once the button is clicked, the selected variable will be moved to the \
                               last column of the displayed dataframe above."
         
             st.write("Click the button below to confirm the selection of the target variable. " + messagePart)      
             
             if st.button(label = "Confirm target variable", 
                          on_click = confirmTargetVariable, 
                          args = [targetVariable]):
                st.rerun()
      
          if targetVariable is None:
             setStage(4)
             
      selectTargetVariable()
      
   else:
       
      targetVariable = "target"
      messagePart = "For the purpose of this binary classification, you can binarize the target variable \
                     by identifying the classes that will be map to 1."
      
      if datasetName == "Breast Cancer":
         st.write("There are 30 features which are all continuous. The target variable has 2 classes. \
                   The 2 classes are: malignant (0) and benign (1).")  
      elif datasetName == "Diabetes":
           st.write("There are 10 features which are all continuous. The target is also continuous. For the \
                     purpose of this binary classification, you can binarize the target variable according \
                     to a threshold. Values greater than the threshold map to 1 while values less than or \
                     equal to the threshold map to 0.")
      elif datasetName == "Digits":
           st.write("There are 64 features which are all continuous. The target variable has 10 classes \
                     where each class refers to a digit. " + messagePart) 
      elif datasetName == "Iris":
           st.write("There are 4 features which are all continuous. The target variable has 3 classes. \
                     The 3 classes are the 3 different types of the Iris flowering plant: setosa (0), \
                     versicolor (1), and virginica (3). " + messagePart) 
      elif datasetName == "Wine":
           st.write("There are 13 features which are all continuous. The target variable has 3 classes. \
                     The 3 classes are 3 different types of wine. " + messagePart) 
   
   if targetVariable is not None:
      
      uniqueValues = np.sort(datasetHolder[targetVariable].unique())
      nUniqueValues = len(uniqueValues)
      st.session_state.nUniqueValues = nUniqueValues
      
      if nUniqueValues == 2:
         
         if datasetAvailability == case3:
            st.button(label = "Next", key = "targetVariableNext", on_click = setStage, args = [7])
          
      elif nUniqueValues > 2:
         
           if datasetAvailability in [case1, case2]:
             
              if st.session_state.confirmTargetVariable:
                  
                 st.write("`" + targetVariable + "` is not a binary variable. You may either change your \
                           selection or binarize the selected variable.")
                  
                 st.button(label = "Binarize target variable", on_click = setStage, args = [5])     
                     
           else:
               
              if datasetName == "Diabetes":
                 variableType = "continuous"
              else:
                 variableType = "categorical"
                 
              output = binarizeTarget(dataset = dataset, classes = uniqueValues, targetVariable = targetVariable,
                                      variableType = variableType, currentStage = 4, nextStage = 7,
                                      positiveClassesKey = datasetName)
    
   features = variables.copy()
   
   if targetVariable in features:
      features.remove(targetVariable)
   
#--------------------------------------------------------------------------------
# Select the type of the selected target variable with more than 2 unique values.
#--------------------------------------------------------------------------------

if st.session_state.stage >= 5:
       
   if nUniqueValues > 2 and datasetAvailability in [case1, case2]:
       
      label = "`" + targetVariable + "` is also a numeric variable. Please identify how this variable will be regarded."
      type1 = "continuous variable"
      type2 = "categorical variable with more than 2 classes"
      
      @st.experimental_fragment 
      def selectTargetVariableType():
          
          numericVariables = list(datasetHolder.select_dtypes(include = "number").columns)
          
          if targetVariable in numericVariables:
         
             vType = st.radio(label = label,
                              options = [type1, type2],
                              index = None,
                              on_change = setStage,
                              args = [5])
             
          else:
             vType = type2
             
          st.session_state.vType = vType
        
          if vType is not None:
              
             if datasetAvailability == case1:
                nextStage = 7
             elif datasetAvailability == case2:
                  nextStage = 6
            
             if vType == type1:
                 
                binarizer = binarizeTarget(dataset = datasetHolder, classes = uniqueValues, targetVariable = targetVariable,
                                           variableType = "continuous", currentStage = 5, nextStage = nextStage)
                st.session_state.binarizer = binarizer
                
             else:
                 
                positiveClasses = binarizeTarget(dataset = datasetHolder, classes = uniqueValues, targetVariable = targetVariable,
                                                 variableType = "categorical", currentStage = 5, nextStage = nextStage)
                st.session_state.positiveClasses = positiveClasses
                
      selectTargetVariableType()
                
#-------------------------------------------------------------------------------------------------
# If case 2 is selected, display the class distribution of the target variable in the training set
# and select on how to handle an imbalanced class distribution.
#-------------------------------------------------------------------------------------------------

if (st.session_state.stage >= 6 or (st.session_state.confirmTargetVariable and nUniqueValues == 2)) and \
   datasetAvailability == case2:
       
   classDistribution = displayClassDistribution(datasetHolder = datasetHolder, 
                                                targetVariable = targetVariable, 
                                                nUniqueValues = nUniqueValues)
    
   balancingOutput = actOnClassImbalance(classDistribution = classDistribution,
                                         nUniqueValues = nUniqueValues, 
                                         trainSet = trainSet, 
                                         features = features, 
                                         targetVariable = targetVariable,
                                         currentStage = 6,
                                         nextStage = 7)
   
   if type(balancingOutput) == tuple:
      trainSet = balancingOutput[0]
   elif type(balancingOutput) == pd.core.frame.DataFrame:
      trainSet = balancingOutput
   else:
      setStage(6)
   
#-----------------------------------------------------------------------------------------------
# Select the categorical features and/or continuous features that will be used in model training
# and testing.
#-----------------------------------------------------------------------------------------------      
      
if st.session_state.stage >= 7 or (st.session_state.toCategorical and nUniqueValues == 2 and \
                                   datasetAvailability == case1):         
    
   if datasetAvailability in [case1, case2]:
      
      @st.experimental_fragment 
      def selectCatAndCont():
       
          st.write(" ") 
          st.write("Select the categorical features and/or continuous features that will be used in \
                    model training and testing.")
        
          label1 = "Select categorical features. "
          label2 = "You may select `All` if all features in the dataset are regarded as categorical and "
          label3 = "they will all be used in model training and testing. "
          label4 = "Leave the field blank if no categorical features will be used."
          categoricalFeatures = st.multiselect(label = label1 + label2 + label3 + label4,
                                               options = features + ["All"],
                                               on_change = setStage,
                                               args = [7],
                                               placeholder = "")
          st.session_state.categoricalFeatures = categoricalFeatures
          continuousFeatures = []
          completeSelection = True
                 
          if len(categoricalFeatures) >= 2 and "All" in categoricalFeatures:
         
              st.markdown(":red[You cannot select All together with one or more features. Please edit \
                           your selection.]")
              setStage(7)
        
          elif len(categoricalFeatures) == 0 or (categoricalFeatures[0] != "All" and len(categoricalFeatures) < len(features)):
                 
               label1 = "Select continuous features. "
               label2 = "You may select `All` if all the numeric features in the dataset including those that are not selected as "
               label3 = "categorical features will be regarded as continuous and will be used in model training and testing. "
               label4 = "Leave the field blank if no continuous features will be used."
               continuousFeaturesOptions = set(datasetHolder.select_dtypes(include = "number").columns)
               continuousFeaturesOptions = continuousFeaturesOptions.difference({targetVariable, "binarized " + targetVariable})
               continuousFeaturesOptions = list(continuousFeaturesOptions.difference(set(categoricalFeatures)))
               continuousFeaturesOptions = [f for f in features if f in continuousFeaturesOptions]
               st.session_state.continuousFeaturesOptions = continuousFeaturesOptions
               
               if len(continuousFeaturesOptions) > 0:
                      
                  if len(continuousFeaturesOptions) > 2:
                     continuousFeaturesOptions.append("All")
                      
                  continuousFeatures = st.multiselect(label = label1 + label2 + label3 + label4,
                                                     options = continuousFeaturesOptions,
                                                     on_change = setStage,
                                                     args = [7],
                                                     placeholder = "")
                  st.session_state.continuousFeatures = continuousFeatures
                  
                  if len(continuousFeatures) >= 2 and "All" in continuousFeatures:
                      
                     st.markdown(":red[You cannot select All together with one or more features. Please edit \
                                  your selection.]")
                     setStage(7)
                     completeSelection = False
                     
                  else:
                     completeSelection = True
                      
          if completeSelection:      
        
             st.write("Click the button below to complete the selection of features.")            
          
             if st.button(label = "Complete selection", key = "featuresSelection", on_click = setStage, 
                          args = [8]):
           
                if len(categoricalFeatures) == 0 and len(continuousFeatures) == 0:
             
                   st.markdown(":red[Please select at least one feature.]")
                   setStage(7)
                   
                st.rerun()
                
      selectCatAndCont()
                         
   else:
       
      categoricalFeatures = []
      toSplit = True
      
#-------------------------------------------------------------------------------
# If case1 or case2 is selected, display the dataset with the selected features.
#-------------------------------------------------------------------------------

if st.session_state.stage >= 8:
   
   if datasetAvailability in [case1, case2]:
       
      categoricalFeatures = st.session_state.categoricalFeatures
      continuousFeaturesOptions = st.session_state.continuousFeaturesOptions
      continuousFeatures = st.session_state.continuousFeatures
       
      if len(categoricalFeatures) == 1 and categoricalFeatures[0] == "All":
         categoricalFeatures = features
        
      if len(continuousFeatures) == 1 and continuousFeatures[0] == "All":
          
         continuousFeatures = continuousFeaturesOptions.copy()
         
         if "All" in continuousFeatures:
            continuousFeatures.remove("All")
              
      featuresToUse = categoricalFeatures + continuousFeatures
     
      nFeaturesToUse = len(featuresToUse)
      nFeatures = len(features)
         
      if nFeaturesToUse < nFeatures:
          
         if nUniqueValues > 2:
            columnsToDisplay = featuresToUse + [targetVariable] + ["binarized " + targetVariable]
         else:
            columnsToDisplay = featuresToUse + [targetVariable] 
            
         if datasetAvailability == case1:   
             
            datasetToDisplay = dataset[columnsToDisplay]
            header = "**Dataset with Selected Features**"   
            
         else:
             
            datasetToDisplay = trainSet[columnsToDisplay]
            
            if type(balancingOutput) == tuple:
               header = "**Balanced Training Set with Selected Features**"
            elif type(balancingOutput) == pd.core.frame.DataFrame:
                 header = "**Training Set with Selected Features**"
               
         displayDataset(dataset = datasetToDisplay, header = header, displayInstancesCount = False)
          
      else:
         st.write("All features in the dataset are selected to be used in model training and testing.") 
         
      features = featuresToUse
         
      if datasetAvailability == case1:
         args = [9]
      else:
         args = [11]
  
      st.button(label = "Next", key = "displayDatasetNext2", on_click = setStage, args = args)   
      
#--------------------------------------------------------------------------------------
# If case1 or case3 is selected, split the dataset into a training set and a test set .
#--------------------------------------------------------------------------------------         
 
if st.session_state.stage >= 9 or toSplit:
   
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
                                       args = [9])
        
      testSetSize = 1 - trainingSetSize
   
      st.write("training set size = %.2f" % trainingSetSize)
      st.write("    test set size = %.2f" % testSetSize)
      
      if nUniqueValues > 2:
          
         X = dataset[features + [targetVariable]]
         y = dataset[["binarized " + targetVariable]]
      
      else:
          
         X = dataset[features]
         y = dataset[[targetVariable]]
         
      xTrain, xTest, yTrain, yTest = train_test_split(X, y, train_size = trainingSetSize)
      
      st.write(" ")
      st.write("Click the button below to confirm the selected train-test split ratio and \
                to display the randomly formed training set.")
      
      st.button(label = "Display training set", on_click = setStage, args = [10])
   
#-------------------------------------------------------------------------------------------
# If case1 or case3 is selected, display the training set and the class distribution of the 
# target variable in the training set. 
#-------------------------------------------------------------------------------------------

if st.session_state.stage >= 10: 
    
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
      
      displayDataset(dataset = trainSet, header = "**Training Set**")
      classDistribution = displayClassDistribution(datasetHolder = trainSet, 
                                                   targetVariable = targetVariable, 
                                                   nUniqueValues = nUniqueValues)
      balancingOutput = actOnClassImbalance(classDistribution = classDistribution,
                                            nUniqueValues = nUniqueValues, 
                                            trainSet = trainSet, 
                                            features = features, 
                                            targetVariable = targetVariable,
                                            currentStage = 10,
                                            nextStage = 11)
         
      if type(balancingOutput) == tuple:
         trainSet = balancingOutput[0]
      elif type(balancingOutput) == pd.core.frame.DataFrame:
         trainSet = balancingOutput
      else:
         setStage(10)
            
#-----------------------------------------------------------------------
# Select whether or not transformations will be applied to the features.
#-----------------------------------------------------------------------
        
if st.session_state.stage >= 11:
    
   st.write(" ")
    
   label = "Please select whether or not transformations will be applied to the features."
    
   if len(categoricalFeatures) == 0:
        
      ftOption1 = "Standardize the continuous features."
      messagePart = "continuous"
       
   elif categoricalFeatures[0] == "All" or len(categoricalFeatures) == len(features):
        
        ftOption1 = "One-hot encode the categorical features."
        messagePart = "categorical"
         
   else:
        
        ftOption1 = "Standardize the continuous features and one-hot encode the categorical features."
        messagePart = ""
       
   ftOption2 = "Retain the form of the " + messagePart + " features."
    
   featureTransformation = st.radio(label = label, 
                                    options = [ftOption1, ftOption2], 
                                    index = None,
                                    on_change = setStage,
                                    args = [11])
           
   if featureTransformation == ftOption1:
     
      if len(categoricalFeatures) == 1 and categoricalFeatures[0] == "All":
         colTransformer = ColumnTransformer(transformers = [("categorical", OneHotEncoder(drop = "first"), features)])
      elif len(categoricalFeatures) == 0:
           colTransformer = ColumnTransformer(transformers = [("continuous", StandardScaler(), features)])
      else:
           colTransformer = ColumnTransformer(transformers = [("categorical", OneHotEncoder(drop = "first"), 
                                                               categoricalFeatures)],
                                              remainder = StandardScaler())
       
      colTransformer.fit(trainSet[features])
      
      xTrain = colTransformer.transform(trainSet[features])
      
      if scipy.sparse.issparse(xTrain):
         xTrain = xTrain.toarray()
          
      if nUniqueValues > 2:
         yTrain = trainSet[["binarized " + targetVariable]]
      elif nUniqueValues == 2:
           yTrain = trainSet[[targetVariable]]
     
      transformedFeatures = colTransformer.get_feature_names_out().tolist()
      
      if len(categoricalFeatures) == 1 and categoricalFeatures[0] == "All":
         transformedFeatures = [i.replace("categorical__", "") for i in transformedFeatures]
      elif len(categoricalFeatures) == 0:
           transformedFeatures = [i.replace("continuous__", "") for i in transformedFeatures]
      else:
          
           transformedFeatures = [i.replace("categorical__", "") for i in transformedFeatures]
           transformedFeatures = [i.replace("remainder__", "") for i in transformedFeatures]
     
      xTrainDf = pd.DataFrame(xTrain, index = trainSet.index, columns = transformedFeatures)
      
      if nUniqueValues > 2:
         transformedTrainSet = pd.concat([xTrainDf, trainSet[targetVariable], yTrain], axis = 1)
      elif nUniqueValues == 2:
           transformedTrainSet = pd.concat([xTrainDf, yTrain], axis = 1)
      
      if type(balancingOutput) == tuple:
         header = "**Balanced Training Set with Transformed Features**"
      elif type(balancingOutput) == pd.core.frame.DataFrame:
           header = "**Training Set with Transformed Features**"
      
      displayDataset(dataset = transformedTrainSet, header = header, displayInstancesCount = False)
       
   else:
      xTrain = trainSet[features].to_numpy()
   
   if featureTransformation:
         
      st.write(" ")
      st.write("Click the button below to select at least one model to train.")     
    
      st.button(label = "Select one or more models", on_click = setStage, args = [12]) 
    
#------------------------------------
# Select at least one model to train.
#------------------------------------

if st.session_state.stage >= 12: 
   
   if nUniqueValues > 2:
      yTrain = trainSet["binarized " + targetVariable].to_numpy()
   else:
      yTrain = trainSet[targetVariable].to_numpy()
   
   model1 = "Decision Tree Classifier"
   model2 = "Gaussian Naive Bayes Classifier"
   model3 = "Gaussian Process Classifier"
   model4 = "k-Nearest Neighbors Classifier"
   model5 = "Logistic Regression"
   model6 = "Multi-layer Perceptron Classifier"
   model7 = "Quadratic Discriminant Analysis"
   model8 = "Random Forest Classifier"
   model9 = "Support Vector Classifier"
   
   @st.experimental_fragment 
   def selectModels():
       
       disabledSelectModels = False
       st.checkbox(label = model1, key = model1, on_change = setAllOptions, disabled = disabledSelectModels)
       st.checkbox(label = model2, key = model2, on_change = setAllOptions, disabled = disabledSelectModels)
       st.checkbox(label = model3, key = model3, on_change = setAllOptions, disabled = disabledSelectModels)
       st.checkbox(label = model4, key = model4, on_change = setAllOptions, disabled = disabledSelectModels)
       st.checkbox(label = model5, key = model5, on_change = setAllOptions, disabled = disabledSelectModels)
       st.checkbox(label = model6, key = model6, on_change = setAllOptions, disabled = disabledSelectModels)
       st.checkbox(label = model7, key = model7, on_change = setAllOptions, disabled = disabledSelectModels)
       st.checkbox(label = model8, key = model8, on_change = setAllOptions, disabled = disabledSelectModels)
       st.checkbox(label = model9, key = model9, on_change = setAllOptions, disabled = disabledSelectModels)
       st.checkbox("Select all models", key = 'allOptions', on_change = setOptions, disabled = disabledSelectModels)
       
       options = [model1, model2, model3, model4, model5, model6, model7, model8, model9]
       st.session_state.options = options
       selected = 0
       for option in options:
           selected += st.session_state[option]
       
       st.write("Click the button below to complete the selection of models.")
       if st.button(label = "Complete selection", key = "modelSelection", on_click = setStage, args = [13]):
       
          if selected == 0:
               
             st.markdown(":red[Please select at least one model to train.]")
             setStage(12)
             
          else: 
             disabledSelectModels = True
             
          st.rerun()
          
   selectModels()
   
#--------------------------------------------------------------------
# Click to set the values of the parameters of the selected model(s).
#--------------------------------------------------------------------
    
if st.session_state.stage >= 13:
    
    trueOptions = []
    options = st.session_state.options
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
             
    st.button(label = "Set model parameters", on_click = setStage, args = [14])

#-----------------------------------------------------------
# Set the values of the parameters of the selected model(s).
#-----------------------------------------------------------
    
if st.session_state.stage >= 14:
    
   st.write("")
    
   paramsValues = {}
   disabledValues = {}
    
   if m == 1:
      paramsValues[trueOptions[0]], disabledValues[trueOptions[0]] = modelParams.setModelParams(trueOptions[0])
   else:
      
      if m >= 5:
         st.write("To scroll through the tabs, click on a tab and use the left-arrow and right-arrow keys.")
       
      tabs = st.tabs([trueOptions[i] for i in range(m)])
    
      for i in range(m):
    
          with tabs[i]:
        
               paramsValues_i, disabledValues_i = modelParams.setModelParams(trueOptions[i])
        
          paramsValues[trueOptions[i]] = paramsValues_i
          disabledValues[trueOptions[i]] = disabledValues_i
          
   disabled = 0
   for i in range(m):
       disabled += disabledValues[trueOptions[i]]
       
   if disabled == 1:
       
      messagePart1 = "input" 
      messagePart2 = "message"
      
   elif disabled > 1:
       
        messagePart1 = "inputs" 
        messagePart2 = "messages"
        
   if m == 1:
      messagePart3 = "above"
   else:
       
      if disabled == 1:
         messagePart3 = "in one of the tabs above"
      elif disabled > 1:
           messagePart3 = "in one or more of the tabs above"
   
   st.write(" ")
        
   if disabled == 0:     
      st.write("Click the `Reset model parameters` button to set the model parameters again or click the \
                `Confirm` button to confirm the assigned values of the model parameters.")
   else:
       st.markdown("Please correct the " + messagePart1 + " " + messagePart3 + " as indicated by the " + messagePart2 + \
                   " in :red[red] to enable the `Confirm model parameters` button.")    

   cols = st.columns(3)
   
   with cols[0]:
        st.button(label = "Reset model parameters", on_click = setStage, args = [13])   
        
   with cols[1]:
        st.button(label = "Confirm model parameters", on_click = setStage, args = [15], disabled = disabled)

#-----------------------------
# Train the selected model(s).
#-----------------------------

if st.session_state.stage >= 15:
    
   if "models" not in st.session_state:
      st.session_state.models = {}
      
   if m == 1:
      messagePart = "model"
   else:
      messagePart = "models"
      
   st.write(" ") 
   st.write("Click the button below to train the selected " + messagePart + ".")

   if st.button(label = "Train " + messagePart, on_click = setStage, args = [16]): 
            
      trainingSucceeded = []
      trainingFailed = []
      for option in trueOptions:
          
          if option == model1:
             model = DecisionTreeClassifier(random_state = 1, **paramsValues[option])
          elif option == model2:
               model = GaussianNB(**paramsValues[option])
          elif option == model3:
               model = GaussianProcessClassifier(random_state = 1, **paramsValues[option])
          elif option == model4:  
               model = KNeighborsClassifier(**paramsValues[option])  
          elif option == model5:
               model = LogisticRegression(random_state = 1, **paramsValues[option])
          elif option == model6:
               model = MLPClassifier(random_state = 1, **paramsValues[option])
          elif option == model7:
               model = QuadraticDiscriminantAnalysis(**paramsValues[option])
          elif option == model8:
               model = RandomForestClassifier(random_state = 1, **paramsValues[option])
          elif option == model9:
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

if st.session_state.stage >= 16:
    
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
          setStage(16)
       else:
                    
          st.button(label = "Display test set", on_click = setStage, args = [17])
       
          if st.session_state.stage >= 17:
             displayDataset(dataset = testSet, header = "**Test Set**")
       
   else:
      
      messagePart = "Upload the test set at the sidebar."
      printTrainingResults(trainingSucceeded = trainingSucceeded, 
                           trainingFailed = trainingFailed,
                           messagePart = messagePart) 
   
      uploadedTestSet = st.sidebar.file_uploader(label = "Upload the test set in `csv` file format.",
                                                 on_change = setStage, 
                                                 args = [17])   
   
      if uploadedTestSet is not None:
    
         testSet = pd.read_csv(uploadedTestSet)
         output = checkUploadedTestSet(testSet, features, targetVariable, nFeaturesToUse, nFeatures)
         
         if type(output) == tuple:
            toDisplay, testSet, header = output
         else:
            toDisplay = output
              
         if toDisplay:
             
            if nUniqueValues > 2:
                
               vType = st.session_state.vType
                               
               if vType == type1:
                     
                  y = testSet[[targetVariable]] 
                  binarizer = st.session_state.binarizer
                  testSet["binarized " + targetVariable] = binarizer.transform(y).astype(int)
                   
               else:
                   
                  positiveClasses = st.session_state.positiveClasses
                  testSet["binarized " + targetVariable] = testSet[targetVariable].apply(lambda x: 1 if x in positiveClasses else 0)
                   
               displayDataset(dataset = testSet, header = header)
                   
            else:
               displayDataset(dataset = testSet, header = header)
      
      else:
         setStage(16)
   
#-----------------------------------
# Click to display the ROC curve(s).
#-----------------------------------

if st.session_state.stage >= 17:
   
   if featureTransformation == ftOption1:
       
      try:
           
         xTest = colTransformer.transform(testSet[features])  
         
         if scipy.sparse.issparse(xTest):
            xTest = xTest.toarray()
         
         if datasetAvailability == case2:
            
            if nUniqueValues > 2:
               yTest = testSet[["binarized " + targetVariable]]
            elif nUniqueValues == 2:
                 yTest = testSet[[targetVariable]]
            
         xTestDf = pd.DataFrame(xTest, index = testSet.index, columns = transformedFeatures)
         
         if nUniqueValues > 2:
            transformedTestSet = pd.concat([xTestDf, testSet[targetVariable], yTest], axis = 1)
         elif nUniqueValues == 2:
              transformedTestSet = pd.concat([xTestDf, yTest], axis = 1)
         
         displayDataset(dataset = transformedTestSet, header = "**Test Set with Transformed Features**",
                        displayInstancesCount = False)
         
         toROC = True
       
      except:
          
          if st.session_state.randomUnderSampling:
               messagePart = "The error may be avoided by re-running random under-sampling or selecting a \
                              different balancing method."
          elif datasetAvailability in [case1, case3]:
               messagePart = "The error may be avoided by increasing the train-test split ratio."
          else:
               messagePart = ""
             
          st.markdown(":red[An error occured in transforming one or more columns of the test set. " + \
                      messagePart + "]")
          toROC = False
          setStage(17)
                      
   else:
       
      xTest = testSet[features].to_numpy()
      toROC = True
   
   if toROC:
       
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
    
      st.write(" ")
      st.write("Click the button below to display the ROC " + messagePart2 + " of the " + messagePart1 + " on the test set.")
      
      st.button(label = "Display the ROC " + messagePart2, on_click = setStage, args = [18])
       
      st.write(" ")
      st.write(" ")

#--------------------------
# Display the ROC curve(s).
#--------------------------

if st.session_state.stage >= 18:
       
   if nUniqueValues > 2: 
      yTest = testSet["binarized " + targetVariable].to_numpy() 
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
      elif nRows == 3:
           figsize = (9, 14)
      elif nRows == 4:
           figsize = (9, 19)
      else:
           figsize = (9, 24)
      
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
           bestModelsMessage = "The best models among the " + str(ts) + messagePart + "models on the basis of \
                               their ROC AUC scores on the test set are " + trainingSucceeded[indicesMaxRocAucScore[0]] + \
                               " and " + trainingSucceeded[indicesMaxRocAucScore[1]] + ". "
      elif n >= 3 and ts >=3 and n != ts:     
           bestModelsMessage = "There are " + str(n) + " best models among the " + str(ts) + messagePart + "models \
                                on the basis of their ROC AUC scores on the test set. "
      elif n == ts:
           bestModelsMessage = "The" + messagePart + "models have the same ROC AUC score on the test set. "
    
   toRetrain = True            
       
#----------------------
# Retrain a best model.
#----------------------

if toRetrain:
   
   st.write(" ")
   st.write(" ") 
   
   if datasetAvailability in [case1, case3]:
      fullDataset = dataset
   else:
      fullDataset = pd.concat([datasetHolder, testSet], ignore_index = True)
            
   if type(balancingOutput) == tuple:
       
      balancingMethod = balancingOutput[1] 
      
      if nUniqueValues > 2:
         xDatasetResampled, yDatasetResampled = balancingMethod.fit_resample(fullDataset[features], 
                                                                             fullDataset[["binarized " + targetVariable]]) 
      else:
         xDatasetResampled, yDatasetResampled = balancingMethod.fit_resample(fullDataset[features], 
                                                                             fullDataset[[targetVariable]])
         
      datasetRetrain = pd.concat([xDatasetResampled, yDatasetResampled], axis = 1)
      messagePart1 = " the dataset obtained by balancing out the class distribution of "
      
   else:
       
      datasetRetrain = fullDataset
      messagePart1 = " "
      
   if featureTransformation == ftOption1:
       
      colTransformer.fit(datasetRetrain[features])     
      x = colTransformer.transform(datasetRetrain[features])
      
   else:
      x = datasetRetrain[features].to_numpy()
   
   if nUniqueValues > 2: 
      y = datasetRetrain["binarized " + targetVariable].to_numpy()
   else:
      y = datasetRetrain[targetVariable].to_numpy()
      
   retrainedModel = {}
   
   if ts == 1 or n == 1:
       
      if n == 1 and ts == 2:
         messagePart2 = " the better "
      elif n == 1 and ts >= 3:
           messagePart2 = " the best "
      else:
           messagePart2 = " "
      
      st.write("Click the button below to retrain" + messagePart2 + "model using" + messagePart1 + \
               "the full dataset (the union of the training and test sets).")
    
      if ts == 1:
         modelName = trainingSucceeded[0]
      elif n == 1:
           modelName = trainingSucceeded[indicesMaxRocAucScore[0]]
           
      model = st.session_state.models[modelName]
    
      modelNameParts = modelName.split() 
      filename = modelNameParts[0].lower()
   
      for i in range(1, len(modelNameParts)):
          filename += modelNameParts[i]
      
      st.button(label = "Retrain" + messagePart2 + "model", on_click = setStage, args = [19])
         
      model.fit(x, y)
      retrainedModel[filename] = model
      st.session_state.retraindedModel = retrainedModel
         
      messagePart = "Click the `Reset` button to go back to the intro page or close the browser's tab \
                     displaying this web app to exit."
         
   else:
      
      if n == ts and tf == 0:
         messagePart2 = " "
      elif n == ts and tf > 0:
           messagePart2 = " successfully trained "
      else:
         messagePart2 = " best " 
             
      label = bestModelsMessage + "Select one" + messagePart2 + "model to retrain using" + messagePart1 + \
              "the full dataset (the union of the training and test sets)."
      modelToRetrain = st.radio(label = label,
                                options = [trainingSucceeded[indicesMaxRocAucScore[i]] for i in range(n)],
                                index = None,
                                on_change = setStage,
                                args = [18])
      
      if modelToRetrain is not None:
          
         st.button(label = "Retrain the selected model", on_click = setStage, args = [19])
         
         model = st.session_state.models[modelToRetrain]
    
         modelNameParts = modelToRetrain.split() 
         filename = modelNameParts[0].lower()
   
         for j in range(1, len(modelNameParts)):
             filename += modelNameParts[j]
      
         model.fit(x, y)
         retrainedModel[filename] = model
         st.session_state.retraindedModel = retrainedModel
         
         messagePart = "You may select another model to retrain and save. You may also click the `Reset` button \
                        to go back to the intro page or close the browser's tab displaying this web app to exit."
         
      else:
         setStage(18)
       
#--------------------------
# Save the retrained model.
#--------------------------

if st.session_state.stage >= 19:
    
   st.write("Retraining completed.")     
    
   retrainedModel = st.session_state.retraindedModel
   filename = list(retrainedModel.keys())[0]
   model = retrainedModel[filename]
   
   st.write(" ")
           
   pickleObject = pickle.dumps(model)
   
   @st.experimental_fragment
   def saveModel():
       
       if st.download_button(label = "Save the retrained model", 
                             data = pickleObject, 
                             file_name = filename + ".pkl",
                             mime = "application/octet-stream"):
          
          st.write("Model saved as a pickle file in the Downloads folder. " + messagePart) 
          
   saveModel()
        