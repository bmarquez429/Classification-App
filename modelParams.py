"""
         File: modelParams.py
 Date Created: February 22, 2024
Date Modified: February 24, 2024
---------------------------------------------------------
This script is imported by scikit-learnClassification.py.
---------------------------------------------------------
"""

import streamlit as st

def setStage(i):
    "Set the session state."
    
    st.session_state.stage = i

def setLRParams():
    "Set the logistic regression parameters."
    
    penalty = st.selectbox(label = "penalty",
                           options = [None, "l1", "l2", "elasticnet"],
                           index = 2,
                           on_change = setStage,
                           args = [6])
       
    tol = st.number_input(label = "tol",
                          value = 1e-4,
                          step = 1e-5,
                          format = "%.5f",
                          on_change = setStage,
                          args = [6])
    
    if penalty is not None:
       C = st.number_input(label = "C",
                           value = 1.0,
                           step = 0.1,
                           format = "%.1f",
                           on_change = setStage,
                           args = [6])
    
    fitIntercept = st.selectbox(label = "fit_intercept",
                                options = [True, False],
                                index = 0,
                                on_change = setStage,
                                args = [6])
    
    interceptScaling = st.number_input(label = "intercept_scaling",
                                       value = 1.0,
                                       step = 0.1,
                                       format = "%.1f",
                                       on_change = setStage,
                                       args = [6])
    
    solverOptions = ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]
    if penalty is None:
       solverOptions.remove("liblinear")
    elif penalty == "l1":
         solverOptions = ["liblinear", "saga"]
    elif penalty == "elasticnet":
         solverOptions = ["saga"]
        
    solver = st.selectbox(label = "solver",
                          options = solverOptions,
                          index = 0,
                          on_change = setStage,
                          args = [6])
    
    if penalty == "l2" and solver == "liblinear":
       dual = st.selectbox(label = "dual",
                           options = [True, False],
                           index = 1,
                           on_change = setStage,
                           args = [6])
    
    maxIter = st.number_input(label = "max_iter",
                              min_value = 1,
                              value = 100,
                              step = 1,
                              on_change = setStage,
                              args = [6])
    
    multiClassOptions = ["auto", "multinomial", "ovr"]
    if solver == "liblinear":
       multiClassOptions.remove("multinomial")
    
    multiClass = st.selectbox(label = "multi_class",
                              options = multiClassOptions,
                              index = 0,
                              on_change = setStage,
                              args = [6])
    
    if penalty == "elasticnet":   
       l1Ratio = st.number_input(label = "l1_ratio",
                                 min_value = 0.00,
                                 max_value = 1.00,
                                 value = None,
                                 step = 0.01,
                                 format = "%.2f",
                                 on_change = setStage,
                                 args = [6],
                                 placeholder = "Input a number between 0 and 1 inclusive.")
    
    params = {}
    params["penalty"] = penalty
    params["tol"] = tol
    params["fit_intercept"] = fitIntercept
    params["intercept_scaling"] = interceptScaling
    params["solver"] = solver
    params["max_iter"] = maxIter
    params["multi_class"] = multiClass
    
    if penalty is not None:
       params["C"] = C
    
    if penalty == "l2" and solver == "liblinear":
       params["dual"] = dual
    
    if penalty == "elasticnet":
       params["l1_ratio"] = l1Ratio
    
    return params

def setNNCParams():
    "Set the nearest neighbors classifier parameters."
    
    nNeighbors = st.number_input(label = "n_neighbors",
                                 min_value = 1,
                                 value = 5,
                                 step = 1,
                                 on_change = setStage,
                                 args = [6])
    
    weights = st.selectbox(label = "weights",
                           options = ["uniform", "distance", None],
                           index = 0,
                           on_change = setStage,
                           args = [6])
    
    algorithm = st.selectbox(label = "algorithm",
                             options = ["auto", "ball_tree", "kd_tree", "brute"],
                             index = 0,
                             on_change = setStage,
                             args = [6])
    
    leafSize = st.number_input(label = "leaf_size",
                               min_value = 1,
                               value = 30,
                               step = 1,
                               on_change = setStage,
                               args = [6])
    
    p = st.number_input(label = "p",
                        min_value = 0.1,
                        value = 2.0,
                        step = 0.1,
                        format = "%.1f",
                        on_change = setStage,
                        args = [6])
    
    allMetricOptions = ["braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "dice", 
                        "euclidean", "hamming", "infinity", "jaccard", "jensenshannon", "kulsinski", "l1", "l2", 
                        "manhattan", "minkowski", "nan_euclidean", "p", "rogerstanimoto", "russellrao", 
                        "sokalmichener", "sokalsneath", "sqeuclidean", "yule"]
    if algorithm == "auto":
        
         metricOptions = ["chebyshev", "cityblock", "euclidean", "infinity", "l1", "l2", "manhattan",
                          "minkowski", "p"]
       
    elif algorithm == "ball_tree":
        
         exclusions = ["correlation", "cosine", "jensenshannon", "kulsinski", "nan_euclidean", "sqeuclidean",
                       "yule"]
         metricOptions = [m for m in allMetricOptions if m not in exclusions]
       
    elif algorithm == "kd_tree":
        
         metricOptions = ["chebyshev", "cityblock", "euclidean", "infinity", "l1", "l2", "manhattan",
                          "minkowski", "p"]
         
         if p < 1:
            metricOptions.remove("minkowski")
         
    elif algorithm == "brute":
         
         exclusions = ["infinity", "jensenshannon", "p"]
         metricOptions = [m for m in allMetricOptions if m not in exclusions]
        
    metric = st.selectbox(label = "metric",
                          options = metricOptions,
                          index = 1,
                          on_change = setStage,
                          args = [6])
    
    params = {}
    params["n_neighbors"] = nNeighbors
    params["weights"] = weights
    params["algorithm"] = algorithm
    params["leaf_size"] = leafSize
    params["p"] = p
    params["metric"] = metric
    
    return params
    
def setRFCParams():
    "Set the random forest classifier parameters."
    
    nEstimators = st.number_input(label = "n_estimators",
                                  min_value = 1,
                                  value = 100,
                                  step = 1,
                                  on_change = setStage,
                                  args = [6])
    
    criterion = st.selectbox(label = "criterion",
                             options = ["gini", "entropy", "log_loss"],
                             index = 0,
                             on_change = setStage,
                             args = [6])
    
    maxDepth = st.number_input(label = "max_depth",
                                  min_value = 1,
                                  value = None,
                                  step = 1,
                                  on_change = setStage,
                                  args = [6],
                                  placeholder = "None")
    
    minSamplesSplit = st.number_input(label = "min_samples_split",
                                      min_value = 1,
                                      value = 2,
                                      step = 1,
                                      on_change = setStage,
                                      args = [6])
    
    minSamplesLeaf = st.number_input(label = "min_samples_leaf",
                                      min_value = 1,
                                      value = 1,
                                      step = 1,
                                      on_change = setStage,
                                      args = [6])
    
    minWeightFractionLeaf = st.number_input(label = "min_weight_fraction_leaf",
                                            min_value = 0.0,
                                            max_value = 1.0,
                                            value = 0.0,
                                            step = 0.1,
                                            format = "%.1f",
                                            on_change = setStage,
                                            args = [6])
    
    maxFeatures = st.selectbox(label = "max_features",
                               options = ["sqrt", "log2", None],
                               index = 0,
                               on_change = setStage,
                               args = [6])
    
    maxLeafNodes = st.number_input(label = "max_leaf_nodes",
                                   min_value = 1,
                                   value = None,
                                   step = 1,
                                   on_change = setStage,
                                   args = [6],
                                   placeholder = "None")
    
    minImpurityDecrease = st.number_input(label = "min_impurity_decrease",
                                          min_value = 0.0,
                                          value = 0.0,
                                          step = 0.1,
                                          format = "%.1f",
                                          on_change = setStage,
                                          args = [6])
    
    bootstrap = st.selectbox(label = "boostrap",
                             options = [True, False],
                             index = 0,
                             on_change = setStage,
                             args = [6])
    
    if bootstrap is True:
       oobScore = st.selectbox(label = "oob_score",
                               options = [True, False],
                               index = 1,
                               on_change = setStage,
                               args = [6])
    
    classWeight = st.selectbox(label = "class_weight",
                               options = ["balanced", "balanced_subsample", None],
                               index = 2,
                               on_change = setStage,
                               args = [6])
    
    ccpAlpha = st.number_input(label = "ccp_alpha",
                               min_value = 0.0,
                               value = 0.0,
                               step = 0.1,
                               format = "%.1f",
                               on_change = setStage,
                               args = [6])
    
    if bootstrap is True:
       maxSamples = st.number_input(label = "max_samples",
                                    min_value = 1,
                                    value = None,
                                    step = 1,
                                    on_change = setStage,
                                    args = [6],
                                    placeholder = "None")
    
    params = {}
    params["n_estimators"] = nEstimators
    params["criterion"] = criterion
    params["max_depth"] = maxDepth
    params["min_samples_split"] = minSamplesSplit
    params["min_samples_leaf"] = minSamplesLeaf
    params["min_weight_fraction_leaf"] = minWeightFractionLeaf
    params["max_features"] = maxFeatures
    params["max_leaf_nodes"] = maxLeafNodes
    params["min_impurity_decrease"] = minImpurityDecrease
    params["bootstrap"] = bootstrap
    params["class_weight"] = classWeight
    params["ccp_alpha"] = ccpAlpha
    
    if bootstrap is True:
        
       params["oob_score"] = oobScore
       params["max_samples"] = maxSamples
    
    return params

def setModelParams(modelName):
    
    if modelName == "Logistic Regression":
       params = setLRParams()
    elif modelName == "k-Nearest Neighbors Classifier":
         params = setNNCParams()
    elif modelName == "Random Forest Classifier":
         params = setRFCParams()
         
    return params   
    