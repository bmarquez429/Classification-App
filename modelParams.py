"""
         File: modelParams.py
 Date Created: February 22, 2024
Date Modified: February 27, 2024
---------------------------------------------------------
This script is imported by scikit-learnClassification.py.
---------------------------------------------------------
"""

import sklearn.gaussian_process.kernels as sgpk
import streamlit as st

def setStage(i):
    "Set the session state."
    
    st.session_state.stage = i
    
def setDTCParams():
    "Set the decision tree classifier parameters."
    
    criterion = st.selectbox(label = "criterion",
                             options = ["gini", "entropy", "log_loss"],
                             index = 0,
                             key = "dtc_criterion",
                             on_change = setStage,
                             args = [6])
    
    splitter = st.selectbox(label = "splitter",
                            options = ["best", "random"],
                            index = 0,
                            on_change = setStage,
                            args = [6])
    
    maxDepth = st.number_input(label = "max_depth",
                                  min_value = 1,
                                  value = None,
                                  step = 1,
                                  key = "dtc_max_depth", 
                                  on_change = setStage,
                                  args = [6],
                                  placeholder = "None")
    
    minSamplesSplit = st.number_input(label = "min_samples_split",
                                      min_value = 2,
                                      value = 2,
                                      step = 1,
                                      key = "dtc_min_samples_split", 
                                      on_change = setStage,
                                      args = [6])
    
    minSamplesLeaf = st.number_input(label = "min_samples_leaf",
                                      min_value = 1,
                                      value = 1,
                                      step = 1,
                                      key = "dtc_min_samples_leaf", 
                                      on_change = setStage,
                                      args = [6])
    
    minWeightFractionLeaf = st.number_input(label = "min_weight_fraction_leaf",
                                            min_value = 0.0,
                                            max_value = 0.5,
                                            value = 0.0,
                                            step = 0.1,
                                            format = "%.1f",
                                            key = "dtc_min_weight_fraction_leaf", 
                                            on_change = setStage,
                                            args = [6])
    
    maxFeatures = st.selectbox(label = "max_features",
                               options = ["sqrt", "log2", None],
                               index = 0,
                               key = "dtc_max_features",
                               on_change = setStage,
                               args = [6])
    
    maxLeafNodes = st.number_input(label = "max_leaf_nodes",
                                   min_value = 2,
                                   value = None,
                                   step = 1,
                                   key = "dtc_max_leaf_nodes",
                                   on_change = setStage,
                                   args = [6],
                                   placeholder = "None")
    
    minImpurityDecrease = st.number_input(label = "min_impurity_decrease",
                                          min_value = 0.0,
                                          value = 0.0,
                                          step = 0.1,
                                          format = "%.1f",
                                          key = "dtc_min_impurity_decrease", 
                                          on_change = setStage,
                                          args = [6])
    
    classWeight = st.selectbox(label = "class_weight",
                               options = ["balanced", "balanced_subsample", None],
                               index = 2,
                               key = "dtc_class_weight", 
                               on_change = setStage,
                               args = [6])
    
    ccpAlpha = st.number_input(label = "ccp_alpha",
                               min_value = 0.0,
                               value = 0.0,
                               step = 0.1,
                               format = "%.1f",
                               key = "dtc_ccp_alpha", 
                               on_change = setStage,
                               args = [6])
    
    params = {}
    params["criterion"] = criterion
    params["splitter"] = splitter
    params["max_depth"] = maxDepth
    params["min_samples_split"] = minSamplesSplit
    params["min_samples_leaf"] = minSamplesLeaf
    params["min_weight_fraction_leaf"] = minWeightFractionLeaf
    params["max_features"] = maxFeatures
    params["max_leaf_nodes"] = maxLeafNodes
    params["min_impurity_decrease"] = minImpurityDecrease
    params["class_weight"] = classWeight
    params["ccp_alpha"] = ccpAlpha
        
    return params
    
def setGPCParams():
    "Set the Gaussian process classifier parameters."
    
    def getValueAndBounds(kernelName, valueName, minValue = None):
        
        value = st.number_input(label = kernelName + " Kernel: " + valueName,
                                  min_value = minValue,
                                  value = 1.0,
                                  step = 0.1,
                                  format = "%.1f",
                                  on_change = setStage,
                                  args = [6])
        
        bounds = st.selectbox(label = kernelName + ": " + valueName + "_bounds",
                              options = ["fixed", "pair of floats"],
                              index = 1,
                              on_change = setStage,
                              args = [6])
        
        if bounds == "pair of floats":
        
           float1 = st.number_input(label = "Input first float for " + valueName + "_bounds.",
                                    min_value = 1e-6,
                                    value = 1e-5,
                                    step = 1e-6,
                                    format = "%.6f",
                                    on_change = setStage,
                                    args = [6])
          
           float2 = st.number_input(label = "Input second float for " + valueName + "_bounds.",
                                    min_value = 1e-6,
                                    value = 1e5,
                                    step = 1e-6,
                                    format = "%.6f",
                                    on_change = setStage,
                                    args = [6])
           
           bounds = (float1, float2)

        return value, bounds
        
    kernelOptions = ["Constant", "Dot Product", "Matern", "Pairwise", "RBF", "Rational Quadratic", 
                     "White", None]
    
    kernel = st.selectbox(label = "kernel",
                          options = kernelOptions,
                          index = 7,
                          on_change = setStage,
                          args = [6])
    
    if kernel == "Constant":    
        
       value, bounds = getValueAndBounds(kernelName = kernel, 
                                         valueName = "constant_value")         
       kernel = sgpk.ConstantKernel(constant_value = value,
                                    constant_value_bounds = bounds)
        
    elif kernel == "Dot Product":
        
         value, bounds = getValueAndBounds(kernelName = kernel, 
                                           valueName = "sigma_0",
                                           minValue = 0.0)  
         kernel = sgpk.DotProduct(sigma_0 = value,
                                  sigma_0_bounds = bounds) 
 
    elif kernel == "Matern":
         
         value, bounds = getValueAndBounds(kernelName = kernel, 
                                           valueName = "length_scale")   
         nu = st.number_input(label = "nu",
                              value = 1.5,
                              step = 0.1,
                              format = "%.1f",
                              on_change = setStage,
                              args = [6])
         kernel = sgpk.Matern(length_scale = value,
                              length_scale_bounds = bounds,
                              nu = nu)
          
    elif kernel == "Pairwise":
         
         value, bounds = getValueAndBounds(kernelName = kernel, 
                                           valueName = "gamma")   
         metricOptions = ["additive_chi2", "chi2", "cosine", "laplacian", "linear", "poly", 
                          "polynomial", "rbf", "sigmoid", ]
         metric = st.selectbox(label = "metric",
                               options = metricOptions,
                               index = 4,
                               on_change = setStage,
                               args = [6])
         kernel = sgpk.PairwiseKernel(gamma = value,
                                      gamma_bounds = bounds,
                                      metric = metric)
         
    elif kernel == "RBF":
        
         value, bounds = getValueAndBounds(kernelName = kernel, 
                                           valueName = "length_scale") 
         kernel = sgpk.RBF(length_scale = value,
                           length_scale_bounds = bounds)
         
    elif kernel == "Rational Quadratic":
        
         value1, bounds1 = getValueAndBounds(kernelName = kernel, 
                                           valueName = "length_scale",
                                           minValue = 0.1)  
         value2, bounds2 = getValueAndBounds(kernelName = kernel, 
                                             valueName = "alpha",
                                             minValue = 0.1) 
         kernel = sgpk.RationalQuadratic(length_scale = value1,
                                         alpha = value2,
                                         length_scale_bounds = bounds1,
                                         alpha_bounds = bounds2)
         
    elif kernel == "White":
        
         value, bounds = getValueAndBounds(kernelName = kernel, 
                                           valueName = "noise_level") 
         kernel = sgpk.WhiteKernel(noise_level = value,
                                   noise_level_bounds = bounds)
         
    optimizer = st.selectbox(label = "optimizer",
                             options = ["fmin_l_bfgs_b", None],
                             index = 0,
                             on_change = setStage,
                             args = [6])
    
    nRestartsOptimizer = st.number_input(label = "n_restarts_optimizer",
                                         min_value = 0,
                                         value = 0,
                                         step = 1,
                                         on_change = setStage,
                                         args = [6])
    
    maxIterPredict = st.number_input(label = "max_iter_predict",
                                     min_value = 1,
                                     value = 100,
                                     step = 1,
                                     on_change = setStage,
                                     args = [6])
    
    warmStart = st.selectbox(label = "warm_start",
                             options = [True, False],
                             index = 1,
                             on_change = setStage,
                             args = [6])
    
    copyXTrain = st.selectbox(label = "copy_X_train",
                              options = [True, False],
                              index = 0,
                              on_change = setStage,
                              args = [6])
    
    multiClass = st.selectbox(label = "multi_class",
                              options = ["one_vs_rest", "one_vs_one"],
                              index = 0,
                              on_change = setStage,
                              args = [6])
    
    params = {}
    params["kernel"] = kernel
    params["optimizer"] = optimizer
    params["n_restarts_optimizer"] = nRestartsOptimizer
    params["max_iter_predict"] = maxIterPredict
    params["warm_start"] = warmStart
    params["copy_X_train"] = copyXTrain
    params["multi_class"] = multiClass
    
    return params

def setkNNCParams():
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
         index = 7
       
    elif algorithm == "ball_tree":
        
         exclusions = ["correlation", "cosine", "jensenshannon", "kulsinski", "nan_euclidean", "sqeuclidean",
                       "yule"]
         metricOptions = [m for m in allMetricOptions if m not in exclusions]
         index = 12
       
    elif algorithm == "kd_tree":
        
         metricOptions = ["chebyshev", "cityblock", "euclidean", "infinity", "l1", "l2", "manhattan",
                          "minkowski", "p"]
         index = 7
         
         if p < 1:
             
            metricOptions.remove("minkowski")
            index = None
         
    elif algorithm == "brute":
         
         exclusions = ["infinity", "jensenshannon", "p"]
         metricOptions = [m for m in allMetricOptions if m not in exclusions]
         index = 14
        
    metric = st.selectbox(label = "metric",
                          options = metricOptions,
                          index = index,
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
                           key = "lrC",
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
                             key = "rfc_criterion",
                             on_change = setStage,
                             args = [6])
    
    maxDepth = st.number_input(label = "max_depth",
                                  min_value = 1,
                                  value = None,
                                  step = 1,
                                  key = "rfc_max_depth", 
                                  on_change = setStage,
                                  args = [6],
                                  placeholder = "None")
    
    minSamplesSplit = st.number_input(label = "min_samples_split",
                                      min_value = 2,
                                      value = 2,
                                      step = 1,
                                      key = "rfc_min_samples_split", 
                                      on_change = setStage,
                                      args = [6])
    
    minSamplesLeaf = st.number_input(label = "min_samples_leaf",
                                      min_value = 1,
                                      value = 1,
                                      step = 1,
                                      key = "rfc_min_samples_leaf", 
                                      on_change = setStage,
                                      args = [6])
    
    minWeightFractionLeaf = st.number_input(label = "min_weight_fraction_leaf",
                                            min_value = 0.0,
                                            max_value = 0.5,
                                            value = 0.0,
                                            step = 0.1,
                                            format = "%.1f",
                                            key = "rfc_min_weight_fraction_leaf", 
                                            on_change = setStage,
                                            args = [6])
    
    maxFeatures = st.selectbox(label = "max_features",
                               options = ["sqrt", "log2", None],
                               index = 0,
                               key = "rfc_max_features",
                               on_change = setStage,
                               args = [6])
    
    maxLeafNodes = st.number_input(label = "max_leaf_nodes",
                                   min_value = 2,
                                   value = None,
                                   step = 1,
                                   key = "rfc_max_leaf_nodes",
                                   on_change = setStage,
                                   args = [6],
                                   placeholder = "None")
    
    minImpurityDecrease = st.number_input(label = "min_impurity_decrease",
                                          min_value = 0.0,
                                          value = 0.0,
                                          step = 0.1,
                                          format = "%.1f",
                                          key = "rfc_min_impurity_decrease", 
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
                               key = "rfc_class_weight", 
                               on_change = setStage,
                               args = [6])
    
    ccpAlpha = st.number_input(label = "ccp_alpha",
                               min_value = 0.0,
                               value = 0.0,
                               step = 0.1,
                               format = "%.1f",
                               key = "rfc_ccp_alpha", 
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

def setSVCParams():
    "Set the support vector classifier parameters."
    
    C = st.number_input(label = "C",
                        value = 1.0,
                        step = 0.1,
                        format = "%.1f",
                        key = "svcC",
                        on_change = setStage,
                        args = [6])
    
    kernel = st.selectbox(label = "kernel",
                          options = ["linear", "poly", "rbf", "sigmoid", "precomputed"],
                          index = 2,
                          on_change = setStage,
                          args = [6])
    
    if kernel == "poly":
       degree = st.number_input(label = "degree",
                                min_value = 1,
                                value = 3,
                                step = 1,
                                on_change = setStage,
                                args = [6])
    
    gamma = st.selectbox(label = "gamma",
                          options = ["scale", "auto", "float"],
                          index = 0,
                          on_change = setStage,
                          args = [6])
    
    if gamma == "float":
       gamma = st.number_input(label = "Input float for gamma.",
                               value = None,
                               step = 0.1,
                               format = "%.1f",
                               on_change = setStage,
                               args = [6])
        
    coef0 = st.number_input(label = "coef0",
                            value = 0.0,
                            step = 0.1,
                            format = "%.1f",
                            on_change = setStage,
                            args = [6])
    
    shrinking = st.selectbox(label = "shrinking",
                             options = [True, False],
                             index = 0,
                             on_change = setStage,
                             args = [6])
    
    probability = st.selectbox(label = "probability",
                               options = [True, False],
                               index = 1,
                               on_change = setStage,
                               args = [6])
    
    tol = st.number_input(label = "tol",
                          value = 1e-3,
                          step = 1e-4,
                          format = "%.4f",
                          on_change = setStage,
                          args = [6])
    
    cacheSize = st.number_input(label = "cache_size",
                                value = 200.0,
                                step = 0.1,
                                format = "%.1f",
                                on_change = setStage,
                                args = [6])
    
    classWeight = st.selectbox(label = "class_weight",
                               options = ["balanced", None],
                               index = 1,
                               on_change = setStage,
                               args = [6])
    
    maxIter = st.number_input(label = "max_iter",
                              min_value = -1,
                              value = -1,
                              step = 1,
                              on_change = setStage,
                              args = [6])
    
    decisionFunctionShape = st.selectbox(label = "decision_function_shape",
                                         options = ["ovo", "ovr"],
                                         index = 1,
                                         on_change = setStage,
                                         args = [6])
    
    breakTies = st.selectbox(label = "break_ties",
                             options = [True, False],
                             index = 1,
                             on_change = setStage,
                             args = [6])
    
    params = {}
    params["C"] = C
    params["kernel"] = kernel
    params["gamma"] = gamma
    params["coef0"] = coef0
    params["shrinking"] = shrinking
    params["probability"] = probability
    params["tol"] = tol
    params["cache_size"] = cacheSize
    params["class_weight"] = classWeight
    params["max_iter"] = maxIter
    params["decision_function_shape"] = decisionFunctionShape
    params["break_ties"] = breakTies
    
    if kernel == "poly":
       params["degree"] = degree
       
    return params

def setModelParams(modelName):
    "Set the model parameters."
    
    if modelName == "Decision Tree Classifier":
       params = setDTCParams() 
    elif modelName == "Gaussian Process Classifier":
         params = setGPCParams()
    elif modelName == "k-Nearest Neighbors Classifier":
         params = setkNNCParams()
    elif modelName == "Logistic Regression":
       params = setLRParams()
    elif modelName == "Random Forest Classifier":
         params = setRFCParams()
    elif modelName == "Support Vector Classifier":
         params = setSVCParams()
         
    return params   
    