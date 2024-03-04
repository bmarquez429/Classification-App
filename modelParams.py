"""
         File: modelParams.py
 Date Created: February 22, 2024
Date Modified: March 4, 2024
---------------------------------------------------------
This script is imported by scikit-learnClassification.py.
---------------------------------------------------------
"""

from helperFunctions import setStage
import math
import sklearn.gaussian_process.kernels as sgpk
import streamlit as st
    
def setDTCParams():
    "Set the decision tree classifier parameters."
    
    #----------
    # criterion
    #----------
    
    criterion = st.selectbox(label = "criterion",
                             options = ["gini", "entropy", "log_loss"],
                             index = 0,
                             key = "dtc_criterion",
                             on_change = setStage,
                             args = [6])
    
    #---------
    # splitter
    #---------
    
    splitter = st.selectbox(label = "splitter",
                            options = ["best", "random"],
                            index = 0,
                            on_change = setStage,
                            args = [6])
    
    #----------
    # max_depth
    #----------
    
    maxDepth = st.number_input(label = "max_depth",
                               value = None,
                               step = 1.0,
                               format = "%f",
                               key = "dtc_max_depth", 
                               on_change = setStage,
                               args = [6],
                               placeholder = "None")
    disabledMaxDepth = False
    
    if maxDepth is not None:
    
       maxDepthDelta = maxDepth - math.floor(maxDepth)
    
       if maxDepth < 0 or (maxDepth > 1 and maxDepthDelta > 0):
        
          st.markdown(":red[For `max_depth`, please input an integer greater than or equal to 1.]")
          disabledMaxDepth = True
       
       else:
          if maxDepth >= 1 and maxDepthDelta == 0:
             maxDepth = int(maxDepth)
    
    #------------------
    # min_samples_split
    #------------------
    
    minSamplesSplit = st.number_input(label = "min_samples_split",
                                      value = 2.0,
                                      step = 1.0,
                                      format = "%f",
                                      key = "dtc_min_samples_split", 
                                      on_change = setStage,
                                      args = [6])
    disabledMinSamplesSplit = False
    
    minSamplesSplitDelta = minSamplesSplit - math.floor(minSamplesSplit)
    
    if minSamplesSplit <= 0 or minSamplesSplit == 1 or (minSamplesSplit > 1 and minSamplesSplitDelta > 0):
        
       st.markdown(":red[For `min_samples_split`, please input an integer greater than or equal to 2 or a float \
                 strictly between 0 and 1.]")
       disabledMinSamplesSplit = True
       
    else:
       if minSamplesSplit > 1 and minSamplesSplitDelta == 0:
          minSamplesSplit = int(minSamplesSplit)
         
    #-----------------
    # min_samples_leaf
    #-----------------
    
    minSamplesLeaf = st.number_input(label = "min_samples_leaf",
                                      value = 1.0,
                                      step = 1.0,
                                      format = "%f",
                                      key = "dtc_min_samples_leaf", 
                                      on_change = setStage,
                                      args = [6])
    disabledMinSamplesLeaf = False
    
    minSamplesLeafDelta = minSamplesLeaf - math.floor(minSamplesLeaf)
    
    if minSamplesLeaf <= 0 or (minSamplesLeaf > 1 and minSamplesLeafDelta > 0):
        
       st.markdown(":red[For `min_samples_leaf`, please input an integer greater than or equal to 1 or a float \
                    strictly between 0 and 1.]")
       disabledMinSamplesLeaf = True
       
    else:
       if minSamplesLeaf >= 1 and minSamplesLeafDelta == 0:
          minSamplesLeaf = int(minSamplesLeaf)
    
    #-------------------------
    # min_weight_fraction_leaf
    #-------------------------
    
    minWeightFractionLeaf = st.number_input(label = "min_weight_fraction_leaf",
                                            value = 0.0,
                                            step = 0.1,
                                            format = "%.1f",
                                            key = "dtc_min_weight_fraction_leaf", 
                                            on_change = setStage,
                                            args = [6])
    disabledMinWeightFractionLeaf = False
    
    if minWeightFractionLeaf < 0 or minWeightFractionLeaf > 0.5:
        
       st.markdown(":red[For `min_weight_fraction_leaf`, please input a float between 0 and 0.5 inclusive.]")
       disabledMinWeightFractionLeaf = True
       
    #-------------
    # max_features
    #-------------
    
    maxFeatures = st.selectbox(label = "max_features",
                               options = ["int or float", "sqrt", "log2", None],
                               index = 3,
                               key = "dtc_max_features1",
                               on_change = setStage,
                               args = [6])
    disabledMaxFeatures = False
    
    if maxFeatures == "int or float":
        
       maxFeatures = st.number_input(label = "Input int or float for max_features.",
                                     value = None,
                                     step = 1.0,
                                     format = "%f",
                                     key = "dtc_max_features2",
                                     on_change = setStage,
                                     args = [6])
       
       if maxFeatures is not None:
           
          maxFeaturesDelta = maxFeatures - math.floor(maxFeatures)
       
          if maxFeatures <= 0 or (maxFeatures > 1 and maxFeaturesDelta > 0):
           
             st.markdown(":red[For `max_features`, please input an integer greater than or equal to 1 or a float \
                          strictly between 0 and 1.]")
             disabledMaxFeatures = True
          
          else:
             if maxFeatures >= 1 and maxFeaturesDelta == 0:
                maxFeatures = int(maxFeatures)
                 
    #---------------
    # max_leaf_nodes
    #---------------
     
    maxLeafNodes = st.number_input(label = "max_leaf_nodes",
                                   value = None,
                                   step = 1.0,
                                   format = "%f",
                                   key = "dtc_max_leaf_nodes",
                                   on_change = setStage,
                                   args = [6],
                                   placeholder = "None")
    disabledMaxLeafNodes = False
    
    if maxLeafNodes is not None:
        
       maxLeafNodesDelta = maxLeafNodes - math.floor(maxLeafNodes)
        
       if maxLeafNodes < 2 or (maxLeafNodes > 2 and maxLeafNodesDelta > 0):
            
          st.markdown(":red[For `max_leaf_nodes`, please input an integer greater than or equal to 2.]")
          disabledMaxLeafNodes = True
          
       else:
          maxLeafNodes = int(maxLeafNodes)
          
    #----------------------
    # min_impurity_decrease
    #----------------------
    
    minImpurityDecrease = st.number_input(label = "min_impurity_decrease",
                                          value = 0.0,
                                          step = 0.1,
                                          format = "%.1f",
                                          key = "dtc_min_impurity_decrease", 
                                          on_change = setStage,
                                          args = [6])
    disabledMinImpurityDecrease = False
    
    if minImpurityDecrease < 0:
            
       st.markdown(":red[For `min_impurity_decrease`, please input a float greater than or equal to 0.]")
       disabledMinImpurityDecrease = True
       
    #-------------
    # class_weight
    #-------------
    
    classWeight = st.selectbox(label = "class_weight",
                               options = ["balanced", None],
                               index = 1,
                               key = "dtc_class_weight", 
                               on_change = setStage,
                               args = [6])
    
    #----------
    # ccp_alpha
    #----------
    
    ccpAlpha = st.number_input(label = "ccp_alpha",
                               value = 0.0,
                               step = 0.1,
                               format = "%.1f",
                               key = "dtc_ccp_alpha", 
                               on_change = setStage,
                               args = [6])
    disabledCcpAlpha = False
    
    if ccpAlpha < 0:
            
       st.markdown(":red[For `ccp_alpha`, please input a float greater than or equal to 0.]")
       disabledCcpAlpha = True
        
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
    
    disabled = disabledMaxDepth + disabledMinSamplesSplit + disabledMinSamplesLeaf + \
               disabledMinWeightFractionLeaf + disabledMaxFeatures + disabledMaxLeafNodes + \
               disabledMinImpurityDecrease + disabledCcpAlpha
        
    return params, disabled
    
def setGPCParams():
    "Set the Gaussian process classifier parameters."
    
    #-------
    # kernel
    #-------
    
    def getValueAndBounds(kernelName, valueName):
        
        value = st.number_input(label = kernelName + " Kernel: " + valueName,
                                value = 1.0,
                                step = 0.1,
                                format = "%.1f",
                                on_change = setStage,
                                args = [6])
        dictDisabled[kernelName + " Kernel: " + valueName] = False
        
        if kernelName in ["Dot Product", "Pairwise"]:
            
           condition = value < 0
           stringPart = " or equal to "
           
        else:
            
           condition = value <= 0 
           stringPart = " " 
        
        if condition:
             
           st.markdown(":red[For] `" + kernelName + " Kernel: " + valueName + "`:red[, please input a float \
                        greater than" + stringPart + "0.]")
           dictDisabled[kernelName + " Kernel: " + valueName] = True
          
        bounds = st.selectbox(label = kernelName + " Kernel: " + valueName + "_bounds",
                              options = ["fixed", "pair of floats"],
                              index = 1,
                              on_change = setStage,
                              args = [6])
        
        if bounds == "pair of floats":
        
           float1 = st.number_input(label = "Input first float for " + valueName + "_bounds.",
                                    value = 1e-5,
                                    step = 1e-6,
                                    format = "%.6f",
                                    on_change = setStage,
                                    args = [6])
           dictDisabled[kernelName + " Kernel: " + valueName + "_bounds_float1"] = False
           
           if float1 < 0:
                
              st.markdown(":red[For the first float for ] `" + valueName + "_bounds`:red[, please input a float \
                           greater than or equal to 0.]")
              dictDisabled[kernelName + " Kernel: " + valueName + "_bounds_float1"] = True
           
           float2 = st.number_input(label = "Input second float for " + valueName + "_bounds.",
                                    value = 1e5,
                                    step = 1e-6,
                                    format = "%.6f",
                                    on_change = setStage,
                                    args = [6])
           dictDisabled[kernelName + " Kernel: " + valueName + "_bounds_float2"] = False
           
           if float2 < 0:
                
              st.markdown(":red[For the second float for ] `" + valueName + "_bounds`:red[, please input a float \
                           greater than or equal to 0.]")
              dictDisabled[kernelName + " Kernel: " + valueName + "_bounds_float2"] = True 
              
           elif float2 < float1:
               
                st.markdown(":red[For the second float for ] `" + valueName + "_bounds`:red[, please input a float \
                             greater than the first float.]")
                dictDisabled[kernelName + " Kernel: " + valueName + "_bounds_float2"] = True 
              
           
           bounds = (float1, float2)

        return value, bounds
        
    kernelOptions = ["Constant", "Dot Product", "Matern", "Pairwise", "RBF", "Rational Quadratic", 
                     "White", None]
    
    kernel = st.selectbox(label = "kernel",
                          options = kernelOptions,
                          index = 7,
                          on_change = setStage,
                          args = [6])
    
    dictDisabled = {}
    
    if kernel == "Constant":    
        
       value, bounds = getValueAndBounds(kernelName = kernel, 
                                         valueName = "constant_value")         
       kernel = sgpk.ConstantKernel(constant_value = value,
                                    constant_value_bounds = bounds)
        
    elif kernel == "Dot Product":
        
         value, bounds = getValueAndBounds(kernelName = kernel, 
                                           valueName = "sigma_0")  
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
         dictDisabled["nu"] = False 
                 
         if nu <= 0:
                
            st.markdown(":red[For `nu`, please input a float greater than 0.]")
            dictDisabled["nu"] = True
         
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
                                           valueName = "length_scale")  
         value2, bounds2 = getValueAndBounds(kernelName = kernel, 
                                             valueName = "alpha") 
         kernel = sgpk.RationalQuadratic(length_scale = value1,
                                         alpha = value2,
                                         length_scale_bounds = bounds1,
                                         alpha_bounds = bounds2)
         
    elif kernel == "White":
        
         value, bounds = getValueAndBounds(kernelName = kernel, 
                                           valueName = "noise_level") 
         kernel = sgpk.WhiteKernel(noise_level = value,
                                   noise_level_bounds = bounds)
    
    #----------
    # optimizer
    #----------
          
    optimizer = st.selectbox(label = "optimizer",
                             options = ["fmin_l_bfgs_b", None],
                             index = 0,
                             on_change = setStage,
                             args = [6])
    
    #---------------------
    # n_restarts_optimizer
    #---------------------
    
    nRestartsOptimizer = st.number_input(label = "n_restarts_optimizer",
                                         value = 0.0,
                                         step = 1.0,
                                         format = "%f",
                                         on_change = setStage,
                                         args = [6])
    disabledNRestartsOptimizer = False
    
    nRestartsOptimizerDelta = nRestartsOptimizer - math.floor(nRestartsOptimizer)
     
    if nRestartsOptimizer < 0 or (nRestartsOptimizer > 0 and nRestartsOptimizerDelta > 0):
         
       st.markdown(":red[For `n_restarts_optimizer`, please input an integer greater than or equal to 0.]")
       disabledNRestartsOptimizer = True
       
    else:
       nRestartsOptimizer = int(nRestartsOptimizer)
    
    #-----------------
    # max_iter_predict
    #-----------------
    
    maxIterPredict = st.number_input(label = "max_iter_predict",
                                     value = 100.0,
                                     step = 1.0,
                                     format = "%f",
                                     on_change = setStage,
                                     args = [6])
    disabledMaxIterPredict = False
    
    maxIterPredictDelta = maxIterPredict - math.floor(maxIterPredict)
     
    if maxIterPredict <= 0 or (maxIterPredict > 0 and maxIterPredictDelta > 0):
         
       st.markdown(":red[For `max_iter_predict`, please input an integer greater than or equal to 1.]")
       disabledMaxIterPredict = True
       
    else:
       maxIterPredict = int(maxIterPredict)
        
    #-------------
    # copy_X_train
    #-------------
    
    copyXTrain = st.selectbox(label = "copy_X_train",
                              options = [True, False],
                              index = 0,
                              on_change = setStage,
                              args = [6])
    
    #------------
    # multi_class
    #------------
    
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
    params["copy_X_train"] = copyXTrain
    params["multi_class"] = multiClass
    
    disabled = disabledNRestartsOptimizer + disabledMaxIterPredict
    keys = list(dictDisabled.keys())
    
    if len(keys) > 0:
    
       for key in keys:
           disabled += dictDisabled[key]
    
    return params, disabled

def setkNNCParams():
    "Set the nearest neighbors classifier parameters."
    
    #-----------
    # n_neigbors
    #-----------
    
    nNeighbors = st.number_input(label = "n_neighbors",
                                 value = 5.0,
                                 step = 1.0,
                                 format = "%f",
                                 on_change = setStage,
                                 args = [6])
    disabledNNeighbors = False
    
    nNeighborsDelta = nNeighbors - math.floor(nNeighbors)
     
    if nNeighbors <= 0 or (nNeighbors > 0 and nNeighborsDelta > 0):
         
       st.markdown(":red[For `n_neighbors`, please input an integer greater than or equal to 1.]")
       disabledNNeighbors = True
       
    else:
       nNeighbors = int(nNeighbors)
    
    #--------
    # weights
    #--------
    
    weights = st.selectbox(label = "weights",
                           options = ["uniform", "distance", None],
                           index = 0,
                           on_change = setStage,
                           args = [6])
    
    #----------
    # algorithm
    #----------
    
    algorithm = st.selectbox(label = "algorithm",
                             options = ["auto", "ball_tree", "kd_tree", "brute"],
                             index = 0,
                             on_change = setStage,
                             args = [6])
    
    #----------
    # leaf_size
    #----------
    
    leafSize = st.number_input(label = "leaf_size",
                               value = 30.0,
                               step = 1.0,
                               format = "%f",
                               on_change = setStage,
                               args = [6])
    disabledLeafSize = False
    
    leafSizeDelta = leafSize - math.floor(leafSize)
     
    if leafSize <= 0 or (leafSize > 0 and leafSizeDelta > 0):
         
       st.markdown(":red[For `leaf_size`, please input an integer greater than or equal to 1.]")
       disabledLeafSize = True
       
    else:
       leafSize = int(leafSize)
    
    #--
    # p
    #--
    
    p = st.number_input(label = "p",
                        value = 2.0,
                        step = 0.1,
                        format = "%.1f",
                        on_change = setStage,
                        args = [6])
    disabledP = False
    
    if p < 1:
         
        st.markdown(":red[For `p`, please input a float greater than or equal 1.]")
        disabledP = True
    
    #-------
    # metric
    #-------
    
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
    
    disabled = disabledNNeighbors + disabledLeafSize + disabledP
    
    return params, disabled

def setLRParams():
    "Set the logistic regression parameters."
    
    #--------
    # penalty
    #--------
    
    penalty = st.selectbox(label = "penalty",
                           options = [None, "l1", "l2", "elasticnet"],
                           index = 2,
                           on_change = setStage,
                           args = [6])
    
    #----
    # tol
    #----
    
    tol = st.number_input(label = "tol",
                          value = 1e-4,
                          step = 1e-5,
                          format = "%.5f",
                          on_change = setStage,
                          args = [6])
    disabledTol = False
    
    if tol < 0:
         
        st.markdown(":red[For `tol`, please input a float greater than or equal 0.]")
        disabledTol = True
    
    #--
    # C
    #--
    
    if penalty is not None:
        
       C = st.number_input(label = "C",
                           value = 1.0,
                           step = 0.1,
                           format = "%.1f",
                           key = "lrC",
                           on_change = setStage,
                           args = [6])
       disabledC = False
       
       if C <= 0:
            
           st.markdown(":red[For `C`, please input a float greater than 0.]")
           disabledC = True
       
    #--------------
    # fit_intercept
    #--------------
    
    fitIntercept = st.selectbox(label = "fit_intercept",
                                options = [True, False],
                                index = 0,
                                on_change = setStage,
                                args = [6])
    
    #------------------
    # intercept_scaling
    #------------------
    
    interceptScaling = st.number_input(label = "intercept_scaling",
                                       value = 1.0,
                                       step = 0.1,
                                       format = "%.1f",
                                       on_change = setStage,
                                       args = [6])
    disabledInterceptScaling = False
       
    if interceptScaling <= 0:
            
       st.markdown(":red[For `intercept_scaling`, please input a float greater than 0.]")
       disabledInterceptScaling = True
       
    #-------------
    # class_weight
    #-------------
    
    classWeight = st.selectbox(label = "class_weight",
                               options = ["balanced", None],
                               index = 1,
                               key = "lr_class_weight", 
                               on_change = setStage,
                               args = [6])
    
    #-------
    # solver
    #-------
    
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
    
    #-----
    # dual
    #-----
    
    if penalty == "l2" and solver == "liblinear":
       dual = st.selectbox(label = "dual",
                           options = [True, False],
                           index = 1,
                           on_change = setStage,
                           args = [6])
    
    #---------
    # max_iter
    #---------
    
    maxIter = st.number_input(label = "max_iter",
                              value = 100.0,
                              step = 1.0,
                              format = "%f",
                              on_change = setStage,
                              args = [6])
    disabledMaxIter = False
    
    maxIterDelta = maxIter - math.floor(maxIter)
     
    if maxIter <= 0 or (maxIter > 0 and maxIterDelta > 0):
         
       st.markdown(":red[For `max_iter`, please input an integer greater than or equal to 1.]")
       disabledMaxIter = True
       
    else:
       maxIter = int(maxIter)
    
    #------------
    # multi_class
    #------------
    
    multiClassOptions = ["auto", "multinomial", "ovr"]
    if solver == "liblinear":
       multiClassOptions.remove("multinomial")

    multiClass = st.selectbox(label = "multi_class",
                              options = multiClassOptions,
                              index = 0,
                              on_change = setStage,
                              args = [6])
    
    #---------
    # l1_ratio
    #---------

    if penalty == "elasticnet":   
        
       l1Ratio = st.number_input(label = "l1_ratio",
                                 value = None,
                                 step = 0.01,
                                 format = "%.2f",
                                 on_change = setStage,
                                 args = [6],
                                 placeholder = "None")
       disabledL1Ratio = False
       
       if l1Ratio is not None:
           
          if l1Ratio < 0 or l1Ratio > 1:
               
             st.markdown(":red[For `l1_ratio`, please input a float between 0 and 1 inclusive.]")
             disabledL1Ratio = True
       
    params = {}
    params["penalty"] = penalty
    params["tol"] = tol
    params["fit_intercept"] = fitIntercept
    params["intercept_scaling"] = interceptScaling
    params["class_weight"] = classWeight
    params["solver"] = solver
    params["max_iter"] = maxIter
    params["multi_class"] = multiClass
    
    disabled = disabledTol + disabledInterceptScaling + disabledMaxIter
    
    if penalty is not None:
      
       params["C"] = C
       disabled += disabledC
    
    if penalty == "l2" and solver == "liblinear":
       params["dual"] = dual
    
    if penalty == "elasticnet":
       
       disabled += disabledL1Ratio
       params["l1_ratio"] = l1Ratio
    
    return params, disabled

def setRFCParams():
    "Set the random forest classifier parameters."
    
    #-------------
    # n_estimators
    #-------------

    nEstimators = st.number_input(label = "n_estimators",
                                  value = 100.0,
                                  step = 1.0,
                                  format = "%f",
                                  on_change = setStage,
                                  args = [6])
    disabledNEstimators = False
    
    nEstimatorsDelta = nEstimators - math.floor(nEstimators)
    
    if nEstimators <= 0 or (nEstimators > 1 and nEstimatorsDelta > 0):
        
       st.markdown(":red[For `n_estimators`, please input an integer greater than or equal to 1.]")
       disabledNEstimators = True
       
    else:
       nEstimators = int(nEstimators)
    
    #----------
    # criterion
    #----------

    criterion = st.selectbox(label = "criterion",
                             options = ["gini", "entropy", "log_loss"],
                             index = 0,
                             key = "rfc_criterion",
                             on_change = setStage,
                             args = [6])
    
    #----------
    # max_depth
    #----------

    maxDepth = st.number_input(label = "max_depth",
                                  value = None,
                                  step = 1.0,
                                  format = "%f",
                                  key = "rfc_max_depth", 
                                  on_change = setStage,
                                  args = [6],
                                  placeholder = "None")
    disabledMaxDepth = False
    
    if maxDepth is not None:
    
       maxDepthDelta = maxDepth - math.floor(maxDepth)
    
       if maxDepth < 0 or (maxDepth > 1 and maxDepthDelta > 0):
        
          st.markdown(":red[For `max_depth`, please input an integer greater than or equal to 1.]")
          disabledMaxDepth = True
       
       else:
          if maxDepth >= 1 and maxDepthDelta == 0:
             maxDepth = int(maxDepth)
    
    #------------------
    # min_samples_split
    #------------------

    minSamplesSplit = st.number_input(label = "min_samples_split",
                                      value = 2.0,
                                      step = 1.0,
                                      format = "%f",
                                      key = "rfc_min_samples_split", 
                                      on_change = setStage,
                                      args = [6])
    disabledMinSamplesSplit = False
    
    minSamplesSplitDelta = minSamplesSplit - math.floor(minSamplesSplit)
    
    if minSamplesSplit <= 0 or minSamplesSplit == 1 or (minSamplesSplit > 1 and minSamplesSplitDelta > 0):
        
       st.markdown(":red[For `min_samples_split`, please input an integer greater than or equal to 2 or a float \
                 strictly between 0 and 1.]")
       disabledMinSamplesSplit = True
       
    else:
       if minSamplesSplit > 1 and minSamplesSplitDelta == 0:
          minSamplesSplit = int(minSamplesSplit)
    
    #-----------------
    # min_samples_leaf
    #-----------------

    minSamplesLeaf = st.number_input(label = "min_samples_leaf",
                                      value = 1.0,
                                      step = 1.0,
                                      format = "%f",
                                      key = "rfc_min_samples_leaf", 
                                      on_change = setStage,
                                      args = [6])
    disabledMinSamplesLeaf = False
    
    minSamplesLeafDelta = minSamplesLeaf - math.floor(minSamplesLeaf)
    
    if minSamplesLeaf <= 0 or (minSamplesLeaf > 1 and minSamplesLeafDelta > 0):
        
       st.markdown(":red[For `min_samples_leaf`, please input an integer greater than or equal to 1 or a float \
                    strictly between 0 and 1.]")
       disabledMinSamplesLeaf = True
       
    else:
       if minSamplesLeaf >= 1 and minSamplesLeafDelta == 0:
          minSamplesLeaf = int(minSamplesLeaf)
    
    #-------------------------
    # min_weight_fraction_leaf
    #-------------------------

    minWeightFractionLeaf = st.number_input(label = "min_weight_fraction_leaf",
                                            value = 0.0,
                                            step = 0.1,
                                            format = "%.1f",
                                            key = "rfc_min_weight_fraction_leaf", 
                                            on_change = setStage,
                                            args = [6])
    disabledMinWeightFractionLeaf = False
    
    if minWeightFractionLeaf < 0 or minWeightFractionLeaf > 0.5:
        
       st.markdown(":red[For `min_weight_fraction_leaf`, please input a float between 0 and 0.5 inclusive.]")
       disabledMinWeightFractionLeaf = True
    
    #-------------
    # max_features
    #-------------

    maxFeatures = st.selectbox(label = "max_features",
                               options = ["int or float", "sqrt", "log2", None],
                               index = 1,
                               key = "rfc_max_features1",
                               on_change = setStage,
                               args = [6])
    disabledMaxFeatures = False
    
    if maxFeatures == "int or float":
        
       maxFeatures = st.number_input(label = "Input int or float for max_features.",
                                     value = None,
                                     step = 1.0,
                                     format = "%f",
                                     key = "rfc_max_features2",
                                     on_change = setStage,
                                     args = [6])
       
       if maxFeatures is not None:
           
          maxFeaturesDelta = maxFeatures - math.floor(maxFeatures)
       
          if maxFeatures <= 0 or (maxFeatures > 1 and maxFeaturesDelta > 0):
           
             st.markdown(":red[For `max_features`, please input an integer greater than or equal to 1 or a float \
                          strictly between 0 and 1.]")
             disabledMaxFeatures = True
          
          else:
             if maxFeatures >= 1 and maxFeaturesDelta == 0:
                maxFeatures = int(maxFeatures)
    
    #---------------
    # max_leaf_nodes
    #---------------

    maxLeafNodes = st.number_input(label = "max_leaf_nodes",
                                   value = None,
                                   step = 1.0,
                                   format = "%f",
                                   key = "rfc_max_leaf_nodes",
                                   on_change = setStage,
                                   args = [6],
                                   placeholder = "None")
    disabledMaxLeafNodes = False
    
    if maxLeafNodes is not None:
        
       maxLeafNodesDelta = maxLeafNodes - math.floor(maxLeafNodes)
        
       if maxLeafNodes < 2 or (maxLeafNodes > 2 and maxLeafNodesDelta > 0):
            
          st.markdown(":red[For `max_leaf_nodes`, please input an integer greater than or equal to 2.]")
          disabledMaxLeafNodes = True
          
       else:
          maxLeafNodes = int(maxLeafNodes)
    
    #----------------------
    # min_impurity_decrease
    #----------------------

    minImpurityDecrease = st.number_input(label = "min_impurity_decrease",
                                          value = 0.0,
                                          step = 0.1,
                                          format = "%.1f",
                                          key = "rfc_min_impurity_decrease", 
                                          on_change = setStage,
                                          args = [6])
    disabledMinImpurityDecrease = False
    
    if minImpurityDecrease < 0:
            
       st.markdown(":red[For `min_impurity_decrease`, please input a float greater than or equal to 0.]")
       disabledMinImpurityDecrease = True
    
    #----------
    # bootstrap
    #----------

    bootstrap = st.selectbox(label = "boostrap",
                             options = [True, False],
                             index = 0,
                             on_change = setStage,
                             args = [6])
     
    #----------
    # oob_score
    #----------

    if bootstrap is True:
       oobScore = st.selectbox(label = "oob_score",
                               options = [True, False],
                               index = 1,
                               on_change = setStage,
                               args = [6])
    
    #-------------
    # class_weight
    #-------------

    classWeight = st.selectbox(label = "class_weight",
                               options = ["balanced", "balanced_subsample", None],
                               index = 2,
                               key = "rfc_class_weight", 
                               on_change = setStage,
                               args = [6])
    
    #----------
    # ccp_alpha
    #----------

    ccpAlpha = st.number_input(label = "ccp_alpha",
                               value = 0.0,
                               step = 0.1,
                               format = "%.1f",
                               key = "rfc_ccp_alpha", 
                               on_change = setStage,
                               args = [6])
    disabledCcpAlpha = False
    
    if ccpAlpha < 0:
            
       st.markdown(":red[For `ccp_alpha`, please input a float greater than or equal to 0.]")
       disabledCcpAlpha = True
    
    #------------
    # max_samples
    #------------

    if bootstrap is True:
        
       maxSamples = st.number_input(label = "max_samples",
                                    value = None,
                                    step = 1.0,
                                    format = "%f",
                                    on_change = setStage,
                                    args = [6],
                                    placeholder = "None")
       disabledMaxSamples = False
       
       if maxSamples is not None:
       
          maxSamplesDelta = maxSamples - math.floor(maxSamples)
       
          if maxSamples <= 0 or (maxSamples > 1 and maxSamplesDelta > 0):
           
             st.markdown(":red[For `max_samples`, please input an integer greater than or equal to 1 or a float \
                          strictly between 0 and 1.]")
             disabledMaxSamples = True
          
          else:
             if maxSamples > 1 and maxSamplesDelta == 0:
                maxSamples = int(maxSamples)
    
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
    
    disabled = disabledNEstimators + disabledMaxDepth + disabledMinSamplesSplit + \
               disabledMinSamplesLeaf + disabledMinWeightFractionLeaf + disabledMaxFeatures + \
               disabledMaxLeafNodes + disabledMinImpurityDecrease + disabledCcpAlpha
               
    if bootstrap is True:
        
       params["oob_score"] = oobScore
       params["max_samples"] = maxSamples
       disabled += disabledMaxSamples
               
    return params, disabled

def setSVCParams():
    "Set the support vector classifier parameters."
    
    #--
    # C
    #--

    C = st.number_input(label = "C",
                        value = 1.0,
                        step = 0.1,
                        format = "%.1f",
                        key = "svcC",
                        on_change = setStage,
                        args = [6])
    disabledC = False
       
    if C <= 0:
            
       st.markdown(":red[For `C`, please input a float greater than 0.]")
       disabledC = True
    
    #-------
    # kernel
    #-------

    kernel = st.selectbox(label = "kernel",
                          options = ["linear", "poly", "rbf", "sigmoid", "precomputed"],
                          index = 2,
                          on_change = setStage,
                          args = [6])
    
    #-------
    # degree
    #-------

    if kernel == "poly":
        
       degree = st.number_input(label = "degree",
                                value = 3.0,
                                step = 1.0,
                                format = "%f",
                                on_change = setStage,
                                args = [6])
       disabledDegree = False
       
       degreeDelta = degree - math.floor(degree)
       
       if degree < 0 or (degree > 1 and degreeDelta > 0):
           
          st.markdown(":red[For `degree`, please input an integer greater than or equal to 0.]")
          disabledDegree = True
          
       else:
          degree = int(degree)
       
    #------
    # gamma
    #------

    gamma = st.selectbox(label = "gamma",
                          options = ["scale", "auto", "float"],
                          index = 0,
                          on_change = setStage,
                          args = [6])
    gammaCopy = gamma
    
    if gamma == "float":
        
       gamma = st.number_input(label = "Input float for gamma.",
                               value = None,
                               step = 0.1,
                               format = "%.1f",
                               on_change = setStage,
                               args = [6])
       disabledGamma = False
       
       if gamma is not None:
           
          if gamma < 0:
           
             st.markdown(":red[For `gamma`, please input a float greater than or equal to 0.]")
             disabledGamma = True
          
    #------
    # coef0
    #------
    
    if kernel in ["poly", "sigmoid"]:
        
       coef0 = st.number_input(label = "coef0",
                               value = 0.0,
                               step = 0.1,
                               format = "%.1f",
                               on_change = setStage,
                               args = [6])
    
    #----------
    # shrinking
    #----------

    shrinking = st.selectbox(label = "shrinking",
                             options = [True, False],
                             index = 0,
                             on_change = setStage,
                             args = [6])
        
    #----
    # tol
    #----

    tol = st.number_input(label = "tol",
                          value = 1e-3,
                          step = 1e-4,
                          format = "%.4f",
                          on_change = setStage,
                          args = [6])
    disabledTol = False
           
    if tol <= 0:
     
       st.markdown(":red[For `tol`, please input a float greater than 0.]")
       disabledTol = True
    
    #-----------
    # cache_size
    #-----------

    cacheSize = st.number_input(label = "cache_size",
                                value = 200.0,
                                step = 1.0,
                                format = "%.1f",
                                on_change = setStage,
                                args = [6])
    disabledCacheSize = False
           
    if cacheSize <= 0:
     
       st.markdown(":red[For `cache_size`, please input a float greater than 0.]")
       disabledCacheSize = True
    
    #-------------
    # class_weight
    #-------------

    classWeight = st.selectbox(label = "class_weight",
                               options = ["balanced", None],
                               index = 1,
                               on_change = setStage,
                               args = [6])
    
    #---------
    # max_iter
    #---------

    maxIter = st.number_input(label = "max_iter",
                              value = -1.0,
                              step = 1.0,
                              format = "%f",
                              on_change = setStage,
                              args = [6])
    disabledMaxIter = False
    
    maxIterDelta = maxIter - math.floor(maxIter)
     
    if (maxIter < 0 and maxIter != -1)  or (maxIter > 0 and maxIterDelta > 0):
         
       st.markdown(":red[For `max_iter`, please input an integer greater than or equal to -1.]")
       disabledMaxIter = True
       
    else:
       maxIter = int(maxIter)
    
    #------------------------
    # decision_function_shape
    #------------------------

    decisionFunctionShape = st.selectbox(label = "decision_function_shape",
                                         options = ["ovo", "ovr"],
                                         index = 1,
                                         on_change = setStage,
                                         args = [6])
    
    #-----------
    # break_ties
    #-----------

    breakTies = st.selectbox(label = "break_ties",
                             options = [True, False],
                             index = 1,
                             on_change = setStage,
                             args = [6])
    
    params = {}
    params["C"] = C
    params["kernel"] = kernel
    params["gamma"] = gamma
    params["shrinking"] = shrinking
    params["probability"] = True
    params["tol"] = tol
    params["cache_size"] = cacheSize
    params["class_weight"] = classWeight
    params["max_iter"] = maxIter
    params["decision_function_shape"] = decisionFunctionShape
    params["break_ties"] = breakTies
    
    disabled = disabledC + disabledTol + disabledCacheSize + disabledMaxIter
    
    if kernel == "poly":
        
       params["degree"] = degree
       disabled += disabledDegree
       
    if kernel in ["poly", "sigmoid"]:
       params["coef0"] = coef0
       
    if gammaCopy == "float":
       disabled += disabledGamma
       
    return params, disabled

def setModelParams(modelName):
    "Set the model parameters."
    
    if modelName == "Decision Tree Classifier":
       params, disabled = setDTCParams() 
    elif modelName == "Gaussian Process Classifier":
         params, disabled = setGPCParams()
    elif modelName == "k-Nearest Neighbors Classifier":
         params, disabled = setkNNCParams()
    elif modelName == "Logistic Regression":
       params, disabled = setLRParams()
    elif modelName == "Random Forest Classifier":
         params, disabled = setRFCParams()
    elif modelName == "Support Vector Classifier":
         params, disabled = setSVCParams()
         
    return params, disabled
    