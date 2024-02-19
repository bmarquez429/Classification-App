"""
        File: rfcParams.py
Date Created: February 20, 2024
---------------------------------------------------------
This script is imported by scikit-learnClassification.py.
---------------------------------------------------------
"""

import streamlit as st

def setState(i):
    "Set the session state."
    
    st.session_state.stage = i

def setRFCParams():
    "Set the random forest classifier parameters."
    
    nEstimators = st.number_input(label = "n_estimators",
                                  min_value = 1,
                                  value = 100,
                                  step = 1,
                                  on_change = setState,
                                  args = [6])
    
    criterion = st.selectbox(label = "criterion",
                             options = ["gini", "entropy", "log_loss"],
                             index = 0,
                             on_change = setState,
                             args = [6])
    
    maxDepth = st.number_input(label = "max_depth",
                                  min_value = 1,
                                  value = None,
                                  step = 1,
                                  on_change = setState,
                                  args = [6],
                                  placeholder = "None")
    
    minSamplesSplit = st.number_input(label = "min_samples_split",
                                      min_value = 1,
                                      value = 2,
                                      step = 1,
                                      on_change = setState,
                                      args = [6])
    
    minSamplesLeaf = st.number_input(label = "min_samples_leaf",
                                      min_value = 1,
                                      value = 1,
                                      step = 1,
                                      on_change = setState,
                                      args = [6])
    
    minWeightFractionLeaf = st.number_input(label = "min_weight_fraction_leaf",
                                            min_value = 0.0,
                                            max_value = 1.0,
                                            value = 0.0,
                                            step = 0.1,
                                            format = "%.1f",
                                            on_change = setState,
                                            args = [6])
    
    maxFeatures = st.selectbox(label = "max_features",
                               options = ["sqrt", "log2", None],
                               index = 0,
                               on_change = setState,
                               args = [6])
    
    maxLeafNodes = st.number_input(label = "max_leaf_nodes",
                                   min_value = 1,
                                   value = None,
                                   step = 1,
                                   on_change = setState,
                                   args = [6],
                                   placeholder = "None")
    
    minImpurityDecrease = st.number_input(label = "min_impurity_decrease",
                                          min_value = 0.0,
                                          value = 0.0,
                                          step = 0.1,
                                          format = "%.1f",
                                          on_change = setState,
                                          args = [6])
    
    bootstrap = st.selectbox(label = "boostrap",
                             options = [True, False],
                             index = 0,
                             on_change = setState,
                             args = [6])
    
    if bootstrap is True:
       oobScore = st.selectbox(label = "oob_score",
                               options = [True, False],
                               index = 1,
                               on_change = setState,
                               args = [6])
    
    classWeight = st.selectbox(label = "class_weight",
                               options = ["balanced", "balanced_subsample", None],
                               index = 2,
                               on_change = setState,
                               args = [6])
    
    ccpAlpha = st.number_input(label = "ccp_alpha",
                               min_value = 0.0,
                               value = 0.0,
                               step = 0.1,
                               format = "%.1f",
                               on_change = setState,
                               args = [6])
    
    if bootstrap is True:
       maxSamples = st.number_input(label = "max_samples",
                                    min_value = 1,
                                    value = None,
                                    step = 1,
                                    on_change = setState,
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