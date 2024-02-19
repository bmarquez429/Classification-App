"""
        File: lrParams.py
Date Created: February 20, 2024
---------------------------------------------------------
This script is imported by scikit-learnClassification.py.
---------------------------------------------------------
"""

import streamlit as st

def setState(i):
    "Set the session state."
    
    st.session_state.stage = i

def setLRParams():
    "Set the logistic regression parameters."
    
    penalty = st.selectbox(label = "penalty",
                           options = [None, "l1", "l2", "elasticnet"],
                           index = 2,
                           on_change = setState,
                           args = [6])
       
    tol = st.number_input(label = "tol",
                          value = 1e-4,
                          step = 1e-5,
                          format = "%.5f",
                          on_change = setState,
                          args = [6])
    
    if penalty is not None:
       C = st.number_input(label = "C",
                           value = 1.0,
                           step = 0.1,
                           format = "%.1f",
                           on_change = setState,
                           args = [6])
    
    fitIntercept = st.selectbox(label = "fit_intercept",
                                options = [True, False],
                                index = 0,
                                on_change = setState,
                                args = [6])
    
    interceptScaling = st.number_input(label = "intercept_scaling",
                                       value = 1.0,
                                       step = 0.1,
                                       format = "%.1f",
                                       on_change = setState,
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
                          on_change = setState,
                          args = [6])
    
    if penalty == "l2" and solver == "liblinear":
       dual = st.selectbox(label = "dual",
                           options = [True, False],
                           index = 1,
                           on_change = setState,
                           args = [6])
    
    maxIter = st.number_input(label = "max_iter",
                              min_value = 1,
                              value = 100,
                              step = 1,
                              on_change = setState,
                              args = [6])
    
    multiClassOptions = ["auto", "multinomial", "ovr"]
    if solver == "liblinear":
       multiClassOptions.remove("multinomial")
    
    multiClass = st.selectbox(label = "multi_class",
                              options = multiClassOptions,
                              index = 0,
                              on_change = setState,
                              args = [6])
    
    if penalty == "elasticnet":   
       l1Ratio = st.number_input(label = "l1_ratio",
                                 min_value = 0.00,
                                 max_value = 1.00,
                                 value = None,
                                 step = 0.01,
                                 format = "%.2f",
                                 on_change = setState,
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