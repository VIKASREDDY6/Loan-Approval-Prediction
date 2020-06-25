import streamlit as st
import pandas as pd
import numpy as np


def main():
    st.title("Loan Approval Prediction with Machine Learning.")
    #load dataset and preprocessing
    df=pd.read_csv('loandataset.csv')
    df=df.drop(columns=['Loan_ID'])
    loandf=pd.get_dummies(df,drop_first=True)
    X=loandf.drop(columns='Loan_Status_Y')
    y=loandf['Loan_Status_Y']
    
    #Train-Test-Split
    from sklearn.model_selection import train_test_split
    testsize=st.slider("Test Data size(%) for splitting:",10,50)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=testsize/100,random_state=42)
    
    #Imputing missing values by mean
    from sklearn.impute import SimpleImputer
    imp=SimpleImputer(strategy='mean')
    impfit=imp.fit(X_train)
    X_train=impfit.transform(X_train)
    X_test=impfit.transform(X_test)
    
    #Input features
    featip=[]
    st.subheader("Enter the required data:")
    appincome=st.slider("Applicant Income(In Thousands):",1,100)
    featip.append(appincome*1000)
    coappincome=st.slider("CoApplicant Income(In Thousands):",0,50)
    featip.append(coappincome*1000)
    loanamt=st.slider("Loan Amount(In Thousands):",10,750)
    featip.append(loanamt)
    loanterm=st.slider("Loan term(Duration) in Years:",1,40)
    loanterm=loanterm*12
    featip.append(loanterm)
    credithis=st.slider("Credit History(0/1):",0,1)
    featip.append(credithis)
    gen=st.radio("Gender:",('Male','Female'))
    if gen=='Male':
        gen=1
    else:
        gen=0
    featip.append(gen)
    married=st.radio("Married:",('Yes','No'))
    if married=='Yes':
        married=1
    else:
        married=0
    featip.append(married)
    dep=st.radio("Number of dependents:",('1','2','3 or more'))
    if dep=='1':
        dep1=1
        dep2=0
        dep3=0
    elif dep=='2':
        dep1=0
        dep2=1
        dep3=0
    else:
        dep1=0
        dep2=0
        dep3=1
    featip.append(dep1)
    featip.append(dep2)
    featip.append(dep3)
    ed=st.radio("Education:",('Graduate','Not Graduate'))
    if ed=='Graduate':
        edn=0
    else:
        edn=1
    featip.append(edn)
    selfemp=st.radio("Employement:",('SelfEmployed','NotSelfEmployed'))
    if selfemp=='SelfEmployed':
        selfempy=1
    else:
        selfempy=0
    featip.append(selfempy)
    prop=st.radio("Property Area:",('Urban','SemiUrban','Rural'))
    if prop=='Urban':
        propu=1
        propsu=0
    elif prop=='SemiUrban':
        propu=0
        propsu=1
    else:
        propu=0
        propsu=0
    featip.append(propsu)
    featip.append(propu)
        
    
    #Classifier DT
    from sklearn.tree import DecisionTreeClassifier
    dt=DecisionTreeClassifier(max_depth=4,min_samples_leaf = 25)
    dt.fit(X_train,y_train)
    pred_dt=dt.predict(X_test)
    from sklearn import metrics
    score=metrics.accuracy_score(y_test,pred_dt)
    recall=metrics.recall_score(y_test,pred_dt)
    precision=metrics.precision_score(y_test,pred_dt)
    f1=metrics.f1_score(y_test,pred_dt)
    print("Confusion Matrix on Test Data")
    if st.button("Get Prediction"):
        res=dt.predict([featip])[0]
        if res==1:
            st.info("Congrats! Your Loan is Approved.")
        else:
            st.warning("Sorry. Your Loan is Not Approved.")
        st.subheader("Metrics:")
        st.write("Accuracy:",score)
        st.write("Precision:",precision)
        st.write("Recall:",recall)
        st.write("F-1 Score:",f1)
        st.write("Confusion Matrix(Actual VS Predicted)")
        st.write(pd.crosstab(y_test, pred_dt, rownames=['True'], colnames=['Predicted'], margins=True))
    
    if st.button("See who created this!"):
        st.info("Name: K. Vikas Reddy")
        st.info("College: SASTRA Deemed to be University")
        st.info("Gmail: reddyvikas995@gmail.com")
        st.write("You can also see my other projects at:")
        st.write("1. https://edawithease.herokuapp.com to perform Exploratory Data Analysis.")
        st.write("2. https://diabetes-prediction-ml.herokuapp.com to predict diabetic condition.")
        st.write("3. https://ml-classifiers-tool.herokuapp.com to utilize automated ML Classifiers.")
        
    st.warning("Please report any bugs and suggestions if any.")
        
    if st.checkbox("About this Project"):
        st.write("This project predicts loan approval status by predicting whether a loan applicant is likely to repay the loan amount or not.")
        st.write("Here I used Decision Tree Classifier.Because both RandomForest and Logistic regression gave similar results like Decision Tree.")
        st.subheader("Note:")
        st.write("The Classfiers's accuracy is less than 80%.It is because of the available data.")
        st.write("The accuracy can be improved by using GradientBoosting and Hyper Parameter optimization using XGBoost.")
        st.write("Loan Approval Status highly depends on Credit History i.e., credit history 1 is likely to predict Loan Approved.")
        
    
if __name__=='__main__':
    main()
