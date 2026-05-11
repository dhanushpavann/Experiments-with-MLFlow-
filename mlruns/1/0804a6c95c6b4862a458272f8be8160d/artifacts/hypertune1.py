from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import pandas as pd
import mlflow



print(f"Current Tracking URI: {mlflow.get_tracking_uri()}")
experiments = mlflow.search_experiments()
print("Experiments found:", [e.name for e in experiments])

# Load the dataset
data=load_breast_cancer()
X=pd.DataFrame(data.data,columns=data.feature_names)
y = pd.Series(data.target, name='target')

# Splitting into training and testing sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# Create a Random Forest Model
rf=RandomForestClassifier(random_state=42)

# Defining the Parameters using GridSearchCV
param_grid={
    'n_estimators':[10,50,100],
    'max_depth':[None,10,20,30]
}

# Apply GridSearchCV
grid_search=GridSearchCV(estimator=rf,param_grid=param_grid,cv=5,n_jobs=-1,verbose=2)


# grid_search.fit(X_train,y_train)

# # Displaying the best params and best score
# best_params=grid_search.best_params_
# best_score=grid_search.best_score_

# print(best_params)
# print(best_score)


mlflow.set_experiment('breast-cancer-rf-hp')

with mlflow.start_run() as parent:

    grid_search.fit(X_train,y_train)

    # Displaying the best params and best score
    best_params=grid_search.best_params_
    best_score=grid_search.best_score_

    # log params
    mlflow.log_params(best_params)

    # log metric
    mlflow.log_metric('acccuracy',best_score)

    # # Log Training data
    # train_df=X_train.copy()
    # train_df['target']=y_train

    # train_df=mlflow.data.from_pandas(train_df)
    # mlflow.log_input(train_df,"Training")


    # # Log Testing data
    # test_df=y_test.copy()
    # test_df['Target']=y_test

    # test_df=mlflow.data.from_pandas(test_df)
    # mlflow.log_input(test_df,"Testing")
    # --- Corrected Data Logging ---
    
    # 1. Training Set: Combine X and y into a single DataFrame for the snapshot
    train_df = X_train.copy()
    train_df['target'] = y_train
    train_ds = mlflow.data.from_pandas(train_df, targets='target', name="BC_Training")
    mlflow.log_input(train_ds, context="training")

    # 2. Testing Set: Combine X and y (Crucial: use X_test, not just y_test!)
    test_df = X_test.copy()
    test_df['target'] = y_test
    test_ds = mlflow.data.from_pandas(test_df, targets='target', name="BC_Testing")
    mlflow.log_input(test_ds, context="testing")

    # Log Source Code
    mlflow.log_artifact(__file__)

    # Log the best model
    mlflow.sklearn.log_model(grid_search.best_estimator_,'random_forest')

    # Set Tags
    mlflow.set_tag("author","Dhanush")

    print(best_params)
    print(best_score)