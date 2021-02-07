# Getting the most of your Comsol mode
Workflow for the analysis and optmization of a Comsol model using machine learning

# Define your Model, inputs and outputs

# Running the batch process

# Design of experiments

# Machine Learning Pipeline
.. code:: python

    def main():
        X, y = load_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        pipeline = Pipeline([('MinMax', MinMaxScaler()),
                             ('pca',PCA(n_components=10)),
                             ('support_vector', LinearSVR()),
                             # or linear regression, or tree methods
                            ])
        # train classifier
        pipeline.fit(X_train, y_train)
        # predict on test data
        y_pred=pipeline.predict(X_test)
        # display results
        display_results(y_test, y_pred)

