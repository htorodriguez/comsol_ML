# Getting the most of your Comsol mode
Workflow for the analysis and optmization of a Comsol model using machine learning

# Define your Model, inputs and outputs

# Running the batch process

# Design of experiments
In the best case make a full factorial DoE.

# Machine learning pipeline to create a meta-model
.. code:: python

    def main():
        X, y = load_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        pipeline = Pipeline([('MinMax', MinMaxScaler()),
                             ('pca',PCA()),
                             ('support_vector', LinearSVR()),
                             # or linear regression, or tree methods
                            ])
        # train classifier
        pipeline.fit(X_train, y_train)
        # predict on test data
        y_pred=pipeline.predict(X_test)

# Sensitivity, Robustness or Pareto frontier
[Sensitivity analysis[(https://renovadotengineering.wordpress.com/2020/03/17/example-post-3/) to identify the inputs have the larger or the smaller effect on the outputs
Robustness analysis to estimate the 6-sigma ranges of the output distributions 
Find pareto frontier for cost against performance

# Confirm on model and confirm on hardware
Return to the original model to confirm predictions. Make experiments. Repeat.

