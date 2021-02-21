# Comsol meta-modelling workflow

## Define your Comsol model

### Inputs 
Under Global definitions, import a Input_Parameters.txt file<br/>

    model.param().loadFile("parameters.txt")

    
### Outputs 
Export you results into a table, e.g.  <br/>

    model.result().export("export_table").run()
### Export Model to .java file
export your model and actions into a .java file<br/>

    my_model.java
## Run the batch process in a .cmd (windows) or .sh(linux) script 
Cast the comsol compilation and batch process into a .cmd or a .sh file<br/>

    "...your_path_to_comsol_bin...\comsolcompile.exe" my_model.java
    "...your_path_to_comsol_bin...\comsolbatch.exe" -inputfile  my_model.class -batchlog log.txt -nosave

## Simulation automation: Python

### Design of experiments
Make an adequate DoE to explore the parameter space according to the use case.

### Run the DoE
* Create child folders, 
* run and store the input parameters matrix (X) 
* and the output result vector (y) 

### Machine learning pipeline to create a meta-model
Make a machine learning pipeline to approximate the simulation by finding an adequate model to map X into y

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

## Sensitivity, Robustness or Pareto frontier
* [Sensitivity analysis](https://renovadotengineering.wordpress.com/2020/03/17/example-post-3/) to identify the inputs have the larger or the smaller effect on the outputs.

* Robustness analysis to estimate the 6-sigma ranges of the output distributions.
 
* Find pareto frontier for cost against performance.

* Confirm the result on the orginal model


