**Model Ensembling using scikit-learn**
-
This sample contains 3 files which uses sklearn's pipeline module to enable Scrybe tracking of hyperparameters used for model ensembling.

`conditional_predictor.py`:
 * A model has already been trained
 * The model is used to get the probabilities of every test sample
 * A threshold is applied on probability to make the final prediction
 * The different values of threshold will automatically get tracked
   
`ensemble_predictor.py`:
 * Two model have already been trained
 * The models are used to get the probabilities of every test sample individually
 * The final probability is the weighted sum of probability of the individual probabilities
 * The different values of weights will automatically get tracked
 
`ensemble_pipeline.py`:
 * Two models are trained as part of the pipeline
 * The models are used to get the probabilities of every test sample individually
 * The final probability is the weighted sum of probability of the individual probabilities
 * The different values of each model's training hyperparameters and the ensembiling weights will automatically get tracked