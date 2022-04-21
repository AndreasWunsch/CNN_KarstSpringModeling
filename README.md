# CNN_KarstSpringModeling


doi of this repo:  
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5184692.svg)](https://doi.org/10.5281/zenodo.5184692)

doi of according publication:  
*https://doi.org/10.5194/hess-2021-403*

This repository provides example model code according to: 

**Wunsch, A., Liesch, T., Cinkus, G., Ravbar, N., Chen, Z., Mazzilli, N., Jourde, H., and Goldscheider, N.: Karst spring discharge modeling based on deep learning using spatially distributed input data, Hydrol. Earth Syst. Sci. Discuss. [preprint], https://doi.org/10.5194/hess-2021-403, Accepted, 2022.**

Contact: [andreas.wunsch@kit.edu](andreas.wunsch@kit.edu)

ORCIDs of first author:   
A. Wunsch:  [0000-0002-0585-9549](https://orcid.org/0000-0002-0585-9549)   
 

For a detailed description please refer to the publication.
Please adapt all absolute loading/saving and software paths within the scripts to make them running, you Python software for a successful application. Our models are implemented in Python 3.8 (van Rossum, 1995) and we use the following libraries and frameworks: [Numpy](https://numpy.org/) (van der Walt et al., 2011), [Pandas](https://pandas.pydata.org/) (McKinney, 2010; Reback et al., 2020), [Scikit-Learn](https://scikit-learn.org/stable/) (Pedregosa et al., 2011), [Unumpy](https://pythonhosted.org/uncertainties/numpy_guide.html) (Lebigot, 2010), [Matplotlib](https://matplotlib.org/) (Hunter, 2007), [BayesOpt](https://github.com/fmfn/BayesianOptimization) (Nogueira, 2014), [TensorFlow](https://www.tensorflow.org/) and its [Keras](https://keras.io/) [API](https://www.tensorflow.org/versions/r2.3/api_docs/python/tf/keras) (Abadi et al., 2015; Chollet, 2015). Large parts of the code comes from [Sam Anderson](https://github.com/andersonsam/cnn_lstm_era) - please check the according publication: Anderson, Sam and Valentina Radic. "Evaluation and interpretation of convolutional-recurrent neural networks for regional hydrological modelling" (Submitted 2021). The model code runs best on a GPU, unfortunately we cannot provide original data due to republication restrictions of some parties. However, all data is accessible, please refer to the publication for references.

