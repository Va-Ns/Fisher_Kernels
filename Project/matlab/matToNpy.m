path = "../data";
load(path + '/FeatureMatrix.mat');

% Load the python executable (environment)
pe = pyenv(Version="D:\Project env\Scripts\python.exe");  % or wherever it is

% Then use the special py variable/struct to call anything from built-in functions,
% built-in and installed libraries/modules. Below we call NumPy to convert
% a MATLAB matrix to a NumPy ndarray and then save it.
sift = py.numpy.array(FeatureMatrix.SIFT_Features_Matrix);
py.numpy.save(path + "/sift feature matrix.npy", sift)
clear sift

rgb = py.numpy.array(FeatureMatrix.RGB_Features_Matrix);
py.numpy.save(path + "/rgb feature matrix.npy", rgb)
clear rgb

redSift = py.numpy.array(FeatureMatrix.Reduced_SIFT_Features_Matrix);
py.numpy.save(path + "/reduced sift feature matrix.npy", redSift)
clear redSift

redRgb = py.numpy.array(FeatureMatrix.Reduced_RGB_Features_Matrix);
py.numpy.save(path + "/reduced rgb feature matrix.npy", redRgb)
clear redRgb FeatureMatrix

% Once done, we terminate the environment to release the resources used.
pe.terminate
pe.Status
