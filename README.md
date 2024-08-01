# Image Classification using Fisher kernels in MATLAB

This project is an attempt to recreate the work done by Florent Perronnin: "_Fisher Kernels on Visual Vocabularies for Image Categorization_" by classifying a set of images with scenes. Using the dataset that is available from Svetlana [Lazebnik et al.](https://slazebni.cs.illinois.edu/) in the SPM experiment, the axis of this project is to use Gaussian Mixture Models, the sufficient statistics of which are used to form a Dictionary for the features extracted by SIFT and the RGB planes of the data. With the Fisher kernel consisting of computing the gradient of the sufficient statistics and concatenating them in the form of a vector, the latter encodes the data into a vector representation that is normalized using an approximation of the diagonal of the Fisher Information Matrix. Finally, the vectors are feeded in a SVM classifier. With the metric of the project being _Accuracy_, it can be seen that Fisher Kernels manage to increase the descriptiveness induced by the SPM schematic.


![image](https://github.com/user-attachments/assets/5ce330c4-4a0d-4c9f-8a33-2f6086c43d05)


*Visualization of the density of four individual populations, as a part of an example*

![image](https://github.com/user-attachments/assets/34a3b450-2d78-4665-86f6-7a1d39a606e4)


*Visualization of the same four populations depicting their surface on the 3D space*


![image](https://github.com/user-attachments/assets/5383ee13-96a4-42e1-be03-4132c5994e69)

*2D Contour visualization of the assignment of points into four discrete clusters based on the results of the GMM. With red circles are marked the centers of the clusters*

![image](https://github.com/user-attachments/assets/01552bbb-a7ef-469f-aad2-3dbd5aca4c08)

*3D visualization of the surface of the density of the GMM with the corresponding points representing the centers of the clusters*


## Project Structure

- **CalculateParamsNV.m**: Calculates the parameters for the Gaussian Mixture Model (GMM).
- **extractImageFeatures.m**: Extracts image features from the dataset.
- **FisherEncodingNV.m**: Encodes features using Fisher encoding.
- **GMM_NV.m**: Implements the Gaussian Mixture Model (GMM) for the project.
- **sEM.m**: Implements the stochastic Expectation-Maximization algorithm.
- **Final_project.m**: Main script to run the project, including data loading, feature extraction, GMM parameter calculation, Fisher encoding, and classification.
- **scene_categories/**: Directory containing the dataset of scene categories.

## How to Run

To run this project:
1. Ensure MATLAB is installed on your system.
2. Clone this repository to your local machine.
3. Place your dataset in the `scene_categories/`directory.
4. Open MATLAB and navigate to the cloned project directory.
5. Run the `Final_project.m` script to start the image classification pipeline.

```matlab
run('Final_project.m')
```

## License

This code is for teaching/research purposes only.
