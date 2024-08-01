# Image Classification using Fisher kernels in MATLAB

This project is inspired by the code snippets available from Svetlana [Lazebnik et al.](https://slazebni.cs.illinois.edu/). It implement the Spatial Pyramid Matching scheme for classifying different scene categories, while yielding the GPU and Parallel Computing Toolboxes of MATLAB. 


![image](https://github.com/user-attachments/assets/5ce330c4-4a0d-4c9f-8a33-2f6086c43d05)


*Visualization of the density of four individual populations, as a part of an example*

![image](https://github.com/user-attachments/assets/34a3b450-2d78-4665-86f6-7a1d39a606e4)


*Visualization of the same four populations depicting their surface on the 3D space*


![image](https://github.com/user-attachments/assets/5383ee13-96a4-42e1-be03-4132c5994e69)

*2D Contour visualization of the assignment of points into four discrete clusters based on the results of the GMM. With red points are marked the centers of the clusters*

![image](https://github.com/user-attachments/assets/01552bbb-a7ef-469f-aad2-3dbd5aca4c08)

*3D visualization of the surface of the density of the GMM with the corresponding points representing the centers of the clusters*


## Project Structure

- **denseSIFTVN.m**: Extracts dense SIFT descriptors from the images.
- **DictionaryFormationVN.m**: Forms a dictionary of visual words using k-means clustering.
- **Final_Experiment.m**: Main script to run the experiment, including feature extraction, dictionary formation, spatial pyramid matching, and SVM classification.
- **gaussVN.m**: Applies Gaussian filtering to the images.
- **hist_intersection_VN.m**: Computes the histogram intersection kernel.
- **miniBatchKMeansVN.m**: Performs mini-batch k-means clustering.
- **resultsTable.mat**: Stores the results of the experiments.
- **scene_categories/**: Directory containing the dataset of scene categories.
- **SIFTnormalizationVN.m**: Normalizes SIFT descriptors.
- **SpatialPyramidVN.m**: Constructs spatial pyramid representations of the images.
- **splitTheDatastore2.m**: Splits the image datastore into training and testing sets.

## How to Run

To run this project:
1. Ensure MATLAB is installed on your system.
2. Clone this repository to your local machine.
3. Place your dataset in the `scene_categories/`directory.
4. Open MATLAB and navigate to the cloned project directory.
5. Run the `Final_Experiment.m` script to start the image classification pipeline.

```matlab
run('Final_Experiment.m')
```

## Disclaimer

This repository is a simple form of reproduction with a few changes compared to the initial files that are provided.  In keeping with that theme, one can identify the use of GPU accelaration and Parallel processing to minimize the experiment's time to completion. Yet, there exist comments and parts inside some code snippets (even cases where the only changes made in the code snippets are just more explanatory comments) that belong in the initial draft of the contributors as the latter are provided in the [link](https://slazebni.cs.illinois.edu/). All acknowledgements for those parts go to the authors!

## License

This code is for teaching/research purposes only.
