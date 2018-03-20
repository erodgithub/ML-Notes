# ML-Notes
Notes on Random ML things I learn
ML Notes
Cohen’s Kappa
tells you how much better your classifier is than a classifier that guesses randomly
negative values mean your classifier is useless
useful with imbalanced data
=po- pe1 - pe=1-1 - po1 - pe
http://www.skampakis.com/performance-measures-cohens-kappa-statistic/ 
Choosing number of Principals Components to take (PCA)
plot cumulative sum of eigenvalues (in descending order) and plot each value divided by total sum of eigenvalues: will visualize what fraction of total variance is retained vs number eigenvalues used (choose point before diminishing returns are experienced
Make sure to standardize before PCA or else all variance is explained by larger features (remember how at first it was 12 vars but then it was 2)
https://stackoverflow.com/questions/12067446/how-many-principal-components-to-take 
Imbalanced Data strategies
Thresholding, weighting, under/oversampling
https://stackoverflow.com/questions/26221312/dealing-with-the-class-imbalance-in-binary-classification/26244744#26244744 
Academic Paper: http://www.ele.uri.edu/faculty/he/PDFfiles/ImbalancedLearning.pdf 
The Curse of Dimensionality
increasing features significantly lowers training sample density in the feature space, which can lead to overfitting
as dimensions →  differences in distance of the point furthest from the centroid and the point closest to the centroid approach 0.  Picture: As dimensions increase more and more points are concentrated in the corners of the hypercube (at 8 dimensions, 98% of features lie in the corners)
training samples should grow exponentially with number of features
non linear classifiers (svm, KNN, NN, Random Forests) do not take high dimensions well
linear classifiers, naive bayes classifiers take high dimensions well because they are ‘less expressive’ classifiers
http://www.visiondummy.com/2014/04/curse-dimensionality-affect-classification/ 
Over/Undersampling (Coding Maniac)
problem: models favor majority classes
oversampling: make a bunch of fake points that are duplicates of underrepresented instances
undersampling: using algorithm to take smaller random sample of majority class
doing cross-validation: use SMOTE only on training sample -- algorithm only used after each split (so that fake points are not in both the training and testing samples)
https://github.com/coding-maniac 
https://www.youtube.com/watch?v=DQC_YE3I5ig 

Importance of ML Pipelines in Python and sklearn
pipelines automate ml workflows (tell the pipeline what to do to the data and the pipeline will do it without any data leakage)
data leakage: manipulations done on the dataset that are affected by the testing set thus creating a model that influenced by the testing set (a big no no)
for instance, when standardization is done, testing data influences the mean and variance used to standardized which means the training data is affected by these and thus the model is affected by the testing data
https://machinelearningmastery.com/automate-machine-learning-workflows-pipelines-python-scikit-learn/ 
sklearn Pipelines
helps you encapsulates steps of preparing and estimating data so that steps don't get very messy
https://stackoverflow.com/questions/14262654/numpy-get-random-set-of-rows-from-2d-array 
feature selection - select k best
https://www.google.com.br/search?q=Select+K+Best&oq=Select+K+Best&aqs=chrome..69i57j0l2.3558j0j7&sourceid=chrome&ie=UTF-8 
grid search cv
http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html 
pipelining
http://scikit-learn.org/stable/auto_examples/plot_digits_pipe.html 
Combining PCA and SMOTE
https://arxiv.org/pdf/1403.1949.pdf 
http://contrib.scikit-learn.org/imbalanced-learn/stable/combine.html 
Look through the kaggle pca random forest pipeline thing, there was a section about memory reduction that seemed important


