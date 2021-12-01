# CZ4042 Neural Networks & Deep Learning
> School of Computer Science & Engineering  
> Nanyang Technological University, Singapore

## Part A: Music Genre Classification
Part A of this assignment aims at building neural networks to classify the GTZAN dataset, which is obtained from (http://marsyas.info/downloads/datasets.html). The GTZAN dataset is a widely used dataset collected from 2000 to 2001 from multiple sources [1]. The original dataset consisted of 1000 audio tracks, spanning 30 seconds each. There are 10 different genres in total. For the purpose of this assignment, we will be using a pre-processed dataset where these audio tracks have been processed into features.

### Thoughts on Part A
For questions in Part A, we examined building a neural network model that predicts the genre of a song by using many features as inputs, and tried to optimize the parameters of the model. Firstly, we built a simple two-layer feedforward neural network. After examining its accuracy and losses, we experimented with different batch sizes and different number of hidden neurons for model training. Afterwards, we investigated whether a three-layer neural network will be better, and the effects of having dropout layers.

In our final results, I conclude that a 2-layer neural network (with dropouts) with a batch size of 4 and hidden neuron size of 32 works optimally on the given dataset.

#### Limitations
In our current approach, we need to first extract the features from the audio clips, adding an extra step to our machine learning pipeline. Therefore, the quality of our model depends on the quality of feature extraction.

We see that in our optimized model, there are still some signs of overftting which should be addressed.

I also think that the number of data points (training examples) are low, and our model will perform better if we have more data.

#### Most Impactful Optimization
Finding the optimal batch size made our model better, however, increase the number of hidden neurons had the most impact on our model performance. As the number of neurons increased, the model was able to learn more complex feature representations of the input data. The functions learnt closely resemble the complexity of the data and the model is able to generalize well on unseen real world data.

#### Better Options
Audio clips are waveform data a.k.a sequential data. There another type of Neural Networks called as Recurrent Neural Networks (RNNs), which are helpful in modelling sequential data. Like recurrent neural networks (RNNs), transformers are another type of netowrks designed to handle sequential input data. These options may be more suitable for audio datasets.

#### Extensions of current modelling approach
Our current approach can be used for any set of features. Like audio, we can use some modelling techniques to extract features from image data, and then use our model to train on the data. Our pipeline will mostly remain the same, however, the image data preprocessing will be different.

---

## Part B: HDB Price Prediction
In Singapore, resale prices of Housing Development Board (HDB) flats1 have been on the rise over the past year. The HDB Resale Price Index is inching towards the all-time high previously made in April 2013. It was claimed that the price increase has been a broad-based one but we want to analyse the data more deeply to see if there are other factors behind this increase. This assignment uses publicly available data on HDB flat prices in Singapore, obtained from data.gov.sg on 5th August 2021. The original set of features have been modified and combined with other datasets to include more informative features, as explained on the next page.

### Thoughts on Part B
For questions in Part B, we examined building a neural network model that predicts the HDB prices with a set of features related to the house, and tried to optimize the parameters of the model. Firstly, we built a model that took one-hot encoded vectors for all the variables. Afterwards, we added embedding layers for the categorical variables for a more richer representation. The embedding layers increased the performance of the model. Finally, we investigated the importance of each feature in the model and removed those which lead to a better performance.

#### Limitations
In our current approach, we only have two layers and the hidden layer has only `10` neurons. It will be better if we have more number of neurons and a deeper network such that our model can learn richer representations of the data.

I also think that the number of data points (training examples) are low, and our model will perform better if we have more data.

#### Most Impactful Optimization
Adding the embedding layers to learn from more richer representative of input data certainly helped the model to increase its performance. However, the recursive feature elimination had the most impact in optimizing the model. After removing `eigenvector_centrality` and `degree_centrality`, the `R2` value (Coefficient of Determination) of the model increased significantly.

#### Better Options
There are many machine learning algorithms to perform regression tasks, similar to our question. Some examples include Decision Tree Regression, Support Vector Machines (SVM), and Random Forests. I think whether they will outperform our current neural network model depends on their hyperparameter tuning and other settings.

#### Factors increasing to Price Increase
With the Recursive Feature Elimination (RFE) algorithm, we were able to remove `eigenvector_centrality` and `degree_centrality` as input features. We can look at the `RMSE` values sorted in the reverse order (highest to loweest) to determine what are the factors that lead to the most increase in prices. From the table plotted above, some factors like `dist_to_dhoby` which represents the distance from Dhoby Ghaut MRT station and `storey_range` are important in predicting prices. This is reasonable as location is an important factor in real-estate and taking Dhoby Ghaut MRT, which is located centrally, serves as a good anchor for this purpose.
