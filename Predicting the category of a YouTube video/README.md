### Predicting the Category of a given YouTube Video

In this part, the category of a given YouTube video is predicted. This is done in three steps.
1. Data Gathering and Preprocessing
2. Training the algorithms
3. Testing with sample hypothetical titles
#### Data Preprocessing
The data from csv files is imported into a pandas dataframe and the unused attributes (such as comment_count, etc) were removed. 
Initially, the description was used to predict the category but as the RMSE values are very high, title was used to predic the category.
After that, data from the json file was read and a dictionary was created by mapping the category_id and the title and stored into a pandas dataframe.

#### Training
The titles were converted into strings using CountVectorizer.
For training, the following algorithms were used:
      - Multinomial NB
      - Support Vector Classifier
      - Random Forest Classifier
      - K Neighbors Classifier
      - Decision Tree Classifier
The data was then fit to these algorithms for training and validated using 90-10 split.

#### Testing
Testing of the project was done using some hypothetical samples to predict the category.
