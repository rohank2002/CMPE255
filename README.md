# Youtube Trending Video Data Analysis
This project was developed as a part of CMPE 255 course.

### Data Description:
This dataset includes several months (and counting) of data on daily trending YouTube videos. Data is included for the US, GB, DE, CA, FR, RU, MX, KR, JP and IN regions (USA, Great Britain, Germany, Canada, France, Russia, Mexico, South Korea, Japan and India respectively), with up to 200 listed trending videos per day.

Each region’s data is in a separate file. Data includes the video title, channel title, publish time, tags, views, likes and dislikes, description, and comment count.

The data also includes a category_id field, which varies between regions. To retrieve the categories for a specific video, find it in the associated JSON. One such file is included for each of the five regions in the dataset.

For more information on specific columns in the dataset refer to the column metadata.
We only used US, CA, GB, IN for this project.
Source: https://www.kaggle.com/datasnaek/youtube-new

### Tasks performed:
Performing analysis and predictions on YouTube video dataset.
- Task 1: Predicting the Category of a YouTube video based on its Title.
- Task 2: Predicting the number of views  (popularity) of a particular video given its Title.
- Task 3: Sentiment analysis of the description, tags and title.


### Instructions to run:
You will require Jupyter Notebook or any Python IDE with Python 3.0 or later installed to run the code.
Change the directory of the data while loading it. 
