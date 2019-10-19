### Sentiment Analysis Introduction

<b> Sentiment analysis </b> refers to the use of natural language processing, text analysis, computational linguistics, 
and biometrics to systematically identify, extract, quantify, and study affective states and subjective information. 
Sentiment analysis is widely applied to voice of the customer materials such as reviews and survey responses, online and social media, 
and healthcare materials for applications that range from marketing to customer service to clinical medicine. [1]<br/>

### Implementation
The data from CSV files was imported and the redundant data was grouped by using video-id. Then, the trends from the data of each country were plotted. These trends include the duration of a video being popular, how long did a video take to become popular, and the most trending categories.
Wordclouds were then individually constructed to find the frequently used words in titles, tags, and descriptions respectively. These were plotted just as an EDA. <br>

Later, sentiment type (Positive, negative, neutral) of title, desciption and tags columns was obtained using Textblob. As there
was no certain metric to evaluate the performance of textblob, it was set as the baseline
and the SVM classifier was applied on the description, tags, title and the sentiment and
it was evaluated using the classification_report in the scikit-learn library. Also used
linear regression and accuracy was considered as the evaluation metric. Marginally,
SVM performed better than Logistic Regression. Tried applying CNN and RNN on the
dataset which were giving Memory Error on the local system.

### References
[1] https://en.m.wikipedia.org/wiki/Sentiment_analysis
