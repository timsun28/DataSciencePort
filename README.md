# Personal Data Science Portfolio Timo Frionnet

## Introduction
As a part of my Minor Data Science at the Hague University of Applied Sciences we had to maintain our own Data Science Portfolio.
This portfolio needed to contain the following points:
- Courses (add screenshots of the online courses you have finished (DataCamp, Coursera, etc))
- Domain Knowledge (Literature, jargon, evaluation, existing data sets, ...)
- Predictive Models
- Data preparation
- Data Visualization
- Data collection
- Evaluation
- Diagnostics of the learning process
- Communication (presentations, summaries, paper, ...)
- Link to the Python Notebooks you have finished (you can dump them to PDF)
- List the tickets from the Scrum backlog that you worked on, linked to deliverables, own experiments, etc.
- Add any other assignment you feel is evidence of your abilities

I will maintain this structure to keep this portfolio organised and easy to read.

## Table of contents
- [Introduction](#introduction)
- [Table of contents](#table-of-contents)
- [Domain Knowledge](#domain-knowledge)
  * [Jargon](#jargon)
  * [Data sets](#data-sets)
- [Courses](#courses)
  * [Datacamp](#datacamp)
      - [Introduction to Python](#introduction-to-python)
      - [Intermediate Python for Data Science](#intermediate-python-for-data-science)
      - [Customizing plots](#customizing-plots)
      - [Introduction and flat files](#introduction-and-flat-files)
      - [Writing your own functions](#writing-your-own-functions)
      - [Data ingestion & inspection](#data-ingestion---inspection)
      - [Exploratory data analysis](#exploratory-data-analysis)
      - [Python Data Science Toolbox (Part 2)](#python-data-science-toolbox--part-2-)
      - [Plotting 2D arrays](#plotting-2d-arrays)
      - [Statistical plots with Seaborn](#statistical-plots-with-seaborn)
      - [Cleaning Data in Python](#cleaning-data-in-python)
      - [Statistical Thinking in Python (Part 1)](#statistical-thinking-in-python--part-1-)
      - [Supervised Learning with scikit-learn](#supervised-learning-with-scikit-learn)
      - [Conclusion](#conclusion)
  * [Coursera](#coursera)
- [Predictive Models](#predictive-models)
- [Data preparation](#data-preparation)
- [Data Visualization](#data-visualization)
- [Data collection](#data-collection)
- [Evaluation](#evaluation)
- [Diagnostics of the learning process](#diagnostics-of-the-learning-process)
- [Communication](#communication)
  * [Tuesday Presentations](#tuesday-presentations)
  * [Friday Presentations](#friday-presentations)
  * [List of presentations](#list-of-presentations)
  * [Paper](#paper)
- [Scrum](#scrum)
  * [Analyzing received dataset](#analyzing-received-dataset)
      - [Description](#description)
      - [Process](#process)
      - [Result](#result)
  * [Cleaning received data](#cleaning-received-data)
      - [Description](#description-1)
      - [Process](#process-1)
      - [Result](#result-1)
  * [Labeling Questions](#labeling-questions)
      - [Description](#description-2)
      - [Process](#process-2)
      - [Result](#result-2)
  * [Setting up the model for training the data](#setting-up-the-model-for-training-the-data)
      - [Description](#description-3)
      - [Process](#process-3)
      - [Result](#result-3)
  * [Trying different methods and storing results in readable format](#trying-different-methods-and-storing-results-in-readable-format)
      - [Description](#description-4)
      - [Process](#process-4)
      - [Result](#result-4)
  * [Update import function for new CSV format](#update-import-function-for-new-csv-format)
      - [Description](#description-5)
      - [Process](#process-5)
      - [Result](#result-5)
- [Other](#other)


## Domain Knowledge
### Jargon
In this personal portfolio the follow terms are frequently used without any further explanation. 
These terms are described in more detail in the following list:

* Machine Learning = A scientific study of algorithms and statistical models 
that improve progressively on a specific task. In our project this is used for classifying relevant questions.
* Naive Bayes = A family of probability classifiers based on the bayes theorem. 
This module assumes strong independence between the features and is a supervised machine learning method.
* Logistic Regression = A statistical model that forms a logistic function to 
model a binary dependent variable. A linear method where the predictions are 
transformed by using a sigmoid function, which can be used for predicting binary dependent variables. 

* Skewed Dataset = A dataset with an imbalance between the different classes. 
in our project this reflects into the amount of relevant questions with a 1:5 ratio to the non-relevant sentences.
* Feature = The input for a predictive model. These features can come in 
different types. In our project most of these features are vectors.

* Data Collection = the process of collecting data that the researchers want to use to create a model to predict a certain value. 
* Data Preparation = The process of converting and cleaning a raw dataset into an input for a predictive model.
* Data Visualisation = The process of visualizing the data into figures which present the data in a meaningful way. 
This also includes the representation of the results.

* Relevant question = A sentence from our dataset that contains a question that 
could be used as an input for a chatbot. 

### Data sets
The data sets that we used in the project were supplied by our project owner. 
Because of confidentiality I am not allowed to supply them. 

## Courses
I will specify which courses I have completed. 

### Datacamp
We were granted access to a full DataCamp account during this course.
Beforehand a couple of courses and chapters were selected for us to finish.
I was able to finish all the courses on time except for the final two, because of prioritization for my project. 
![DataCamp Courses](/images/DataCamp_courses.png)

##### Introduction to Python
Because of my previous experience with Python. This course was not adding much to my knowledge in Python. 

##### Intermediate Python for Data Science
In this course I have gained experience in working with Pandas. This was a great help during the rest of my project. 
This was also my first time working with matplotlib. This helped me to visualize my findings in data science.

##### Customizing plots
I have continued to expand my knowledge in different ways to visualize the given Data. 
It was difficult to visualize the data at first, because we only received raw email data from the project owner.
After some research I was able to convert the text to values that I was able to plot and which were helpful for the project.
An example of this is the following graph depicting the amount of emails sent per month to the project owner about the noted categories. 
This graph gave us and the project owner a good visual overview of when emails were sent. 
![DataCamp Courses](/images/email_categorien_per_maand.jpg)

##### Introduction and flat files
For this course we only had to finish the first chapter. This chapter mostly taught me different ways of reading our files. 
At first we did not have access to a csv file and we had to work with text files for the project. 
This later changed when we received transformed data in csv format.
These exports could be easily imported into a Pandas dataframe, because of this course.

##### Writing your own functions
This course was mostly basic knowledge for me as well. 
It did not really help me much in the project, because I was already working with functions before.


##### Data ingestion & inspection
This chapter mostly helped me understand the basic functionality of Pandas. 
Because I didn't have a lot of experience with dataframes this course was really helpful to show the possibilities.
In our main project we mostly used pandas to store our data.

##### Exploratory data analysis
This chapter was not as helpful as the others. This was mostly because the focus 
during the chapter was mostly on representing numeric values in graphs through the use of pandas.
Because our data consisted only of strings, there was no useful possibility to use this knowledge.

##### Python Data Science Toolbox (Part 2)
This course was not really helpful for our project. 
However it was an interesting course on using different methods to process large amounts of data.
It was also the first time for me to use a generator in python. 
The data for the project was not big enough, so it did not need to be loaded in chunks. 
It was small enough for our laptops to load all the data at once. 

##### Plotting 2D arrays
For our project we mostly plotted accuracies and f1 scores to visualize the performance of a certain model or type of feature. 
Therefore besides that the course about 2d arrays was interesting, it was not very relevant for the project. 
The issue I had with plotting function was that I always needed to look them up after I was done with the chapter.
It was helpful to see the possibilities, but it will still take lots of practice to get used to the different functions and parameters. 

##### Statistical plots with Seaborn
This chapter was unfortunately not very useful for our project, but it helped 
me to use seaborn and see the possibilities it can provide.

##### Cleaning Data in Python
Because this course was mainly focused on numerical data analyses it was not as useful as I hoped it to be.
For our project we performed a lot of data cleaning. 
This mostly contained retrieving the email from a large file and cleaning the replies from all the emails. 
Because our data was all text I had to look elsewhere for ways to clean out all the data except the body of the emails.

##### Statistical Thinking in Python (Part 1)
This entire chapter was very useful as a way to practise working with a numerical dataset. 
However I have not been able to use the knowledge in the project. 

##### Supervised Learning with scikit-learn
This was probably the most useful course in the entire Datacamp selection for our project. 
Especially the classification and fine-tuning your model chapters were very useful and gave clear instructions of ways to combat a certain issue.
The chapter about fine-tuning your model helped us once we got our first model working with the data. 
It assisted me to find different ways on finding possibilities to improve the accuracy of our predictive model. 

##### Conclusion
In the beginning DataCamp did not really add anything to my existing knowledge on Python. 
It did help my project members progress and get a better understanding on what I was working on.
Some courses were not related to our project. Because of this I was not able to put everything into practise. 
I did learn a lot from the courses and chapters that I will take with me to upcoming projects.

### Coursera
Coursera was mostly theoretical assistance for the understanding of machine learning. 
At the beginning of our minor we were given lessons on machine learning 
mathematics that were given by: Dr. J. Vuurens who did his PhD at the TU Delft on Data Science.
After these lessons we would take the online Machine learning course by Stanford University. 
This course was mostly repetition and more examples concerning machine learning.

Beforehand we were given a list of quizzes we had to finish in certain weeks of the course. 
These were: Week 1, 2, 3 and 6 quizzes. The completion of these quizzes can be seen in the following screenshot:

![Coursera Courses](/images/Coursera.png)


## Predictive Models
This chapter will explain more on how I have used different predictive models during my project.
I will reference certain scrum tasks for more detail.

We used Predictive Models in the first part of our project. We classified our 
dataset per sentence if it contains a relevant question or not. 
After our classification we used the data to train and test our different predictive models.
For our project we have worked with the following predictive models:

- Naive Bayes Classifier
- Linear Classifier
- Support Vector machine
- Bagging Models
- Boosting Models

A predictive model doesn't accept a list of raw strings as an input. 
So we had to convert these strings into vectors that could be used by our models.
With most of these models we have tried different feature sets. 
The following feature sets were used for testing the performance of a model. 
Creation of these feature sets is explained in further detail in the part: [Data preparation](#data-preparation).

- Count Vectors
- TF-IDF Vectors 
    - Word level
    - N-gram level
    - Character level

The final two models we did most of our testing with, were Naive Bayes Classifier and Logistic Regression.
In our model we came to the conclusion that for classifying relevant questions from our dataset,
it was best to use Logistic Regression with Count Vectors set to 1500 features.

A basic representation of how these models are used, can be seen in the following example:
```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_fscore_support as score
import pandas as pd

# DataFrame would be loaded at this point
trainDF = pd.DataFrame()

# Split on train, test and cross validation (60%, 20%, 20%)
X_train, X_test, y_train, y_test = train_test_split(trainDF['cleaned_sentence'], trainDF['classification'], test_size=0.4, random_state=42)
X_test, X_cross, y_test, y_cross = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# Create the countvectorizer and create the features for the model
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_df=1.0, max_features=1500)
count_vect.fit(trainDF['sentence'])
xtrain_count = count_vect.transform(X_train)
xvalid_count = count_vect.transform(X_test)
xcross_count = count_vect.transform(X_cross)

# Prepare the model and fit it on the training data
model = LogisticRegression(solver='lbfgs')
mc_model = OneVsRestClassifier(model)
classifier = mc_model.fit(xtrain_count, y_train)

# Predict the labels and compare them to the pre-classified labels
print('training scores:')
print(score(y_train,classifier.predict(xtrain_count),average='weighted'))
print('test scores:')
print(score(y_test,classifier.predict(xvalid_count),average='weighted'))
print('cross-validation scores:')
print(score(y_cross,classifier.predict(xcross_count),average='weighted'))
```

## Data preparation
Our data preparation mostly consisted of creating features for our models and cleaning the data.

In order for the predictive models to understand our data we needed to prepare it first. 
All the sentences were converted into vectors for the model to be able to understand the data.
I will describe the two different methods that we followed for converting our sentences into these vectors.

In the following example the Count Vectorizer is created and the full data set is fitted on the vectorizer.
Once the vectorizer is done it will create a Vector for the given sentences (X)
The output of this process can be used as a input for a predictive model
```python
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# The full vocabulary of the file
all_text = pd.Series
X = pd.Series

# Variable amount of features could be set beforehand to prevent the model to overfit the data.
amount_features = 1500

# Creating the count vectorizer with the given parameters.
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=amount_features)

# Fit all the data on the count vectorizer.
count_vect.fit(all_text)

# Transform the wanted sentences into a count vector
X_count = count_vect.transform(X)
```

The same goes for the TF-IDF transformer. In the given example the three possible transformers are displayed.
```python
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# The full vocabulary of the file
all_text = pd.Series
X = pd.Series

# Variable amount of features could be set beforehand to prevent the model to overfit the data.
amount_features = 1500

# Creating the Tfidf Vectorizer with the given parameters.
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=amount_features)
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2, 5), max_features=amount_features)
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=amount_features)

# Fit all the data on the TF-IDF vectorizer.
tfidf_vect.fit(all_text)
tfidf_vect_ngram.fit(all_text)
tfidf_vect_ngram_chars.fit(all_text)

# Transform the wanted sentences into a TF-IDF vector
X_TF_IDF = tfidf_vect.transform(X)
X_TF_IDF_ngram = tfidf_vect_ngram.transform(X)
X_TF_IDF_ngram_chars = tfidf_vect_ngram_chars.transform(X)
```

In order to clean the data I researched different methods to clean text. 
I would than try those methods out on our data and compare them
directly with the original score to see if the accuracy would improve. 
The following class was created for our project to clean every sentence.

```python
import string
import unicodedata
import re
from nltk.stem.snowball import SnowballStemmer
from DataSources.fixtures import get_gemeenten
from itertools import groupby, chain
from nltk.corpus import stopwords

class PreTraining:
    def __init__(self):
        self.sentence = None
        self.stemmer = SnowballStemmer('dutch', ignore_stopwords=True)
        self.locations = get_gemeenten()
        self.dutch_stopwords = stopwords.words('dutch')

    def prepare_sentence(self, sentence):
        self.sentence = sentence
        self.to_lower()
        self.remove_urls()
        self.remove_punctuation()
        self.remove_diacritics()
        self.remove_spaces()
        self.normalize_numbers()
        self.normalize_location_data()
        return self.sentence

    def to_lower(self):
        self.sentence = self.sentence.lower()

    def remove_urls(self):
        self.sentence = re.sub(r"http\S+", "", self.sentence)

    def remove_punctuation(self):
        table = str.maketrans({key: None for key in string.punctuation})
        self.sentence = self.sentence.translate(table)

    def remove_diacritics(self):
        self.sentence = unicodedata.normalize('NFKD', self.sentence).encode('ASCII', 'ignore').decode('utf-8')

    def remove_spaces(self):
        self.sentence = ' '.join(self.sentence.split())

    # todo doesn't work well, needs better accuracy
    def sentence_stemmer(self):
        self.sentence = self.stemmer.stem(self.sentence)

    def normalize_location_data(self):
        regex = re.compile(r'\b%s\b' % r'\b|\b'.join(map(re.escape, self.locations)))
        self.sentence = regex.sub('LOCATIE', self.sentence)

    def normalize_numbers(self):
        grouped = groupby(self.sentence, str.isdigit)
        self.sentence = ''.join(chain.from_iterable('NUMWAARDE' if k else g for k,g in grouped))

    def remove_stopwords(self):
        regex = re.compile(
            r'\b%s\b' % r'\b|\b'.join(map(re.escape, self.dutch_stopwords)))
        self.sentence = regex.sub('', self.sentence)
```

It could be used in the following way: 
```python
PreTrainer = PreTraining()
cleaned_sentence = PreTrainer.prepare_sentence(sentence)
```

Most of the cleaning we performed did not have a big impact on the scores, 
and some results even made the score worse.
In the following two figures it is shown that for both recall and precision the 
cleaning on our data had marginal effect.

![Precision score](/images/precision_cross_val_cleaning.png)
![Recall score](/images/Recall_cross_val_cleaning.png)

As you can see the difference between these values is so small it does not have an 
effect on the outcome of our project.

## Data Visualization
Once all the coding and testing is done, the most important final chapter is to 
visualize your findings. After doing research on possibilities to visualize your findings we tried multiple graphs.
Most of the graphs were for clarification in the research paper, but we also 
made some graphs that were used in the friday presentations and just for 
the team to better understand the results.

The visualization I made for our project were the Confusion Matrices that are 
shown in the scrum ticket: [Trying different methods and storing results in readable format](#trying-different-methods-and-storing-results-in-readable-format).

## Data collection
The data set we received from our project owner was dumped from their email system. 
All the emails with their responses were stored in a csv format.
It was delivered to us as 150 different csv files. 
Each file contained all the messages that were manually categorised.  

As part of our data cleaning process an external dataset was used. 
We wanted to see if we would improve the accuracy of our model if all the Dutch city names were normalized.

A dump of all the cities in the Netherlands was used in Json format. 
This function is used to replace city names with LOCATION in our dataset.

```python
import re
import json

def get_locations():
    f = open('DataSources/locations_json', 'r', encoding='UTF-8')
    file = f.read()
    encoded_json = json.loads(file)
    all_locations = list()
    for location in encoded_json['value']:
        all_locations.append(location['Title'].lower())
    return all_locations
    
all_locations = get_locations()

sentence = 'This is a test sentence from Amsterdam'
regex = re.compile(r'\b%s\b' % r'\b|\b'.join(map(re.escape, all_locations)))
sentence = regex.sub('LOCATION', sentence)
# Sentence is now: This is a test sentence from LOCATION
```

After comparing the performance of our model with and without normalization of 
the locations, we found that it improved on average with 0.1%.
Unfortunately it didn't help much, but it was still helpful to use an external 
data source for personal development. 


## Evaluation
The evaluation of the models was usually done with the precision, recall and f1 scores. 
These scores made it easy to compare models when using a variable amount of features for comparing different models.

Apart from numeric evaluation, we also performed an error analyses. 
By exporting all the incorrect predictions we have been able to improve our 
initial classification and find patterns on where the model has issues to perform.
It was interesting to see how the model was in some cases better at recognizing 
relevant questions than we intentionally labeled.
Finding patterns in the mistakes was also a part of our error analyses. 
The most noticeable mistake that was made had to do with greetings in a sentence.
In some cases a sentence was made up with a greeting and a relevant question. 
The model labeled these sentences as not relevant, because of the greeting, that would have been labeled as not relevant.

At this point we haven't figured out a way to give a certain weight to these 
questions, but this could be a way to elevate this project to a next level to 
get above the 90% accuracy.

## Diagnostics of the learning process
For me the learning process of working with Datacamp was pleasant. 
Most of the courses contained something interesting that I could use in our own project. 
This made it so that I could directly apply the things that i had learned 
during the DataCamp courses. Unfortunately not all the courses were relevant for
our project. Nevertheless Datacamp gave me a good feeling of the basics of Machine Learning 
and if I am able to use it in the future I would definitely. 

The Coursera courses were helpful for background understanding of the models and their functionality.
Also to see the bigger picture and how the process of machine learning was better explained.  
Less practical was the fact that in the Coursera courses Python was not the 
programming language used for coding. This meant I could not use all of the same tools they used. 
This way I always had to take an extra step to look for an alternative and I 
could not directly apply everything I learned to my project.

The entire Minor that I followed was a very entry level way of learning the basics of Machine Learning. 
Because I had previous experience with working with Python I could directly 
start with the Machine learning part of the minor. It was also helpful for my team because I could help 
them with their problems and I could teach them best practices in Python.  

## Communication

### Tuesday Presentations
Every tuesday we worked at the office of our project owner. 
As a group we presented our findings from the week before. 
We also discussed issues we were having and they helped us where possible.

### Friday Presentations
For our project we had to present our progress and issues each friday to an audience.
Once every two weeks the audience consisted of only students and the teachers. 
The other friday was a public presentation where project owners were allowed to come and sit in.
It was important to keep in mind when it was a public presentation in order to prepare the presentation for the correct audience.

As a group we always prepared the presentation together. Because our team was 
split up in different specialities we always came together before the 
presentation to discuss our progress and issues. This was then converted into a presentation. 
We tried to get everyone to present as often as possible in order for everyone to do their part. 
This meant I presented at least once every two weeks. We also tried to present in groups 
of two where there is at least one student who is working primarily on the code and one student who focuses more on the report.


### List of presentations
* [Week 1](/presentations/2018.08.31%20Presentatie.pptx)
* [Week 2](/presentations/2018.09.07%20Presentatie.pptx)
* [Week 3](/presentations/2018.09.14%20Presentatie.pptx)
* [Week 4](/presentations/2018.09.21%20Presentatie.pptx)
* [Week 5](/presentations/2018.09.28%20Presentatie.pptx)
* [Week 6](/presentations/2018.10.05%20Presentatie.pptx)
* [Week 7](/presentations/2018.10.12%20Presentatie.pptx)
* [Week 8](/presentations/2018.10.19%20Presentatie.pptx)
* [Week 10](/presentations/2018.11.02%20Presentatie.pptx)
* [Week 11](/presentations/2018.11.09%20Presentatie.pptx)
* [Week 12](/presentations/2018.11.16%20Presentatie.pptx)
* [Week 14](/presentations/2018.11.30%20Presentatie.pptx)
* [Week 15](/presentations/2018.12.07%20Presentatie.pptx)
* [Week 17](/presentations/2018.12.21%20Presentatie.pptx)

### Paper
The paper was mostly written using the results I created in Python. 
I have personally written about topic modelling. The communication about the results went well. 
In cases where something was unclear, I was able to give the rest of the group a clear view of the results.
This way everybody understood what I made and the thought behind it.

## Scrum
At the start of the course we were advised to use some kind of scrum 
methodology to keep track of our project.
We decided to use Scrumwise to keep track of our sprint progress. 
In this chapter I'm going to pick out certain tasks that I worked on. 
In the following tasks i write as if I worked on it alone. This is not the case 
for all the tasks, but it would make it unclear if I would note for every line 
who worked on what.
I will explain my progress and what I have learned from working on this task. 
As well as the final result and how it was useful for our project. 

### Analyzing received dataset  
##### Description
This ticket was made to analyze the data we received from our project owner. 
The data was given in a  txt file with no explanation of the structure. 
So we had to do  analysis of the file by ourself and figure out the way the 
file was dumped from their database. 

##### Process
We started by opening the file in a text editor to get an idea of the structure. 
After some time we noticed it was a dump of emails containing the original 
questions asked to our project owner and replies from the people who answer 
their questions. For us the goal of this task was to find a specific separator 
to filter out the questions asked. 

##### Result
In the file we found that the word: Vraag:  (question in dutch) was a separator 
that we could use to filter out the asked question. After this separator the 
questions was noted and it would end with the word: Category. 
This could be used as a second separator to filter out the original question. 
With this result we could start our next task to build a data cleaner to filter 
out all the asked questions from the datafile. 

### Cleaning received data
##### Description
This task consisted of creating a function that would split the input file 
into a list of emails. This function should be made so it can always be used 
for every new file and returns the result as a list.

##### Process
With the results from the data analysis I took the two separators and created a 
function for cleaning the data. This function takes the filename as a parameter 
so it can be used with new files. Apart from the split on the separators, 
I also added some basic cleaning like the \n that got hardcoded into the file. 
There was also a need for cleaning white spaces from the sentences. 

##### Result
The result of this task was a function that would retrieve all the questions 
asked in the dump file. It is also able to clean the emails from ‘\n’ and 
whitespaces.

```python
def clean_file(file_name):
         """
         function that cleans the data and returns a list with all the questions
         :return:
         """
         f = open(file_name, 'r')
         file = f.read()
         result = list()
         split = file.split('Vraag:')
         for index, question in enumerate(split):
            try:
                res = question.split('Category')
                result_cleaned = res[0].strip().replace('E-mail content by category', '').strip()
                result_cleaned = result_cleaned.replace('\n', '')
                result_cleaned = ' '.join(result_cleaned.split())
                if result_cleaned not in result:
                    result.append(result_cleaned)
            except IndexError:
            # in case the question couldn't be split a second time
                continue
         return result
```

### Labeling Questions
##### Description
After creating an export where we split all the questions up into sentences we 
were going to label the questions. We decided to use binary labels, where a 0 
stands for a non relevant sentence and a 1 for a relevant sentence. 

##### Process
We uploaded the export to excel online, where we could work on it together. 
Because the first dataset that we labeled was around 4000 sentences, it would 
be much more efficient to work on it as a group. After labeling all the 
questions we stored it as a csv again and created our first dataset that we 
could use to train our selected machine learning models.

##### Result
A csv file with two columns: Classification and sentence. This csv file could 
than be imported into our code and we could use it to fit our models.


### Setting up the model for training the data
##### Description
In this task we had to prepare our project for training our classified data. 
We wanted to try out as many different models with different type of features 
as possible. Another goal was to figure out a way to compare the different 
results to evaluate the models. 

##### Process
We started by looking on the internet to figure out best practises for this 
issue. We came across this tutorial, that seemed to match our issue. 
[Guide Text Classifications](https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python)

After implementing most of the functions that worked for us we tried to get 
some baseline results from all the different models.
This way we wanted to be able to focus more on specific models. 

##### Result
At first almost all the models had an accuracy of around 80%. 
This later came to light as a rookie mistake, because our classification was 
labeled where about 80% as 0 and 20% as 1. This meant that some models 
predicted all their sentences as 0 and got 80% correct. With this conclusion we 
created new tasks to figure out different ways of measuring the performance of 
a certain set of model and feature. 


### Trying different methods and storing results in readable format
##### Description
This task described the process of creating code to store our results in an 
easy way, so the rest of the team could use them to draw conclusions from them. 
I was not well aware of the different possibilities to visualize the 
performance of a model, I also had to do some research. 

##### Process
It started by looking at sklearn and their metrics module. 
This module is used to score functions and create performance metrics. 
It became clear that apart from the accuracy we also needed the recall to 
calculate the total amount of true positives. Together with the F1 score, 
that combines the recall and precision we decided to use those metrics to 
describe the performance of our model. I also started working with confusion matrices. 
These visualization helped me understand our results in a better 
way and was also very helpful to display our findings during public presentations. 
Because we used a binary classification the confusion matrix was a 2x2 table 
that was easy to explain and use.

##### Result
The result from this task was a class to calculate the metrics for our results. 
I also finished a function to create a confusion matrix given the 
classification and predictions. Part of this code was found on the sklearn 
website. I added my own function that could be called like this. 
It would then create two confusion matrices. 

```python
create_confusion_matrix(classifications, predictions, 'Logistic Regression')
```

```python
def create_confusion_matrix(self, valid_y, predictions_valid, model_name):
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(valid_y, predictions_valid)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    class_names = ['not interesting', 'interesting']
    self.plot_confusion_matrix(cnf_matrix,
                               classes=class_names,
                               title= model_name + ' Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    self.plot_confusion_matrix(cnf_matrix, classes=class_names,
                               normalize=True,
                               title= model_name + ' Normalized confusion matrix')

    plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
```

![Confusion Matrix](/images/confusion_matrix.png)
What is depicted in this image: 
69 sentences that we classified as interesting were predicted as interesting
28 sentences that we classified as interesting were predicted as not interesting
79 sentences that we classified as not interesting were predicted as interesting
371 sentences that we classified as not interesting were predicted as not interesting

### Update import function for new CSV format
##### Description
The function we created previously got outdated, because we received a new data 
format from our project owner.
They were able to export the data into csv files. This new format would be 
easier to use for us and would have more information. 
In this task we needed to create a new function that would read the csv file. 
We also needed to analyse the data to see if we would also need a new cleaner 
function or if we could use the one we already had.

##### Process
I first started this task with the analyses of the given csv files. This file 
was structured differently than the previous file we used in our project. 
From this point it was decided to use Pandas for storing our data in the code.
This was very useful because of the from_csv function pandas supplies. 
This was added to our project and we could now store the new files in Pandas Dataframes
One of the columns was labeled: TEXT_CONTENT which still had the same structure as the original file.
This meant that we only had to run some basic tests to see if the result was still as expected.
Apart from this there we tried to fix the issue where replies on the emails were also extracted from the data dump.
Because we now had access to the subject of the email, we made a function to 
separate the emails into a dataframe with only replies and one without replies.

##### Result
After this task we finished our second importer. 
From now on we had more certainty that the data was correct and because of the
structure we could also possibly use more metadata from the file like date and
type of message (html or txt)

The function that splits the dataset into two different sets (one for replies and one without the replies)
we made the following function: 

```python
def split_replies_from_file(data_frame):
    """
    Based on certain keywords we can find the subjects that are a reply to a certain person.
    These answers do not contain any questions, so they were kept out.
    These can be used for different research.
    :param data_frame:
    :return:
    """
    unwantedSubjectStarts = ('re', 'undelivered', 'fw', 'geen onderwerp')
    unwanted_code = '#'
    without_classification = data_frame[~data_frame['SUBJECT'].str.lower().str.contains(unwanted_code)]
    without_replies = without_classification[~without_classification['SUBJECT'].str.lower().str.startswith(unwantedSubjectStarts)]
    only_classification = data_frame[data_frame['SUBJECT'].str.lower().str.contains(unwanted_code)]
    only_replies = only_classification[only_classification['SUBJECT'].str.lower().str.startswith(unwantedSubjectStarts)]
    return without_replies, only_replies
```

The unwantedSubjectStarts is a tuple with strings of common words used in
the email subjects. In case a subject starts with one of these words it gets filtered out.

 ## Other
 During our error analysis we discovered our models had issues processing sentences that contained greetings.
 I was interested in finding a solution to this problem, so I did some research. 
 On github I came across a tool that would split an email into a salutation, body and signature. 
 This seemed interesting at first, but the tool was hardcoded in English. So it would not have worked on our Dutch sentences. 
 I made a fork from their branch in order to add internationalization to their tool. 
 This fork can be seen ![here](https://github.com/timsun28/EFZP). 
 
 Unfortunately after more testing it did not work properly on our data. 
 Improvements needed to be made, but because of time management it could not be done yet. 
  






