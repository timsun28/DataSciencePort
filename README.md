# Personal Data Science Portfolio Timo Frionnet

## Introduction
As a part of my Minor Data Science at the Hague University of Applied Sciences we had to maintain our own Data Science Portfolio.
This portfolio needed to contain the following points:
1. Courses (add screenshots of the online courses you have finished (DataCamp, Coursera, etc))
1. Domain Knowledge (Literature, jargon, evaluation, existing data sets, ...)
1. Predictive Models
1. Data preparation
1. Data Visualization
1. Data collection
1. Evaluation
1. Diagnostics of the learning process
1. Communication (presentations, summaries, paper, ...)
1. Link to the Python Notebooks you have finished (you can dump them to PDF)
1. List the tickets from the Scrum backlog that you worked on, linked to deliverables, own experiments, etc.
1. Add any other assignment you feel is evidence of your abilities

I will maintain this structure to keep this portfolio organised and easy to read.

## Table of contents
- [Introduction](#introduction)
- [Courses](#courses)
  - [Datacamp](#datacamp)
      - [Introduction to Python](#introduction-to-python)
      - [Intermediate Python for Data Science](#intermediate-python-for-data-science)
      - [Customizing plots](#customizing-plots)
      - [Introduction and flat files](#introduction-and-flat-files)
      - [Writing your own functions](#writing-your-own-functions)
      - [Data ingestion & inspection](#data-ingestion-inspection)
      - [Exploratory data analysis](#exploratory-data-analysis)
      - [Python Data Science Toolbox (Part 2)](#python-data-science-toolbox-part-2)
      - [Plotting 2D arrays](#plotting-2d-arrays)
      - [Statistical plots with Seaborn](#statistical-plots-with-seaborn)
      - [Cleaning Data in Python](#cleaning-data-in-python)
      - [Statistical Thinking in Python (Part 1)](#statistical-thinking-in-python-part-1)
      - [Supervised Learning with scikit-learn](#supervised-learning-with-scikit-learn)
      - [Conclusion](#conclusion)
  - [Coursera](#coursera)
- [Friday Presentations](#friday-presentations)
  - [Preperation](#preperation)
  - [List of presentations](#list-of-presentations)
- [Scrum](#scrum)




## Courses
I will specify which courses I have taken and in cases where the courses were relevant for our project I will add a note to the course. 

### Datacamp
We were granted access to a full DataCamp account during this course.
Beforehand a couple of courses and chapters were selected for us to finish.
I was able to finish all the courses on time except for the final two, because of prioritization for my project. 
![DataCamp Courses](/images/DataCamp_courses.png)

##### Introduction to Python
Because of my previous experience with Python, this course didn't really add much to my knowledge on Python. 

##### Intermediate Python for Data Science
In this course I have gained experience working with Pandas, which was a great help during the rest of my project. 
This was also my first time working with matplotlib. This helped me to visualize my future findings in data science.

##### Customizing plots
I have continued to expand my knowledge on different ways of visualizing the given Data. 
Because we only received raw email data from our Project owner it was difficult to visualize our data at first. 
After some experimenting I was able to convert the text to values that I was able to plot and that were helpful for the project.
An example of this is the following graph depicting the amount of emails sent per month to the project owner about the noted categories. 
This graph gave us an the project owner a good visual overview of when emails were send and the outliers. 
![DataCamp Courses](/images/email_categorien_per_maand.jpg)

##### Introduction and flat files
For this course we only had to finish the first chapter. This chapter mostly teached me about different ways of reading our files. 
At first we didn't have access to csv file and we had to work with text files for our project. This later changed when we received new exports. 
These exports could be easily imported into a Pandas dataframe. 

##### Writing your own functions
This course was mostly basic knowledge for me as well. It didn't really help me much in the project, because I was already working with functions.


##### Data ingestion & inspection
This chapter mostly helped me understand the basic functionality from Pandas. 
Because I didn't have a lot of experience with Dataframes this course was really helpful to showcase the possibilities.
In our main project we mostly used pandas to store our data in.

##### Exploratory data analysis
This chapter wasn't as helpful as the others. This was mostly because the focus 
during the chapter was mostly on representing numeric values in graphs through the use of pandas.
Because of the shape of our data in strings, we didn't have the ability to use this knowledge.

##### Python Data Science Toolbox (Part 2)
This course wasn't really helpful for our project. However it was a nice refresh of using different methods to process large amounts of data.
It was also the first time for me to use a generator in python. It was interesting to see the use cases for this function. 
I haven't used the data loading in chunks methods in our project. Our dataset was small enough for our laptops to load all the data at once. 

##### Plotting 2D arrays
For our project we mostly plotted accuracies and f1 scores to visualize the performance of a certain model or type of feature. 
This made it that the course about 2d arrays was interesting, but not very useful for our project. 
The issue I had with plotting function was that I always needed to look them up after I was done with the chapter.
It was helpful tho showcase the possibilities, but it will still take lots of practice to get used to the different functions and parameters. 

##### Statistical plots with Seaborn
This chapter was again not very useful for our project. It helped me to use seaborn and see the possibilities it can provide.
I wish I would have been able to use it in our project, but i will keep it in mind for future projects. 

##### Cleaning Data in Python
Because this course was mainly focused on numerical data analyses it wasn't as useful as I hoped it to be.
For our project we performed a lot of data cleaning. This mostly contained retrieving the email from a large file and cleaning our replies from all the emails. 
Because our data was all text I had to look elsewhere for ways to clean out the unnecessary data.

##### Statistical Thinking in Python (Part 1)
This entire chapter was very useful as a way to practise working with a numerical dataset. 
However it was of no use for our project. As mentioned above, I wasn't able to convert text to meaningful numerical values. 

##### Supervised Learning with scikit-learn
This was probably the most useful course in the entire Datacamp selection for our project. 
Especially the classification and fine-tuning your model chapters were very useful and gave clear instructions of ways to combat a certain issue.
The chapter about fine-tuning your model helped us once we got our first model working with the data. 
It assisted me to find different ways on finding possibilities to improve the accuracy of our model. 

##### Conclusion
In the beginning DataCamp didn't really add anything to my knowledge on Python. 
It did help my team mates progress and understand better at what I was working on.
Some courses were not related to our project. Because of this I wasn't able to put everything into practise. 
I did learn a lot from the courses and chapters that I will take with me to upcoming projects.

### Coursera
Coursera was mostly theoretical assistance for the understanding of machine learning. 
At the beginning of our minor we were given lessons on machine learning 
mathematics that were given by: Dr. J. Vuurens who did his PhD at the TU Delft on Data Science.
After these lessons we would take the online Machine learning course by Stanford University. 
This course was mostly repetition and more examples concerning machine learning.

Beforehand we were given a list of quizes we had to finish in certain weeks of the course. 
These were: Week 1, 2, 3 and 6 quizes. The completion of these quizes can be seen in the following screenshot:

![Coursera Courses](/images/Coursera.png)


## Friday Presentations
For our project we had to present our progress and issues each friday to an audience.
Once every two weeks the audience consisted of only students and the teachers. 
The other friday was a public presentation where project owners were allowed to come and sit in.
It was important to keep in mind when it was a public presentation in order to prepare the presentation for the correct audience.

### Preperation
As a group we always prepared the presentation together. Because our team was 
split up in different specialities we always came together before the 
presentation to discuss our progress and issues. This was than converted into a presentation. 
We tried to get everyone to present as often as possible in order for everyone to do their part. 
This meant presenting once every two weeks. We also tried to present in groups 
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

#### Analyzing received dataset  
##### Description: 
This ticket was made to analyze the data we received from our project owner. 
The data was given in a  txt file with no explanation of the structure. 
So we had to do our analyses the file by ourself and figure out the way the 
file was dumped from their database. 

##### Process:
We started by opening the file in a text editor to get an idea of the structure. 
After some time we noticed it was a dump of emails containing the original 
questions asked to our project owner and replies from the people who answer 
their questions. For us the goal of this task was to find a specific separator 
to filter out the questions asked. 

##### Result:
In the file we found that the word: Vraag:  (question in dutch) was a separator 
that we could use to filter out the asked question. After this separator the 
questions was noted and it would end with the word: Category. 
This could be used as a second separator to filter out the original question. 
With this result we could start our next task to build a data cleaner to filter 
out all the asked questions from the datafile. 

#### Cleaning received data
##### Description:
This task consisted of creating a function that would split the input file 
into a list of emails. This function should be made so it can always be used 
for every new file and returns the result as a list.

##### Process: 
With the results from the data analysis I took the two separators and created a 
function for cleaning the data. This function takes the filename as a parameter 
so it can be used with new files. Apart from the split on the separators, 
I also added some basic cleaning like the \n that got hardcoded into the file. 
There was also a need for cleaning white spaces from the sentences. 

##### Result:
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

#### Labeling Questions
##### Description:
After creating an export where we split all the questions up into sentences we 
were going to label the questions. We decided to use binary labels, where a 0 
stands for a non relevant sentence and a 1 for a relevant sentence. 

##### Process:
We uploaded the export to excel online, where we could work on it together. 
Because the first dataset that we labeled was around 4000 sentences, it would 
be much more efficient to work on it as a group. After labeling all the 
questions we stored it as a csv again and created our first dataset that we 
could use to train our selected machine learning models.

##### Result:
A csv file with two columns: Classification and sentence. This csv file could 
than be imported into our code and we could use it to fit our models.


#### Setting up the model for training the data
##### Description:
In this task we had to prepare our project for training our classified data. 
We wanted to try out as many different models with different type of features 
as possible. Another goal was to figure out a way to compare the different 
results to evaluate the models. 

##### Process:
We started by looking on the internet to figure out best practises for this 
issue. We came across this tutorial, that seemed to match our issue. 
[Guide Text Classifications](https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python)

After implementing most of the functions that worked for us we tried to get 
some baseline results from all the different models.
This way we wanted to be able to focus more on specific models. 

##### Result:
At first almost all the models had an accuracy of around 80%. 
This later came to light as a rookie mistake, because our classification was 
labeled where about 80% as 0 and 20% as 1. This meant that some models 
predicted all their sentences as 0 and got 80% correct. With this conclusion we 
created new tasks to figure out different ways of measuring the performance of 
a certain set of model and feature. 


#### Trying different methods and storing results in readable format
##### Description:
This task described the process of creating code to store our results in an 
easy way, so the rest of the team could use them to draw conclusions from them. 
I wasn’t well aware of the different possibilities to visualize the 
performance of a model, I also had to do some research. 

##### Process:
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

##### Result:
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

#### Update import function for new CSV format
##### Description:
The function we created previously got outdated, because we received a new data 
format from our project owner.
They were able to export the data into csv files. This new format would be 
easier to use for us and would have more information. 
In this task we needed to create a new function that would read the csv file. 
We also needed to analyse the data to see if we would also need a new cleaner 
function or if we could use the one we already had.

##### Process:
I first started this task with the analyses of the given csv files. 







