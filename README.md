# Personal Data Science Portfolio Timo Frionnet

## Introduction:
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

Table of contents:
- [Introduction:](#introduction)
- [Courses:](#courses)
  - [Datacamp:](#datacamp)
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
      - [Conclusion:](#conclusion)
  - [Coursera:](#coursera)
- [Friday Presentations:](#friday-presentations)
  - [Preperation:](#preperation)
  - [List of presentations:](#list-of-presentations)
- [Scrum:](#scrum)




## Courses: 
I will specify which courses I have taken and in cases where the courses were relevant for our project I will add a note to the course. 

### Datacamp: 
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

##### Conclusion:
In the beginning DataCamp didn't really add anything to my knowledge on Python. 
It did help my team mates progress and understand better at what I was working on.
Some courses were not related to our project. Because of this I wasn't able to put everything into practise. 
I did learn a lot from the courses and chapters that I will take with me to upcoming projects.

### Coursera: 
Coursera was mostly theoretical assistance for the understanding of machine learning. 
At the beginning of our minor we were given lessons on machine learning 
mathematics that were given by: Dr. J. Vuurens who did his PhD at the TU Delft on Data Science.
After these lessons we would take the online Machine learning course by Stanford University. 
This course was mostly repetition and more examples concerning machine learning.

Beforehand we were given a list of quizes we had to finish in certain weeks of the course. 
These were: Week 1, 2, 3 and 6 quizes. The completion of these quizes can be seen in the following screenshot:

![Coursera Courses](/images/Coursera.png)


## Friday Presentations:
For our project we had to present our progress and issues each friday to an audience.
Once every two weeks the audience consisted of only students and the teachers. 
The other friday was a public presentation where project owners were allowed to come and sit in.
It was important to keep in mind when it was a public presentation in order to prepare the presentation for the correct audience.

### Preperation:
As a group we always prepared the presentation together. Because our team was 
split up in different specialities we always came together before the 
presentation to discuss our progress and issues. This was than converted into a presentation. 
We tried to get everyone to present as often as possible in order for everyone to do their part. 
This meant presenting once every two weeks. We also tried to present in groups 
of two where there is at least one student who is working primarily on the code and one student who focuses more on the report.

### List of presentations:
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


## Scrum:
At the start of the course we were advised to use some kind of scrum methodology to keep track of our project.
We decided to use Scrumwise to keep track of our sprint progress. 
In this chapter I'm going to pick out certain tasks that I worked on. 
I will explain my progress and what I have learned from working on this task. 
As well as the final result and how it was useful for our project. 

###








