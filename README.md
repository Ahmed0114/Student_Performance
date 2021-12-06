# Student Performance: An Exploratory Data Analysis on Student Performance Prediction 

## Table of Contents
* [Dataset](https://github.com/Bayunova28/Student_Performance/tree/main/dataset)
* [Code](https://github.com/Bayunova28/Student_Performance/blob/main/student-performance.ipynb)
* [Background](#background)
* [Exploratory Data Analysis](#exploratory-data-analysis)

## Background
<img src="https://github.com/Bayunova28/Student_Performance/blob/main/images/85fd215613e6e9ed88281879187f0ee1.png" width="1000">
This data approach student achievement in secondary education of two Portuguese schools. The data attributes include student grades, demographic, social and school related features) and it was collected by using school reports and questionnaires. Two datasets are provided regarding the performance in two distinct subjects: Mathematics (mat) and Portuguese language (por). In Cortez and Silva, 2008, the two datasets were modeled under binary/five-level classification and regression tasks. Important note: the target attribute G3 has a strong correlation with attributes G2 and G1. This occurs because G3 is the final year grade (issued at the 3rd period), while G1 and G2 correspond to the 1st and 2nd period grades. It is more difficult to predict G3 without G2 and G1, but such prediction is much more useful (see paper source for more details)

## Exploratory Data Analysis

### Correlation
![](https://github.com/Bayunova28/Student_Performance/blob/main/images/__results___12_0.png)

This is correlation for each attribute in student performance data, target of prediction is G3. On the picture above, there are 16 attributes which target for prediction has positive correlation

### Machine Learning Model
For prediction target, Machine Learning Algorithm used Linear Regression Model with accuracy score 89%, Mean Absolute Error on 0.84, Mean Squared Error on 1.66 and Root Mean Squared Error on 0.91

### Visualization
![](https://github.com/Bayunova28/Student_Performance/blob/main/images/78.png)

On the picture above, there are 2 gender based on school type of student performance. In fact, the most gender based on school type GP and MS is female rather than male

![](https://github.com/Bayunova28/Student_Performance/blob/main/images/__results___38_0.png)

On the picture above, there are 2 gender based on 4 reason of student performance. In fact, the most gender based on 4 reason like course, home, other and reputation is female rather than male
