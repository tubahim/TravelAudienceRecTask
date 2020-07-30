
# Travel Audience Data Scientist Recruiting Task -Tuba Yaman Him

## Business Problem

Business Problem: What is the likelihood of booking to be performed by every individual user.

## How to Run

Jupyter notebook is used for this problem. Please run the notebooks according to their orders:
<ol>
<li>Features
<li>EDA
<li>Model
</ol>

<b>Note:</b> if the magic cell does not work in Features Notebook use below code to install requirements
```bash
pip install -r requirements.txt
```

## 1-Feature Preparation


Before beginning the task I needed to work on data in order to generate features for my model. I have followed below steps in order to achieve my goal:

<ol>
<li>First I needed to join original data provided with geographical data by using join keys. </li> 
<li>Secondly, I have decided to calculate features based on given rows which represents transactions.I have named this mi-phase dataframe as 'staging dataframe'</li> 
    
In the staging dataframe I have generated below features:

 <b>vac_duration</b>= Indicates the duration of the booked/searched vacation in days.<br>
 <b>book_vac_duration</b>=The lag between booking/searching and the beginning  of the vacation in days<br>
 <b>book_days_ago</b>= Indicates how many days ago booking/searching is performed.<br>
 <b>child_per_adult</b>= It is basically the number of children per adult.<br>
 <b>booked</b>= Indicates if the action is booking or not. 1 indicates booking, 0 indicates search.<br>
 <b>searched</b>=Indicates if the action is searching or not. 1 indicates searching, 0 indicates book.<br>
 <b>is_summer</b>=Indicates if the vacation period is in summer or not. <br>
 <b>christmas</b>=Indicates if the vacation period is at Christmas or not. <br>
 <b>vac_distance1</b>= It represents the distance from the origin to destination in KMs. Geopy package is utilized.<br>
 <b>vac_distance2</b>= It represents the distance from the origin to destination in KMs. Math formula is utilized.<br>
 <b>Note:</b> I wanted to compare 2 different distance calculations.<br>
 
 <li> In the staging dataframe all features were calculated based on transaction rows. But in the model, it is a better idea to utilize user-based features since each prediction will be performed for each individual user. That's why I decided to aggregate features on the user level. I have classified some attributes to be summed and some to be averaged. Then I have basically aggregated them </li> 
 </ol>

## 2-EDA

Before diving into the model I wanted to dig the data in order have enough insights about how to approach the problem and understand which information I can gain from this dataset.<br>

<li>As a first step,I wanted to check correlations between features. The results did not surprise me I was expecting that num_children and children_per_adult features are highly correlated to each other. 
<li>I have also checked the distribution of distances. As seen in the graphs most of the vacations are planned within 5000kms.
<li> I needed to analyze the data on deeper levels. But unfortunately I did not have enough time for extra vvisualizations.

## Model Training for The Business Problem

Business Problem: What is the likelihood of booking to be performed.
What Needs To Be Done: Calculate the likelihood of a user to perform booking. It is a binary classification problem
### My Approach
First of all, I thought about splitting  the dataset according to the time frame. I thought I could use the first 16 days as train data and the last 4 days as validation data and my model would predict the bookings for the next 4 days. Then I decided that data size and time period of 20 days is not sufficient  for this approach.
Then I decided to predict booking likelihood without considering any time frame.

I have chosen 'is_booked' as target variable. I have dropped 'booked' feature which indicates no of bookings performed by a user. Because it is an obvious clue for 'is_booking' target.
I have used %20 of data as validation dataset.
### Model
I thought I can utilize trees for this kind of binary classification. I have chosen Lightgbm because it is easy to use and I am familiar with it.

Before running the training I have used Hyperopt to find out optimal parameters for Lightgbm. But I did not want Hyperopt to decide on learning rate since it can cause overfitting. I have changed the learning rate and l1 regularization value according to the standard deviation of cross-validation results.

The objective was obviously 'Binary' and I thought for likelihood calculation it is a good idea to utilize log loss. 

### Important Features
I have utilized Lightgbm's importance plot to find out the most important features. Below you could find top 3 features:

<b>origin_lat:</b> I am a little bit surprised that this is the most important feature. I was expecting search has the most effect on booking decisions. But it seems that the latitude of the origin has more importance.<br>
<b>searched:</b> It was an expected result for me. Users who performed searches would be more likely to book a vacation.<br>
<b>origin_lon:</b>  It seems that this feature is also important for booking decisions like origin_lat.<br>

### Evaluation
I have tried different evaluation metrics. The model already gives me the binary log loss value.

<li>For training binary_logloss: 0.0871094	for validation binary_logloss: 0.0881209

<li>For validation dataset: AUC is 0.96916, rmse is 0.02347, precision is 0.97849, accuracy is 0.96969 

<li>I have also tried RMSE since likelihood calculation is a kind of regression other than a classification.

<li>I think the most suitable evaluation metric for marketing binary classifications  is precision since it evaluates the accuracy of positive predictions.

### Cross-Validation

According to cross validation results min logloss is 0.0895635 maximum logloss is 0.2215380, average is 0.1149708 
I am suspicious about this high accurate results. But I did not have time to dig deeper.

### Notes

I did not have time to refactor the code. Especially, feature preparation notebook could be refactored in order to make it more readable. <br>
I also wish I had time to add extra data visualizations.

### Extra Business Problem

I have used the model for an extra business problem. I was curious about how model performs for a specific destination. I have chosen Berlin as a target destination. You could see the results in the notebook named 'Extra'.
