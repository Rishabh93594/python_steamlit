import numpy as np  
import pandas as pd 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 
from sklearn import svm 
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer




dataset = pd.read_csv("dementia_dataset.csv")

# print(dataset.head())

# print(dataset.isnull())

# print(dataset.isnull().sum())

# print(dataset.isnull().sum().sum())


simple_median = SimpleImputer(strategy='median')
dataset['SES'] = simple_median.fit_transform(dataset[['SES']])
dataset['MMSE'] = simple_median.fit_transform(dataset[['MMSE']])


# print(dataset.isnull().sum())

# print(dataset.describe())




# print(dataset['Group'].value_counts())

dataset = dataset.replace(to_replace='Nondemented',value= '0')
dataset = dataset.replace(to_replace='Demented',value= '1')
dataset = dataset.replace(to_replace='Converted',value= '2')


## Male--->> 0
## Female--->> 1
dataset = dataset.replace(to_replace='M',value= '0')
dataset = dataset.replace(to_replace='F',value= '1')

## Right hand--->> 0
## Left hand--->> 1

dataset = dataset.replace(to_replace='R',value= '0')
dataset = dataset.replace(to_replace='L',value= '1')




y = dataset['Group']



x = dataset.drop(columns=['Subject ID','MRI ID','Group'], axis=1)


x = x.astype(int)


# dataset.to_csv("dementia_new_dataset.csv")


### Data Standardization 
Stand = StandardScaler()
x = x.values
Stand.fit(x)
Stand_data = Stand.transform(x)

x = Stand_data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)

classifier = svm.SVC(kernel='linear')
classifier.fit(x_train,y_train)




x_train_predit = classifier.predict(x_train)
training_data_acc = accuracy_score(x_train_predit,y_train)

print("Accuracy score of x_train is ",training_data_acc)



x_test_predit = classifier.predict(x_test)
testing_data_acc = accuracy_score(x_test_predit,y_test)

print("Accuracy score of x_test is ",testing_data_acc)

input_data = (1,0,0,0,72,20,1.0,26.0,0.5,1911,0.719,0.919)

### changing the list data into numpy arrays
input_data_as_numpy = np.asarray(input_data)


# this reshape tell the model we need the prediction for only one data
input_data_reshape = input_data_as_numpy.reshape(1,-1)

# standardize the data to get the output
# std_data = Stand.transform(input_data_reshape)
# print(std_data)

# printing the prediction 
prediction = classifier.predict(input_data_reshape)
print(prediction)

# printing the prediction 
if(prediction[0] == '0'):
    print("The patient is NonDemented")
elif(prediction[0] == '1'):
    print("The patient is Demented")
else:
    print("The patient is Converted")


### Saving the Trained Model
import pickle

filename = 'trained_model.sav'
pickle.dump(classifier, open(filename, 'wb'))

### Loading the Saved Model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

input_data = (1,0,0,0,72,20,1.0,26.0,0.5,1911,0.719,0.919)

### changing the list data into numpy arrays
input_data_as_numpy = np.asarray(input_data)


# this reshape tell the model we need the prediction for only one data
input_data_reshape = input_data_as_numpy.reshape(1,-1)

# standardize the data to get the output
# std_data = Stand.transform(input_data_reshape)
# print(std_data)

# printing the prediction 
prediction = loaded_model.predict(input_data_reshape)
print(prediction)

# printing the prediction 
if(prediction[0] == '0'):
    print("The patient is NonDemented")
elif(prediction[0] == '1'):
    print("The patient is Demented")
else:
    print("The patient is Converted")

