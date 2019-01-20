# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('sample.csv')
dataset1 = pd.read_csv('mumbai-2017.csv')
dataset2 = pd.read_csv('mumbai-2018.csv')
dataset3 = pd.read_csv('pune-2017.csv')
dataset4 = pd.read_csv('pune-2018.csv')
dataset5 = pd.read_csv('nagpur-2017.csv')
dataset6 = pd.read_csv('nagpur-2018.csv')


#Assigning the training data
X = dataset.iloc[:, [0, 1, 2, 3]].values
y = dataset.iloc[:, [4]].values


#Assigning the test data
X_test1 = dataset1.iloc[:, [0, 1, 2, 3]].values
X_test2 = dataset2.iloc[:, [0, 1, 2, 3]].values
X_test3 = dataset3.iloc[:, [0, 1, 2, 3]].values
X_test4 = dataset4.iloc[:, [0, 1, 2, 3]].values
X_test5 = dataset5.iloc[:, [0, 1, 2, 3]].values
X_test6 = dataset6.iloc[:, [0, 1, 2, 3]].values


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

X_test1 = sc_X.transform(X_test1)
X_test2 = sc_X.transform(X_test2)
X_test3 = sc_X.transform(X_test3)
X_test4 = sc_X.transform(X_test4)
X_test5 = sc_X.transform(X_test5)
X_test6 = sc_X.transform(X_test6)


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)

#setting scales for plot
y_pred1f = [[0.1]]
y_pred2f = [[0.09]]
y_pred3f = [[0.1]]
y_pred4f = [[0.09]]
y_pred5f = [[0.1]]
y_pred6f = [[0.09]]

# Predicting the Test set results
y_pred1 = regressor.predict(X_test1)
y_pred2 = regressor.predict(X_test2)
y_pred3 = regressor.predict(X_test3)
y_pred4 = regressor.predict(X_test4)
y_pred5 = regressor.predict(X_test5)
y_pred6 = regressor.predict(X_test6)

#Adjusting values to scale
y_pred1 = np.multiply(y_pred1,y_pred1f)
y_pred2 = np.multiply(y_pred2,y_pred2f)
y_pred3 = np.multiply(y_pred3,y_pred3f)
y_pred4 = np.multiply(y_pred4,y_pred4f)
y_pred5 = np.multiply(y_pred5,y_pred5f)
y_pred6 = np.multiply(y_pred6,y_pred6f)



X_plot =[1,2,3,4,5,6,7,8,9,10,11,12]

plt.plot(X_plot, y_pred1, color = 'blue', label='mumbai')
plt.plot(X_plot, y_pred3, color = 'red', label='pune')
plt.plot(X_plot, y_pred5, color = 'green', label='nagpur')
plt.title('Malaria prediction 2017')
plt.xlabel('months')
plt.ylabel('severity')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.savefig('mnp17.png')
plt.show()
plt.close()

plt.plot(X_plot, y_pred2, color = 'blue', label='mumbai')
plt.plot(X_plot, y_pred4, color = 'red', label='pune')
plt.plot(X_plot, y_pred6, color = 'green', label='nagpur')
plt.title('Malaria prediction 2018')
plt.xlabel('months')
plt.ylabel('severity')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.savefig('mnp18.png')
plt.show()
plt.close()

plt.plot(X_plot, y_pred1, color = 'brown', label='2017')
plt.plot(X_plot, y_pred2, color = 'pink', label='2018')
plt.title('Malaria prediction mumbai')
plt.xlabel('months')
plt.ylabel('severity')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.savefig('m.png')
plt.show()
plt.close()

plt.plot(X_plot, y_pred3, color = 'brown', label='2017')
plt.plot(X_plot, y_pred4, color = 'pink', label='2018')
plt.title('Malaria prediction pune')
plt.xlabel('months')
plt.ylabel('severity')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.savefig('n.png')
plt.show
plt.close()

plt.plot(X_plot, y_pred5, color = 'brown', label='2017')
plt.plot(X_plot, y_pred6, color = 'pink', label='2018')
plt.title('Malaria prediction nagpur')
plt.xlabel('months')
plt.ylabel('severity')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.savefig('p.png')
plt.show()
plt.close()

y_pred1l = y_pred1.tolist()
y_pred2l = y_pred2.tolist()
y_pred3l = y_pred3.tolist()
y_pred4l = y_pred4.tolist()
y_pred5l = y_pred5.tolist()
y_pred6l = y_pred6.tolist()

f= open("intensities-2017.txt","w+")
for i in range(0,12):
	sr = str(i+1)
	mum ,pun, nag = str(int(y_pred1l[i][0])), str(int(y_pred3l[i][0])), str(int(y_pred5l[i][0]))
	row = sr +","+ mum + "," + pun +","+ nag + "\n"
	f.write(row)
    
    
f.close()

f1= open("intensities-2018.txt","w+")
for i in range(0,12):
	sr = str(i+1)
	mum ,pun, nag = str(int(y_pred2l[i][0])), str(int(y_pred4l[i][0])), str(int(y_pred6l[i][0]))
	row = sr +","+ mum + "," + pun +","+ nag + "\n"
	f1.write(row)
    
    
f1.close()

