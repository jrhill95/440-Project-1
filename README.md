
CAS CS 440/660 Artificial Intelligence - Spring 2018
This assignment is due on Wednesday, Feburary 28, at 00:00 EST (i.e. the night from Wednesday to Thursday). You are encouraged to work in teams of two students. Each student must submit code along with the report.
Programming Assignment 1
In this assignment, you will build upon the ideas you learned in class and in the labs. Your team will design and implement a simple neural network.

Each team must write a webpage report on the assignment. Use, for example, http://www.cs.bu.edu/faculty/betke/cs440/restricted/p1/p1-template.html. To ensure that each team member builds his or her own electronic portfolio, we ask that everybody submits his or her own report.

Learning Objectives
Understand how neural networks work
Implement a simple neural network
Understand the role of different parameters of a neural network, such as learning rate.
You need to do
Refering to the logistic regression code provided in Lab 3, implement a 2-layer neural network (without any hidden layers). You will have to implement/modify 3 functions: compute_cost(), predict() and fit(). Train this model using the dataset DATA/LinearX.csv, DATA/LinearY.csv and visualize the decision boundary learned by your model.
Now, train your neural network model using the dataset DATA/NonlinearX.csv, DATA/NonlinearY.csv and visualize the decision boundary learned by your model. Can your 2-layer neural network model learn non-linear decision boundaries? Why or why not?
Implement a neural network class with 1 hidden layer. Train this model using the dataset DATA/LinearX.csv, DATA/LinearY.csv and visualize the decision boundary learned by your model. Then, train your neural network model using the dataset DATA/NonlinearX.csv, DATA/NonlinearY.csv and visualize the decision boundary learned by your model. Can your 3-layer neural network model (with one hidden layer) learn non-linear decision boundaries? Why or why not?
What effect does learning rate have on how your neural network is trained? Illustrate your answer by training your model using different learning rates.
What effect does the number of nodes in the hidden layer have on how your neural network is trained? Illustrate your answer by training your model using differnet numbers of hidden layer nodes.
What is overfitting and why does it occur in practice? Name and briefly explain 3 ways to reduce overfitting.
One common technique used to reduce overfitting is L2 regularization. How does L2 regularization prevent overfitting? Implement L2 regularization. How differently does your model perform before and after implementing L2 regularization?
Extra Credit: Now, let's try to solve real world problem. You are given hand-written digits as below, each digit image has been stored in csv files. You need to implement the neural network class with 1 hidden layer to recognize the hand-written digits, you should train your model on DATA/Digit_X_train.csv and DATA/Digit_y_train.csv, then test your model on DATA/Digit_X_test.csv and DATA/Digit_y_test.csv. Show your results.
hand-written digits
To visualize each digit in csv, you could use the following codes:
X_train = np.genfromtxt('DATA/Digit_X_train.csv', delimiter=',')
data = np.reshape(255*X_train[i],(8,8)) #i--the ith digit
image = Image.fromarray(data)
image.show()
Submission The programming assignment, along with the webpage report, should be made into a .zip file and submitted using gsubmit.
Margrit Betke, Professor
Computer Science Department
Email: betke@cs.bu.edu
