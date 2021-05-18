load P; 
load T;

%% Dividing feature and target data set to train and test
[trainP,valP,testP,trainInd,valInd,testInd] = dividerand(P,0.7,0,0.3);
[trainT,valT,testT] = divideind(T,trainInd,valInd,testInd);


%% Defining class one and two and defining test sample
C1=trainP(:,(find(trainT==-1))); % Defining the benign sample as class one or C1
C2=trainP(:,(find(trainT==1))); % Defining the cancerous sample as class two or C2
C1Test=testP(:,(find(testT==-1))); % Performing the same for test data set 
C2Test=testP(:,(find(testT==1))); % Performing the same for test data se

% Transposing the train / test matrices for expected shape
trainP = trainP';
testP = testP';
trainT = trainT';
testT = testT';


cvp = cvpartition(trainT,'k',10); % Applying 10 fold cross validation
opt = statset('display','iter'); % Applying options for print the info after everey iteration


fun1 = @(trainP,trainT,testP,testT)loss(fitcsvm(trainP,trainT,'Standardize',true,'KernelFunction','RBF','KernelScale','auto'),testP,testT); % Function for Getting optimal feature set for RBF kernel type
fun2 = @(trainP,trainT,testP,testT)loss(fitcsvm(trainP,trainT,'Standardize',true,'KernelFunction','linear','KernelScale','auto'),testP,testT); % Function for Getting optimal feature set for linear kernel type
fun3 = @(trainP,trainT,testP,testT)loss(fitcsvm(trainP,trainT,'Standardize',true,'KernelFunction','Polynomial','PolynomialOrder',2, 'KernelScale','auto'),testP,testT); % Function for Getting optimal feature set for second order polynomial kernel type
fun4 = @(trainP,trainT,testP,testT)loss(fitcsvm(trainP,trainT,'Standardize',true,'KernelFunction','Polynomial','PolynomialOrder',3, 'KernelScale','auto'),testP,testT); % Function for Getting optimal feature set for third order polynomial kernel type



[fs1,history1] = sequentialfs(fun1,trainP,trainT,'cv',cvp,'options',opt);  % displaying optimal feature set type 1
[fs2,history2] = sequentialfs(fun2,trainP,trainT,'cv',cvp,'options',opt);  % displaying optimal feature set type 2
[fs3,history3] = sequentialfs(fun3,trainP,trainT,'cv',cvp,'options',opt);  % displaying optimal feature set type 3
[fs4,history4] = sequentialfs(fun4,trainP,trainT,'cv',cvp,'options',opt);  % displaying optimal feature set type 4




