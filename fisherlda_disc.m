function[acc] =  fisherlda_disc(trainFeature,trainTarget,testFeature,testTarget)

    %% Defining class one and two and defining test sample
    C1=trainFeature(:,(find(trainTarget==-1))); % Defining the benign sample as class one or C1
    C2=trainFeature(:,(find(trainTarget==1))); % Defining the cancerous sample as class two or C2

    %% building fisher LDA and calculating w, J, m and reg
    [w, ~, m, ~]=LDA3(C1,C2);

    projected_value_train = w'*trainFeature-w' *m; % projecting the values for train
    known_train = trainTarget; % This is the ground truth / actual label for train
    projected_value_test = w'*testFeature-w' *m; % projecting the values for test
    known_test = testTarget; % This is the ground truth / actual label for test


    %% creating the prediction as per negative and positive for training
    predicted_train = zeros(size(projected_value_train));
    [i,j]=find(projected_value_train < 0);
    predicted_train(i,j)=-1;
    [i,j]=find(projected_value_train > 0);
    predicted_train(i,j)=1;


    %% creating the prediction as per negative and positive for test
    predicted_test = zeros(size(projected_value_test));
    [i,j]=find(projected_value_test < 0);
    predicted_test(i,j)=-1;
    [i,j]=find(projected_value_test > 0);
    predicted_test(i,j)=1;


    %% creating the confussion matrix and calculating accuracy for training and testing data
    cm_train = confusionmat(known_train, predicted_train);
    cm_test = confusionmat(known_test, predicted_test);
    acc = ((cm_test(1,1) + cm_test(2,2))/sum(cm_test,'all'))*100;