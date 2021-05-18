
function [TSTpred_svm_score,acc] = mysvmfunc_lite(trainFeature,trainTarget,testFeature,testTarget,ktype)

    C1=trainFeature(:,(find(trainTarget==-1))); % Defining the benign sample as class one or C1
    C2=trainFeature(:,(find(trainTarget==1))); % Defining the cancerous sample as class two or C2
    C1Test=testFeature(:,(find(testTarget==-1))); % Performing the same for test data set 
    C2Test=testFeature(:,(find(testTarget==1))); % Performing the same for test data set

   
    
    if(ktype == 1) % Calculating prediction performance for RBF kernel type
      
        DISCR_svm = fitcsvm([C1';C2'],[-ones(size(C1,2),1);ones(size(C2,2),1)],'Standardize',true,'KernelFunction','RBF', 'KernelScale','auto'); % Using default BoxConstrainTarget = 1
        [TSTpred_svm,TSTpred_svm_score, ~] = predict(DISCR_svm,[C1Test';C2Test']);
        c = confusionmat(TSTpred_svm,[-ones(size(C1Test,2),1);ones(size(C2Test,2),1)]);
        acc = ((c(1,1) + c(2,2))/sum(c,'all'))*100;
            
    elseif(ktype == 2) % Calculating prediction performance for linear kernel type
       
        DISCR_svm = fitcsvm([C1';C2'],[-ones(size(C1,2),1);ones(size(C2,2),1)],'Standardize',true,'KernelFunction','linear', 'KernelScale','auto'); % Using default BoxConstrainTarget = 1
        [TSTpred_svm,TSTpred_svm_score, ~] = predict(DISCR_svm,[C1Test';C2Test']);
        c = confusionmat(TSTpred_svm,[-ones(size(C1Test,2),1);ones(size(C2Test,2),1)]);
        acc = ((c(1,1) + c(2,2))/sum(c,'all'))*100;
            
        
    elseif(ktype == 3) % Calculating prediction performance for second order polynomial kernel type
       
        DISCR_svm = fitcsvm([C1';C2'],[-ones(size(C1,2),1);ones(size(C2,2),1)],'Standardize',true,'KernelFunction','Polynomial','PolynomialOrder',2, 'KernelScale','auto'); % Using default BoxConstrainTarget = 1
        [TSTpred_svm,TSTpred_svm_score, ~] = predict(DISCR_svm,[C1Test';C2Test']);
        c = confusionmat(TSTpred_svm,[-ones(size(C1Test,2),1);ones(size(C2Test,2),1)]);
        acc = ((c(1,1) + c(2,2))/sum(c,'all'))*100;
                
        
    elseif(ktype == 4)  % Calculating prediction performance for third order polynomial kernel type
        
        
        DISCR_svm = fitcsvm([C1';C2'],[-ones(size(C1,2),1);ones(size(C2,2),1)],'Standardize',true,'KernelFunction','Polynomial','PolynomialOrder',3, 'KernelScale','auto'); % Using default BoxConstrainTarget = 1
        [TSTpred_svm,TSTpred_svm_score, ~] = predict(DISCR_svm,[C1Test';C2Test']);
        c = confusionmat(TSTpred_svm,[-ones(size(C1Test,2),1);ones(size(C2Test,2),1)]);
        acc = ((c(1,1) + c(2,2))/sum(c,'all'))*100;
            
    end
end


