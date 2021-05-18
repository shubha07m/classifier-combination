
function [acc] = svm_disc(trainFeature,trainTarget,testFeature,testTarget,box,ktype)
   
    
    if(ktype == 1) % Calculating prediction performance for RBF kernel type
        
        if(box == 1)
            DISCR_svm = fitcsvm(trainFeature',trainTarget','Standardize',true,'KernelFunction','RBF', 'KernelScale','auto','BoxConstrainTarget',2e-1); % Using BoxConstrainTarget = .2
        elseif(box == 2)
            DISCR_svm = fitcsvm(trainFeature',trainTarget','Standardize',true,'KernelFunction','RBF', 'KernelScale','auto'); % Using default BoxConstrainTarget = 1
        elseif(box == 3)
            DISCR_svm = fitcsvm(trainFeature',trainTarget','Standardize',true,'KernelFunction','RBF', 'KernelScale','auto','BoxConstrainTarget',10); % Using BoxConstrainTarget = 10
        end
      
        TSTpred = predict(DISCR_svm,testFeature');
        c = confusionmat(TSTpred,testTarget');
        acc = ((c(1,1) + c(2,2))/sum(c,'all'))*100;
            
            
    elseif(ktype == 2) % Calculating prediction performance for linear kernel type
        
        if(box == 1)
            DISCR_svm = fitcsvm(trainFeature',trainTarget','Standardize',true,'KernelFunction','linear', 'KernelScale','auto','BoxConstrainTarget',2e-1);
        elseif(box == 2)
            DISCR_svm = fitcsvm(trainFeature',trainTarget','Standardize',true,'KernelFunction','linear', 'KernelScale','auto');
        elseif(box == 3)
            DISCR_svm = fitcsvm(trainFeature',trainTarget','Standardize',true,'KernelFunction','linear', 'KernelScale','auto','BoxConstrainTarget',10);
        end
       
            TSTpred = predict(DISCR_svm,testFeature');
            c = confusionmat(TSTpred,testTarget');
            acc = ((c(1,1) + c(2,2))/sum(c,'all'))*100;
            
            
        
    elseif(ktype == 3) % Calculating prediction performance for second order polynomial kernel type
        
        if(box == 1)
            DISCR_svm = fitcsvm(trainFeature',trainTarget','Standardize',true,'KernelFunction','Polynomial', 'PolynomialOrder',2, 'KernelScale','auto','BoxConstrainTarget',2e-1);
        elseif(box == 2)
            DISCR_svm = fitcsvm(trainFeature',trainTarget','Standardize',true,'KernelFunction','Polynomial', 'PolynomialOrder',2, 'KernelScale','auto');
        elseif(box == 3)
            DISCR_svm = fitcsvm(trainFeature',trainTarget','Standardize',true,'KernelFunction','Polynomial', 'PolynomialOrder',2, 'KernelScale','auto','BoxConstrainTarget',10);
        end
       
        TSTpred = predict(DISCR_svm,testFeature');
        c = confusionmat(TSTpred,testTarget');
        acc = ((c(1,1) + c(2,2))/sum(c,'all'))*100;
            
            
        
    elseif(ktype == 4)  % Calculating prediction performance for third order polynomial kernel type
        
        if(box == 1)
            DISCR_svm = fitcsvm(trainFeature',trainTarget','Standardize',true,'KernelFunction','Polynomial', 'PolynomialOrder',3, 'KernelScale','auto','BoxConstrainTarget',2e-1);
        elseif(box == 2)
            DISCR_svm = fitcsvm(trainFeature',trainTarget','Standardize',true,'KernelFunction','Polynomial', 'PolynomialOrder',3, 'KernelScale','auto');
        elseif(box == 3)
            DISCR_svm = fitcsvm(trainFeature',trainTarget','Standardize',true,'KernelFunction','Polynomial', 'PolynomialOrder',3, 'KernelScale','auto','BoxConstrainTarget',10);
        end
       
       
        TSTpred = predict(DISCR_svm,testFeature');
        c = confusionmat(TSTpred,testTarget');
        acc = ((c(1,1) + c(2,2))/sum(c,'all'))*100;
            
    end
 