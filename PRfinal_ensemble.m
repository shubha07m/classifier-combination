load P; 
load T;

P0 = P;
P1 = (P0([2 8 12 16 22 24 25 27],:));
P2 = (P0([10 21 22 24 25 27],:));
P3 = (P0([22 24 25 29],:));
P4 = (P0([5 8 22 23 24 25 28],:));

for j = 0:4
    if(j==0)
        P = P0;
    
    elseif(j==1)
        P = P1;
    
    elseif(j==2)
        P = P2;
        
    elseif(j==3)
        P = P3;
    
    elseif(j==4)
        P = P4;
    end
      
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Dividing feature and target data set to train and test
    [trainP,~,testP,trainInd,valInd,testInd] = dividerand(P,0.7,0,0.3);
    [trainT,~,testT] = divideind(T,trainInd,valInd,testInd);
    
    %% calculating score and ROC for selected classifiers
    [svm_rbf_score, acc_svm_rbf] = (mysvmfunc_lite(trainP,trainT,testP,testT,1));
    [X,Y,~,AUC_svm_rbf] = perfcurve([zeros(103,1);ones(68,1)]',svm_rbf_score(:,2)',1);
    if(j==0)
        figure;
        plot(X,Y,'b','LineWidth',3)
        xlabel('False positive rate'); ylabel('True positive rate');
        title(['test ROC for RBF SVM - with AUC ' num2str(AUC_svm_rbf)])
        savefig('ROC_svm_rbf.fig')
    end
    svm_rbf_score = mapminmax(svm_rbf_score);
    
    
    [svm_lin_score, acc_svm_lin] = (mysvmfunc_lite(trainP,trainT,testP,testT,2));
    [X,Y,~,AUC_svm_lin] = perfcurve([zeros(103,1);ones(68,1)]',svm_lin_score(:,2)',1);
    if(j==0)
        figure;
        plot(X,Y,'b','LineWidth',3)
        xlabel('False positive rate'); ylabel('True positive rate');
        title(['test ROC for linear SVM - with AUC ' num2str(AUC_svm_lin)])
        savefig('ROC_svm_linear.fig')
    end
    svm_lin_score = mapminmax(svm_lin_score);
    
    [svm_2ndpoly_score,acc_2ndpoly] = (mysvmfunc_lite(trainP,trainT,testP,testT,3));
    [X,Y,~,AUC_svm_2ndpoly] = perfcurve([zeros(103,1);ones(68,1)]',svm_2ndpoly_score(:,2)',1);
    if(j==0)
        figure;
        plot(X,Y,'b','LineWidth',3)
        xlabel('False positive rate'); ylabel('True positive rate');
        title(['test ROC for second order polynomial - with AUC ' num2str(AUC_svm_2ndpoly)])
        savefig('ROC_svm_2ndpoly.fig')
    end
        svm_2ndpoly_score = mapminmax(svm_2ndpoly_score);
    
    
    [svm_3rdpoly_score,acc_3rdpoly] = (mysvmfunc_lite(trainP,trainT,testP,testT,4));
    [X,Y,~,AUC_svm_3rdpoly] = perfcurve([zeros(103,1);ones(68,1)]',svm_3rdpoly_score(:,2)',1);
    if(j==0)
        figure;
        plot(X,Y,'b','LineWidth',3)
        xlabel('False positive rate'); ylabel('True positive rate');
        title(['test ROC for third order polynomial - with AUC ' num2str(AUC_svm_3rdpoly)])
        savefig('ROC_svm_3rdpoly.fig')
    end
        svm_3rdpoly_score = mapminmax(svm_3rdpoly_score);
    
    
    %% Defining training and testing of both the classes for prediction
    
    C1=trainP(:,(find(trainT==-1))); % Defining the benign sample as class one or C1
    C2=trainP(:,(find(trainT==1))); % Defining the cancerous sample as class two or C2
    C1Test=testP(:,(find(testT==-1))); % Performing the same for test data set
    C2Test=testP(:,(find(testT==1))); % Performing the same for test data set
    
    target_test = [-ones(size(C1Test,2),1);ones(size(C2Test,2),1)];
    testlength = length(testT);
    
    %% performance of sum rule
    
    sum_rule = (svm_rbf_score + svm_lin_score + svm_2ndpoly_score + svm_3rdpoly_score)/4;
    [X,Y,~,AUC_sum_rule] = perfcurve([zeros(103,1);ones(68,1)]',sum_rule(:,2)',1);
    if(j==0)
        figure;
        plot(X,Y,'b','LineWidth',3)
        xlabel('False positive rate'); ylabel('True positive rate');
        title(['test ROC for sum rule - with AUC ' num2str(AUC_sum_rule)])
        savefig('ROC_sum_rule.fig')
    end
    
    % Creating predictions from newly created sum rule classifier
    pred_test_sum = [];
    
    for i = 1:testlength
        if(sum_rule(i,1) > 0)
            pred_test_sum = horzcat(pred_test_sum,-1);
        else
            pred_test_sum = horzcat(pred_test_sum,1);
        end
    end
    
    c = confusionmat(pred_test_sum',target_test);
    acc_sum_rule = ((c(1,1) + c(2,2))/sum(c,'all'))*100;
    
    
    %% performance of max rules
    
    max_rule_rbf_lin = max(svm_rbf_score, svm_lin_score ); % calculating max between RBF svm and linear SVM score matrix
    
    max_rule_poly = max(svm_2ndpoly_score , svm_3rdpoly_score); % calculating max between 2nd order polynomial svm and 3r order polynomial SVM score matrix
    
    max_rule_combo = max(max_rule_rbf_lin, max_rule_poly); % calculating max between last two max calculated
    
    
    [X,Y,~,AUC_max_rule_combo] = perfcurve([zeros(103,1);ones(68,1)]',max_rule_combo(:,2)',1);
    if(j==0)
        figure;
        plot(X,Y,'b','LineWidth',3)
        xlabel('False positive rate'); ylabel('True positive rate');
        title(['test ROC for max rule combination- with AUC ' num2str(AUC_max_rule_combo)])
        savefig('ROC_max_rule.fig')
    end
    
    % Creating predictions from newly created max rule classifier
    pred_test_combo = [];
    
    for i = 1:testlength
        if(max_rule_combo(i,1) > 0)
            pred_test_combo = horzcat(pred_test_combo,-1);
        else
            pred_test_combo = horzcat(pred_test_combo,1);
        end
    end
    
    
    c = confusionmat(pred_test_combo',target_test);
    acc_max_rule_combo = ((c(1,1) + c(2,2))/sum(c,'all'))*100;
    
    
    %% performance of min rules
     
    min_rule_rbf_lin = min(svm_rbf_score, svm_lin_score );
    
    min_rule_poly = min(svm_2ndpoly_score , svm_3rdpoly_score);
    
    min_rule_combo = min(min_rule_rbf_lin, min_rule_poly);
    
    
    [X,Y,~,AUC_min_rule_combo] = perfcurve([zeros(103,1);ones(68,1)]',min_rule_combo(:,2)',1);
    if(j==0)
        figure;
        plot(X,Y,'b','LineWidth',3)
        xlabel('False positive rate'); ylabel('True positive rate');
        title(['test ROC for min rule combination- with AUC ' num2str(AUC_min_rule_combo)])
        savefig('ROC_min_rule.fig')
    end
    
    % Creating predictions from newly created min rule classifier
    pred_test_min = [];
    
    for i = 1:testlength
        if(min_rule_combo(i,1) > 0)
            pred_test_min = horzcat(pred_test_min,-1);
        else
            pred_test_min = horzcat(pred_test_min,1);
        end
    end
    
    
    c = confusionmat(pred_test_min',target_test);
    acc_min_rule_combo = ((c(1,1) + c(2,2))/sum(c,'all'))*100;
    
    
    %% performance of product rules
    
    product_rule = (svm_rbf_score.*svm_lin_score.*svm_2ndpoly_score.*svm_3rdpoly_score);
    %product_rule= mapminmax(product_rule);
    [X,Y,~,AUC_product_rule] = perfcurve([zeros(103,1);ones(68,1)]',product_rule(:,2)',1);
    if(j==0)
        figure;
        plot(X,Y,'b','LineWidth',3)
        xlabel('False positive rate'); ylabel('True positive rate');
        title(['test ROC for product rule- with AUC ' num2str(AUC_product_rule)])
        savefig('ROC_product_rule.fig')
    end
    
    % Creating predictions from newly created product rule classifier
    pred_test_product = [];
    
    for i = 1:testlength
        if(product_rule(i,1) > 0)
            pred_test_product = horzcat(pred_test_product,-1);
        else
            pred_test_product = horzcat(pred_test_product,1);
        end
    end
    
    
    c = confusionmat(pred_test_product',target_test);
    acc_product_rule = ((c(1,1) + c(2,2))/sum(c,'all'))*100;
    
    
    %% clearing not necessary variables for better result visibility
    clearvars X Y trainP testP testP trainT testT trainInd testInd valInd target_test testlength
    clearvars i c C1 C2 C1Test C2Test max_rule_poly max_rule_combo min_rule_combo min_rule_poly
    clearvars pred_test_min pred_test_combo pred_test_product pred_test_sum
    clearvars max_rule_rbf_lin min_rule_rbf_lin sum_rule product_rule
    clearvars svm_lin_score svm_rbf_score svm_2ndpoly_score svm_3rdpoly_score
    close all;
    
    save(['iter', num2str(j), '.mat']);

end