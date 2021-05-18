function[acc] =  misc_disc(trainFeature,trainTarget,testFeature,testTarget,distype)

    
    %% Defining class one and two and defining test sample
    C1=trainFeature(:,(find(trainTarget==-1))); % Defining the benign sample as class one or C1
    C2=trainFeature(:,(find(trainTarget==1))); % Defining the cancerous sample as class two or C2
    C1Test=testFeature(:,(find(testTarget==-1))); % Performing the same for test data set 
    C2Test=testFeature(:,(find(testTarget==1))); % Performing the same for test data se


    %% Creating various types of discriminant for classification

    if(distype ==1)

        DISCR_lin=fitcdiscr([C1';C2'],[-ones(size(C1,2),1);ones(size(C2,2),1)],'DiscrimType','linear');
        [TSTpred_lin,~, ~]  = predict(DISCR_lin,[C1Test';C2Test']);
        cm_test = confusionmat(TSTpred_lin,[-ones(size(C1Test,2),1);ones(size(C2Test,2),1)]);
        acc = ((cm_test(1,1) + cm_test(2,2))/sum(cm_test,'all'))*100;
        %f2 = ((cm_test(1,1) + cm_test(2,2))/sum(cm_test,'all'))*100;

    elseif(distype ==2)

        DISCR_diaglin=fitcdiscr([C1';C2'],[-ones(size(C1,2),1);ones(size(C2,2),1)],'DiscrimType','diagLinear');
        [TSTpred_diaglin, ~, ~] = predict(DISCR_diaglin,[C1Test';C2Test']);
        cm_test = confusionmat(TSTpred_diaglin,[-ones(size(C1Test,2),1);ones(size(C2Test,2),1)]);
        acc = ((cm_test(1,1) + cm_test(2,2))/sum(cm_test,'all'))*100;
        %f2 = ((cm_test(1,1) + cm_test(2,2))/sum(cm_test,'all'))*100;

    elseif(distype ==3)

            DISCR_quad=fitcdiscr([C1';C2'],[-ones(size(C1,2),1);ones(size(C2,2),1)],'DiscrimType','quadratic');
            [TSTpred_quad, ~, ~] = predict(DISCR_quad,[C1Test';C2Test']);
            cm_test = confusionmat(TSTpred_quad,[-ones(size(C1Test,2),1);ones(size(C2Test,2),1)]);
            acc = ((cm_test(1,1) + cm_test(2,2))/sum(cm_test,'all'))*100;
            %f2 = ((cm_test(1,1) + cm_test(2,2))/sum(cm_test,'all'))*100;

    elseif(distype ==4)

            DISCR_diagquad=fitcdiscr([C1';C2'],[-ones(size(C1,2),1);ones(size(C2,2),1)],'DiscrimType','diagQuadratic');
            [TSTpred_diagquad,~, ~] = predict(DISCR_diagquad,[C1Test';C2Test']);
            cm_test = confusionmat(TSTpred_diagquad,[-ones(size(C1Test,2),1);ones(size(C2Test,2),1)]);
            acc = ((cm_test(1,1) + cm_test(2,2))/sum(cm_test,'all'))*100;
            %f2 = ((cm_test(1,1) + cm_test(2,2))/sum(cm_test,'all'))*100;

    end