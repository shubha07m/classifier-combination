load P;
load T;

[trainP,~,testP,trainInd,valInd,testInd] = dividerand(P,0.7,0,0.3);
[trainT,~,testT] = divideind(T,trainInd,valInd,testInd);

[test_acc_flda] = fisherlda_disc(trainP,trainT,testP,testT);



[test_acc_linear] = misc_disc(trainP,trainT,testP,testT,1);

[test_acc_diaglinear] = misc_disc(trainP,trainT,testP,testT,2);

[test_acc_quad] = misc_disc(trainP,trainT,testP,testT,3);

[test_acc_diagquad] = misc_disc(trainP,trainT,testP,testT,4);




[test_acc_svm_rbf] = svm_disc(trainP,trainT,testP,testT,2,1);

[test_acc_svm_linear] = svm_disc(trainP,trainT,testP,testT,2,2);

[test_acc_svm_2ndpoly] = svm_disc(trainP,trainT,testP,testT,2,3);

[test_acc_svm_3rdpoly] = svm_disc(trainP,trainT,testP,testT,2,4);



clearvars P T trainP testP testP trainT testT trainInd testInd valInd
