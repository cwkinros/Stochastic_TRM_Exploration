function [] = run_w_params_plus_test(dataset,tests,maxiter,sub_maxiter)
% here we've set up all the datasets with their params

[sgd_lr,sgd_lr_mb,lr,lr_mb,b_m_mini,b_m_mini_MBGD,b_m_big,b_w] = getParams(dataset);
run_test(dataset,sgd_lr,sgd_lr_mb,lr,lr_mb,b_w,b_m_mini,b_m_mini_MBGD,b_m_big,tests,true,'sub_maxiter',sub_maxiter,maxiter,10,sub_maxiter,false,30*60);

function [sgd_lr,sgd_lr_mb,lr,lr_mb,b_m_mini,b_m_mini_MBGD,b_m_big,b_w] = getParams(dataset)

params = readtable(strcat('results/',dataset,'params_results.txt'));
sgd_lr = params.sgdlr;
sgd_lr_mb = params.sgdlrmb;
lr = params.lr;
lr_mb = params.lrmb;
b_w = params.bw;
b_m_mini = params.bmmini;
b_m_mini_MBGD = params.bmmini_MBGD;
b_m_big = params.bmbig;