function [] = run_w_params(dataset,tests)
% here we've set up all the datasets with their params
maxiter = 10000000;
[sgd_lr,lr,b_m_mini,b_m_big,b_w] = getParams(dataset);
run_test(dataset,sgd_lr,lr,b_w,b_m_mini,b_m_big,tests,false,0,0,maxiter);

function [sgd_lr,lr,b_m_mini,b_m_big,b_w] = getParams(dataset)

params = readtable(strcat('results/',dataset,'params_results.txt'));
sgd_lr = params.sgdlr;
lr = params.lr;
b_w = params.bw;
b_m_mini = params.bmmini;
b_m_big = params.bmbig;