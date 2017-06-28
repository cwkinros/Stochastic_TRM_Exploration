function [WS,MS,TRMstep,GD,gamma] = getParams(test,lr,sgd_lr,b_m_mini)
    WS = false;
    MS = 0;
    TRMstep = false;
    GD = false;
    gamma = 1;

    if strcmp(test,'BTRM_WS')
        WS = true;
        MS = 2;
        TRMstep = true;
    else if strcmp(test,'BTRM')
            MS = 2;
            TRMstep = true;
        else if strcmp(test,'SGD')
                MS = 1;
                GD = true;
                gamma = sgd_lr;
            else if strcmp(test,'MBGD')
                    MS = 2;
                    GD = true;
                    gamma = sgd_lr;
                else if strcmp(test,'TRM')
                        TRMstep = true;
                    else if strcmp(test,'TRM_WS')
                            WS = true;
                            TRMstep = true;
                        else if strcmp(test,'STRM')
                                MS = 1;
                                gamma = lr;
                            else if strcmp(test,'STRM_WS')
                                    WS = true;
                                    MS = 1;
                                    gamma = lr;
                                else if strcmp(test,'MBTRM')
                                        MS = 2;
                                        gamma = lr;
                                    else if strcmp(test,'MBTRM_WS')
                                            WS = true;
                                            MS = 2;
                                            gamma = lr;
                                        else
                                            disp(strcat('Method: ',test,' is not covered'));
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end




