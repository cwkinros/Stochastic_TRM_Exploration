function [WS,MS,TRMstep,GD,gamma,b_m_mini_general] = getParams(test,lr,sgd_lr,lr_mb,sgd_lr_mb,b_m_mini,b_m_mini_MBGD)
WS = false;
MS = 0;
TRMstep = false;
GD = false;
gamma = 1;
b_m_mini_general = b_m_mini;
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
                b_m_mini_general = b_m_mini_MBGD;
                gamma = sgd_lr_mb;
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
                                    gamma = lr_mb;
                                else if strcmp(test,'MBTRM_WS')
                                        WS = true;
                                        MS = 2;
                                        gamma = lr_mb;
                                    else if strcmp(test,'TRM_MBGD')
                                            GD = true;
                                            TRMstep = true;
                                            MS = 2;
                                            b_m_mini_general = b_m_mini_MBGD;
                                            gamma = sgd_lr_mb;
                                        else if strcmp(test,'STRM_NC')
                                                MS = 1;
                                                gamma = lr;       
                                        
                                            else if strcmp(test,'GD')
                                                    GD = true;
                                                    gamma = sgd_lr;
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
    end
end



                        


