function [X_sel,Y_sel] = filter_data(X_struct, Y_struct, config)
% describe

% INPUTS:
%   Trials       : Struct array (1 x N_trials).
%
% OUTPUT:
%   Filtered data for given subjects according to assigned condition and
%   directions

% ASSUMPTIONS (devono essere verificate)
%  si assume che i trial selezionati siano gli stessi e nello stesso ordine
%  oltre che accoppiati allo stesso modo (etichette). Se manca un trial
%  sono cazzi quindi 
% Prima dell'analisi va quindi controllato
% - numero di trial selezionati;
% - `trialTypeDir` elemento per elemento;
% - `trialTypeCond` elemento per elemento;
% - eventuale identificativo comune del trial.
% I contatori `pippo_X` e `pippo_Y` non hanno effetto matematico.

    dir_=config.dir;
    cond_= config.cond;
            % filter data according to direction and condition
    
    keep_X = false(1, numel(X_struct));
    keep_Y = false(1, numel(Y_struct));
    
    pippo_X = 0;
    pippo_Y = 0;
    
    for i = 1:numel(X_struct)
        keep_X(i) = ...
                ismember(X_struct(i).trialTypeDir, dir_) && ...
                ismember(X_struct(i).trialTypeCond, cond_);
          %ismember(X_struct(i).trialTypeDir, dir_) && ...
          %ismember(X_struct(i).trialTypeCond, cond_);
          pippo_X = pippo_X + 1;
        
    end
    
    for i = 1:numel(Y_struct)
        keep_Y(i) = ...
            any(ismember(Y_struct(i).trialTypeDir, dir_)) && ...
            any(ismember(Y_struct(i).trialTypeCond, cond_));
        pippo_Y = pippo_Y + 1;
      
    end
    
    X_sel = X_struct(keep_X);
    Y_sel = Y_struct(keep_Y);

end