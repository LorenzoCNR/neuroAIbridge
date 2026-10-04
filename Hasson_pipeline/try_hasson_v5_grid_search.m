data_trials_ok=data_trials_ok
N=numel(data_trials_ok)
cond=ones(N,1)
for i=1:N
    data_trials_ok(i).trialTypeCond=cond(i)
end

for i=1:N

    S_Struct(i).Spikes      = data_trials_ok(i).S_GS_data;
    S_Struct(i).Manifold    = data_trials_ok(i).S_dpca;
    S_Struct(i).trialTypeDir= data_trials_ok(i).trialTypeDir;
    S_Struct(i).trialTypeCond= data_trials_ok(i).trialTypeCond;
end

for i=1:numel(data_trials_ok)
    K_Struct(i).Spikes      = data_trials_ok(i).K_GS_data;
    K_Struct(i).Manifold    = data_trials_ok(i).K_dpca;
    K_Struct(i).trialTypeDir= data_trials_ok(i).trialTypeDir;
    K_Struct(i).trialTypeCond= data_trials_ok(i).trialTypeCond;
end



% S_Struct=S_Struct'
% K_Struct=K_Struct'

% In generale, la struttura dei dati per soggetto deve essere con almeno i
% campi:
% - Condizione (ad ora trialTypeCond)
% - Direzione   (ad ora trialTypeDir) 
% - Spikes (in cui ogni cella è di dimensione Neuroni*#bin_trial)
%   N.B.! i trials per questo tipo di analisi devono avere tutti medesima
%   lunghezza
% - MAnifold: strutturato come i trials
%
%A_struct=K_Struct'
%B_struct=S_Struct'

% per far girare la funzione Hasson (compute_LL_matrix): 
% intanto genero la struttura di configurazione cui passo tutte le info necessarie

config_=struct;

config_.corr_obj= "hat-obs"  % "manifold" | "hat-obs" | "hat-hat"
    % a seconda degli oggetti che voglio correlare, scelgo
    % manifold: correli le manifold e basta
    % Negli altri due casi, si passa per un decoder per avere dei valori
    % stimati dell'attività neurale, a partire dalle manifold stesse: 
    % - "hat-obs" correlo le stime di un soggetto con i dati osservati
    %    dell'altro (e viceversa). 
    % -  Nel caso "hat-hat" correlo le stime fatte con decoder dei soggetti

% use_neural indica se i dati neurali vengono utilizzati nella pipeline
% (non necessariamente per la correlazione). 
config_.use_neural=1

% Ci deve essere coerena tra l'uso di dati neurali ed gli oggetti da
% correlare; se sceglo manifold i dati neurali non occorrono per la
% correlazione (possono cmq occorrere per altro). 
    switch config_.corr_obj
        case "manifold"
            % use_neural can be true or false (future-proof)
        case {"hat-obs", "hat-hat"}
            assert(config_.use_neural == true, ...
           'use_neural must be true for corr_obj = %s', config_.corr_obj);
        otherwise
            error("Unknown corr_obj");
    end


 % parametri decoder (nei casi hat-obs e hat-hat, devo scegliere un
 % decoder per fare le correlazioni tra dati osservati e stimati)
if ismember(config_.corr_obj, ["hat-obs","hat-hat"])

    % struttura coerente col decoder (per ora abbiamo solo ridge reg e knn)
    config_.decoder = struct;
    config_.decoder.method = "ridge";   % "ridge" | "knn" | ...
   

    switch config_.decoder.method
        case "ridge"
            config_.decoder.ridge_lambdas = logspace(-4,4,10);
        case "knn"
            config_.decoder.knn_k_grid= [3, 5, 7, 9];

                %config_.decoder.knn_k = 
           
        otherwise
                error("Unknown decoder method");
        end
end


% I tre parametri seguenti determinano il numero di blocchi (lag).
% block_stride è una quota di block_size e definisce lo SHIFT del blocco:
%   stride_abs = floor(block_stride * block_size)
% L’overlap implicito è:
%   block_size - stride_abs
% Il numero di blocchi è:
%   n_blocchi = floor((trial_length - block_size) / stride_abs) + 1
% trial length (in bins)
config_.trial_length= 100 % lunghezza dei trial (int Fix)
% trial lenght timespan in ms 

grid_block=[5,8,10, 20,25,50]
grid_block_stride=[0.25, 0.5]
grid_sub_block= [2 4, 5, 10, 20, 25]
grid_sub_block_stride=[0.5,1]

config_.t_start=0
config_.t_end=1000
% config_.block_size =  10 % dimensione dei blocchi (lag) (int < trial_length)
% config_.block_stride = 0.2  
% config_.sub_block_size = 2   % 

% i due parametri successivi occorrono (Francesco docet) a determinare il
% numero di trial permettendo una media degli stessi:
% sub_block_size è la finestra temporale su cui si calcola la media locale.
% sub_block_stride controlla lo shift tra sottoblocchi (overlap implicito).
% Caso degenere:
%   sub_block_size = block_size --> una sola osservazione per trial per blocco.
%   per direzione 

% soggetti, direzione e condizione  (task)
% soggetto attivo e passivo dipendono dalla condiz.
        config_.dir = [1]  % list of direction(s)
        config_.cond = [1] % condizione
        
        
%%% Just notice that:
% In compute_LL_matrix(Y_subject, X_subject, ...),
% Y_subject maps to the ROWS (Y-axis) of the lag–lag matrix,
% X_subject maps to the COLUMNS (X-axis) of the lag–lag matrix.
Y_subject=S_Struct
X_subject=K_Struct
% define which subject 
config_.y_subject='s' 
config_.x_subject='k'

 % filtering data
[X_sel, Y_sel]=filter_data(Y_subject, X_subject, config_)
% % c_r2_x = [];
% % m_r2_x = [];
% % c_r2_y = [];
% % m_r2_y = [];
results=struct()
idx=1
for i =1:numel(grid_block)
    %disp(grid_block(i))
    fprintf('Block size: %d\n', grid_block(i));
    config_.block_size =  grid_block(i)
    for iS= 1:numel(grid_block_stride)
        config_.block_stride=grid_block_stride(iS);
        for S = 1:numel(grid_sub_block)
            if grid_sub_block(S)<=(1/2)*grid_block(i)
                config_.sub_block_size = grid_sub_block(S);
                for Ss=1:numel(grid_sub_block_stride)
                    config_.sub_block_stride =   grid_sub_block_stride(Ss);
                     % coerenza dei parametri temporali
                    assert(config_.block_size < config_.trial_length, ...
                        'block_size must be smaller than trial_length');
                    assert(config_.block_stride > 0 && config_.block_stride <= 1, ...
                        'block_stride must be in (0,1]');
                    assert(config_.sub_block_size <= config_.block_size, ...
                        'sub_block_size must be <= block_size');
                    assert(config_.sub_block_stride > 0 && config_.sub_block_stride <= 1, ...
                        'sub_block_stride must be in (0,1]');
                      % filtering data
                     [lag_manif_X, lag_manif_Y, lag_n_X, lag_n_Y] = ...
                     convert_to_lag_struct(X_sel, Y_sel, config_);


%                 end
%             end
%         end
%     end
% end

       
                     [Y_hat_c,~,RSS_mean_x,~,corr_mean_x,~,RSS_shuffle_mean_x,corr_shuffle_mean_x,count_r2_x,r2_mean_x] = decoders(lag_manif_X, lag_n_X, config_);
                     % c_r2_x=[c_r2_x, count_r2_x]
                     % m_r2_x=[m_r2_x, r2_mean_x]
                     [Y_hat_y,~,RSS_mean_y,~,corr_mean_y,~,RSS_shuffle_mean_y, corr_shuffle_mean_y,count_r2_y,r2_mean_y] =  decoders(lag_manif_Y, lag_n_Y, config_);
                     % c_r2_y=[c_r2_y, count_r2_y]
                     % m_r2_y=[m_r2_y, r2_mean_y]
                     results(idx).block_size        = config_.block_size;
                     results(idx).block_stride      = config_.block_stride;
                     results(idx).sub_block_size    = config_.sub_block_size;
                     results(idx).sub_block_stride  = config_.sub_block_stride;
                     results(idx).lagged_y= lag_n_Y;
                     results(idx).lagged_x= lag_n_X;
                     results(idx).lag_manif_y= lag_manif_Y;
                     results(idx).lag_manif_x= lag_manif_X;
                     results(idx).R2_x = r2_mean_x;
                     results(idx).R2_y = r2_mean_y;
                     results(idx).RSS_x = RSS_mean_x;
                     results(idx).RSS_y = RSS_mean_y;
                     results(idx).corr_x = corr_mean_x;
                     results(idx).corr_y = corr_mean_y;
                     results(idx).count_x = count_r2_x;
                     results(idx).count_y = count_r2_y;
                     results(idx).RSS_shuffle_x = RSS_shuffle_mean_x;
                     results(idx).RSS_shuffle_y = RSS_shuffle_mean_y;
                     results(idx).corr_shuffle_x = corr_shuffle_mean_x;
                     results(idx).corr_shuffle_y = corr_shuffle_mean_y;
                     idx=idx+1
                end
            end
        end
    end
end
            
results_ridge=results
top_k = 5;

summary_ridge = analyze_decoder_results(results_ridge, top_k);
summary_knn   = analyze_decoder_results(results_knn, top_k);

fprintf('\n===== RIDGE =====\n');

disp(summary_ridge)

fprintf('\n===== KNN =====\n');

disp(summary_knn)        
 
        
    
        % 
        % % build lag structures
        % if config.use_neural
        %     [lag_manif_X, lag_manif_Y, lag_n_X, lag_n_Y] = ...
        %         convert_to_lag_struct(X_sel, Y_sel, config);
        % else
        %     [lag_manif_X, lag_manif_Y] = ...
        %         convert_to_lag_struct(X_sel, Y_sel, config);
        % end
        % 
        % %%  select objects to correlatie representation 
        % switch config.corr_obj
        % 
        %     case "manifold"
        %         % filter data according to direction and condition
        %         LL_matrix=f_corr(lag_manif_X, lag_manif_Y)
        % 
        %     case "hat-obs"
        %         % Predizioni
        %         Y_hat_X = decoders(lag_manif_X, lag_n_X, config);
        %         Y_hat_Y = decoders(lag_manif_Y, lag_n_Y, config);
        % 
        %         % Direzione A --> B
        %         LL_XY = f_corr(Y_hat_X, lag_n_Y);
        % 
        %         % Direzione B --> A
        %         LL_YX = f_corr(Y_hat_Y, lag_n_X);
        % 
        %         % Media simmetrica (come fanno Hasson e Zada nel paper)
        %         LL_matrix = 0.5 * (LL_XY + LL_YX);
        % 
        % 
        %     case "hat-hat" 
        %        Y_hat_X = decoders(lag_manif_X, lag_n_X, config);
        %        Y_hat_Y = decoders(lag_manif_Y, lag_n_Y, config); 
        %        LL_matrix =f_corr(Y_hat_X, Y_hat_Y);
        % 
        %     otherwise
        %         error("Unknown corr_obj");
        % end
        % 
        % 
        % 
        % 
        % 
