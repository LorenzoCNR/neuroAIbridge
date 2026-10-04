clear all; clc
data_trials_ok=data_trials
N=numel(data_trials_ok)
% con questa cosa sto aggregando le condizioni (scelta legittima ma va 
% specificato)
%cond=ones(N,1)
% for i=1:N
%     data_trials_ok(i).trialTypeCond=cond(i)
% end

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

config_.corr_obj= "manifold"  % "manifold" | "hat-obs" | "hat-hat"
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

    config_.decoder = struct;
    config_.decoder.method = "ridge";

    switch config_.decoder.method
        case "ridge"
            config_.decoder.ridge_lambdas = logspace(-4,4,10);

        case "knn"
            config_.decoder.knn_k_grid = [3 5 7 9];

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
config_.trial_length= 34% lunghezza dei trial (int Fix)
% trial lenght timespan in ms 
config_.t_start=0
config_.t_end=1000
config_.block_size =10 % dimensione dei blocchi (lag) (int < trial_length)
config_.block_stride = 1 

% i due parametri successivi occorrono (Francesco docet) a determinare il
% numero di trial permettendo una media degli stessi:
% sub_block_size è la finestra temporale su cui si calcola la media locale.
% sub_block_stride controlla lo shift tra sottoblocchi (overlap implicito).
% Caso degenere:
%   sub_block_size = block_size --> una sola osservazione per trial per blocco.
%   per direzione 
config_.sub_block_size = 5% 
config_.sub_block_stride = 0.5% 

% coerenza dei parametri temporali
assert(config_.block_size < config_.trial_length, ...
    'block_size must be smaller than trial_length');
assert(config_.block_stride > 0 && config_.block_stride <= 1, ...
    'block_stride must be in (0,1]');
assert(config_.sub_block_size <= config_.block_size, ...
    'sub_block_size must be <= block_size');
assert(config_.sub_block_stride > 0 && config_.sub_block_stride <= 1, ...
    'sub_block_stride must be in (0,1]');

if config_.sub_block_size == config_.block_size && config_.sub_block_stride ~= 1
    warning(['sub_block_size == block_size: sub_block_stride has no effect ', ...
             '(single average per block).']);
end

% soggetti, direzione e condizione  (task)
% soggetto attivo e passivo dipendono dalla condiz.
config_.dir = [1]  % list of direction(s)8
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
L_L_matrix = compute_LL_matrix(Y_subject,X_subject, config_);


[c_m,k, stats]=LL_plot_detail(L_L_matrix,config_, 0,1000, 0,1000)
