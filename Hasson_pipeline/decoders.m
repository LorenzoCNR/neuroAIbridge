function [Y_hat, RSS_mat, RSS_mean, MSE_mean, corr_mean, ...
          RSS_shuffle_mat, RSS_shuffle_mean, corr_shuffle_mean, ...
          count_R2, R2_mean] = decoders(lag_X, lag_Y, config)

assert(isfield(config,'decoder'),'Missing config.decoder');
assert(isfield(config.decoder,'method'),'Missing config.decoder.method');

method = config.decoder.method;

n_lags = numel(lag_X);
[~, n_ch] = size(lag_Y);

Y_hat = cell(n_lags,n_ch);

% storage metriche reali
R2       = nan(n_lags,n_ch);
RSS_mat  = nan(n_lags,n_ch);
MSE_mat  = nan(n_lags,n_ch);
corr_mat = nan(n_lags,n_ch);

% storage metriche shuffle
RSS_shuffle_mat  = nan(n_lags,n_ch);
corr_shuffle_mat = nan(n_lags,n_ch);
k_best_mat = nan(n_lags,n_ch);

switch lower(method)
   % ----
   %%--- RIDGE DECODER
   % -------
    case 'ridge'
        lambdas = config.decoder.ridge_lambdas;
        n_lambda = numel(lambdas);
      
        count_R2 = 0;
        R2_mean = NaN;

        for b = 1:n_lags
            X_ = lag_X{b};
            d  = size(X_,2);
            I  = eye(d);

            % centro X una volta
            Xc = X_ - mean(X_);

            for e = 1:n_ch
                y_ = lag_Y{b,e};
                yc = y_ - mean(y_);
                 % -------- shuffle target ----------
                y_shuffle = y_(randperm(length(y_)));
                yc_shuffle = y_shuffle - mean(y_shuffle);

                % GCV  selection
                gcv_scores = zeros(n_lambda,1);
                for i = 1:n_lambda
                    lambda = lambdas(i);
                    beta   = (Xc' * Xc + lambda * I) \ (Xc' * yc);
                    y_hat_ = Xc * beta;

                    H_diag = sum((Xc / (Xc' * Xc + lambda * I)) .* Xc, 2);
                    gcv_scores(i) = mean(((yc - y_hat_) ./ (1 - H_diag)).^2);
                end

                [~, best_i] = min(gcv_scores);
                lambda_best = lambdas(best_i);
                % Fit finale
                beta = (Xc' * Xc + lambda_best * I) \ (Xc' * yc);
                intercept = mean(y_) - mean(X_) * beta;
                Y_hat{b,e} = X_ * beta + intercept;
                yhat = Y_hat{b,e};

                % -fit finale shuffle -
                beta_shuffle = (Xc'*Xc + lambda_best*I) \ (Xc'*yc_shuffle);
                intercept_shuffle = mean(y_shuffle) - mean(X_)*beta_shuffle;
                yhat_shuffle = X_*beta_shuffle + intercept_shuffle;
                
                % error Metrics
                resid = y_ - yhat;
                
                RSS = sum(resid.^2);

                RSS_mat(b,e) = RSS;
                MSE_mat(b,e)=mean(resid.^2);
                r = corr(y_, yhat,'Rows','complete');
                corr_mat(b,e) = r;
                TSS = sum((y_ - mean(y_)).^2);
                if TSS > 0
                    R2(b,e) = 1 - RSS/TSS;
                end
                  % -------- metriche shuffle ----------
                resid_shuffle = y_shuffle - yhat_shuffle;
    
                RSS_shuffle = sum(resid_shuffle.^2);
                RSS_shuffle_mat(b,e) = RSS_shuffle;
    
                r_shuffle = corr(y_shuffle,yhat_shuffle,'Rows','complete');
                corr_shuffle_mat(b,e) = r_shuffle;



            end
        end
     

  case 'knn'

        % assert(isfield(config.decoder, 'knn_k'), ...
        %     'Missing config.decoder.knn_k');
        % 
        assert(isfield(config.decoder,'knn_k_grid'), ...
         'Missing config.decoder.knn_k_grid');
        %k = config.decoder.knn_k;
        k_grid = config.decoder.knn_k_grid;
        n_k = numel(k_grid);
         k_max = max(k_grid);
        for b = 1:n_lags
            X_ = lag_X{b};
    
            %  standardizzazione (knn sensibile e alla scala delle feature)
  
            Xs = zscore(X_);
                 % ======== CALCOLO VICINI UNA SOLA VOLTA ========

            idx_full = knnsearch(Xs, Xs, 'K', k_max + 1);
            idx_full = idx_full(:,2:end);   % tolgo self
        
    
            % precompute neighbors (escludendo self)
            % idx = knnsearch(Xs, Xs, 'K', k+1);
            % idx = idx(:, 2:end);  % tolgo self
    
            for e = 1:n_ch
                % targetr da ricostruire)
                y_ = lag_Y{b,e};
                y_shuffle=y_(randperm(length(y_)))
                mse_k = zeros(n_k,1);

               for kk = 1:n_k
        
                    k = k_grid(kk);
        
                    idx = knnsearch(Xs,Xs,'K',k+1);
                    idx = idx(:,2:end);
        
                    yhat = mean(y_(idx),2);
        
                    mse_k(kk) = mean((y_ - yhat).^2);
        
                end
        
                [~,best_i] = min(mse_k);
                k_best = k_grid(best_i);

               % final fit
                idx = idx_full(:,1:k_best);
                yhat = mean(y_(idx),2);
                Y_hat{b,e} = yhat;


                % media sui vicini y (corrispettivi nelle X)
                % punti vicini nella manifold, hanno attività neurale
                % simil

                % error Metrics
                resid = y_ - yhat;  
                RSS = sum(resid.^2);
                RSS_mat(b,e) = RSS;
                TSS = sum((y_ - mean(y_)).^2);
                MSE_mat(b,e) = mean(resid.^2);
                r = corr(y_, yhat,'Rows','complete');
                corr_mat(b,e) = r;
                if TSS > 0
                    R2(b,e) = 1 - RSS/TSS;
                end
               
             % predizione shuffle
                yhat_shuffle = mean(y_shuffle(idx),2);
    
                resid_shuffle = y_shuffle - yhat_shuffle;
    
                RSS_shuffle_mat(b,e) = sum(resid_shuffle.^2);
                corr_shuffle_mat(b,e) = corr(y_shuffle,yhat_shuffle,'Rows','complete');
                k_best_mat(b,e) = k_best;


            end
        end


otherwise
        error('Unknown decoder method: %s', method);
end
%%% GLobal Stats
RSS_mean = mean(RSS_mat(:),'omitnan');
MSE_mean  = mean(MSE_mat(:),'omitnan');
corr_mean = mean(corr_mat(:),'omitnan');
RSS_shuffle_mean = mean(RSS_shuffle_mat(:),'omitnan');
corr_shuffle_mean = mean(corr_shuffle_mat(:),'omitnan');
count_R2 = sum(~isnan(R2(:)));
R2_mean  = mean(R2(:),'omitnan');

end