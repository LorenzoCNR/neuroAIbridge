function summary = analyze_decoder_results(results, top_k)


n = length(results);

% oprealloco vettori
corr_x = zeros(n,1);
corr_y = zeros(n,1);

corr_shuffle_x = zeros(n,1);
corr_shuffle_y = zeros(n,1);

RSS_x = zeros(n,1);
RSS_y = zeros(n,1);

RSS_shuffle_x = zeros(n,1);
RSS_shuffle_y = zeros(n,1);

% estraggo risutoati dalla struttura
for i = 1:n
    
    corr_x(i) = results(i).corr_x;
    corr_y(i) = results(i).corr_y;
    
    corr_shuffle_x(i) = results(i).corr_shuffle_x;
    corr_shuffle_y(i) = results(i).corr_shuffle_y;
    
    RSS_x(i) = results(i).RSS_x;
    RSS_y(i) = results(i).RSS_y;
    
    RSS_shuffle_x(i) = results(i).RSS_shuffle_x;
    RSS_shuffle_y(i) = results(i).RSS_shuffle_y;
    
end

% score basato sulla correlazione rispetto allo shuffle
score_x = corr_x - corr_shuffle_x;
score_y = corr_y - corr_shuffle_y;

score_total = (score_x + score_y)/2;

% ordino configurazioni
[~, idx] = sort(score_total,'descend');

best_idx = idx(1:top_k);

summary.mean_corr_x = mean(corr_x(best_idx));
summary.mean_corr_y = mean(corr_y(best_idx));

summary.mean_corr_shuffle_x = mean(corr_shuffle_x(best_idx));
summary.mean_corr_shuffle_y = mean(corr_shuffle_y(best_idx));

summary.mean_RSS_x = mean(RSS_x(best_idx));
summary.mean_RSS_y = mean(RSS_y(best_idx));

summary.mean_RSS_shuffle_x = mean(RSS_shuffle_x(best_idx));
summary.mean_RSS_shuffle_y = mean(RSS_shuffle_y(best_idx));

summary.best_idx = best_idx;

end