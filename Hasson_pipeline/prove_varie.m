start_=0
end_=500
config =config_  
corr_mat=L_L_matrix
n_lags = size(corr_mat,1);
full_length = config.t_end - config.t_start;

% se non passi start_/end_ usa dominio completo
if nargin < 3
    t_start = config.t_start;
    t_end   = config.t_end;
else
    t_start = start_;
    t_end   = end_;
end

% clamp finestra nel dominio reale
t_start = max(config.t_start, min(config.t_end, t_start));
t_end   = max(config.t_start, min(config.t_end, t_end));

% mappatura tempo (ms) -> indice matrice
t_start_ = 1 + (t_start - config.t_start) / full_length * (n_lags - 1);
t_end_   = 1 + (t_end   - config.t_start) / full_length * (n_lags - 1);

t_start_ = floor(t_start_);
t_end_   = ceil(t_end_);

% clamp indici in [1, n_lags]
t_start_ = max(1, min(n_lags, t_start_));
t_end_   = max(1, min(n_lags, t_end_));

idx = t_start_:t_end_;


Lag_full = linspace(config.t_start, config.t_end, n_lags);
Lag_X = Lag_full(idx);
Lag_Y = Lag_full(idx);
h.fig = figure;
h.ax  = axes(h.fig);

h.im = imagesc(h.ax, Lag_X, Lag_Y, corr_mat(idx, idx));
    % 
    % Lag_X = linspace(t_start_, t_end_, n_lags);
    % Lag_Y = linspace(t_start_, t_end_, n_lags);;
    set(h.ax, 'YDir', 'normal');
    colormap(redbluecmap);
    colorbar;
    caxis([-1 1]);

    xlabel(['Lag (ms) ', upper(config.x_subject)]);
    ylabel(['Lag (ms) ', upper(config.y_subject)]);

    title(sprintf('\nCondition %d: \nDirection %d',config.cond, config.dir));

    hold on
    axis(h.ax, 'tight');
    xlim([Lag_X(1) Lag_X(end)]);
    ylim([Lag_Y(1) Lag_Y(end)]);

    hold on
    plot([Lag_X(1) Lag_X(end)], ...
     [Lag_Y(1) Lag_Y(end)], ...
      '--', 'Color',[.5 .5 .5], 'LineWidth',1.2);
    hold off

    hold off
