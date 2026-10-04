function h = LL_plot(corr_mat, config, start_, end_)
    n_lags = size(corr_mat,1);
    full_length = config.t_end - config.t_start;

    % if config.cond == 2
    %     corr_mat = corr_mat_';
    %     y_lab = ['Lag (ms) ', upper(config.obs_subject)];
    %     x_lab = ['Lag (ms) ', upper(config.active_subject)];
    % else
    %     corr_mat = corr_mat_;
    %     y_lab = ['Lag (ms) ', upper(config.active_subject)];
    %     x_lab = ['Lag (ms) ', upper(config.obs_subject)];
    % end


   if nargin < 3
        t_start = config.t_start;
        t_end   = config.t_end;
    else
        t_start = start_;
        t_end   = end_;
    end
    t_start = max(config.t_start, min(config.t_end, t_start));
    t_end   = max(config.t_start, min(config.t_end, t_end));

    % map time to matrix index
    t_start_ = floor(1 + (t_start - config.t_start) / full_length * (n_lags - 1));
    t_end_   = ceil(1 + (t_end   - config.t_start) / full_length * (n_lags - 1))
    t_start_ = max(1, min(n_lags, t_start_));
    t_end_   = max(1, min(n_lags, t_end_));
   
    y_lab = ['Lag (ms) ', upper(config.y_subject)];
    x_lab = ['Lag (ms) ', upper(config.x_subject)];

    idx = t_start_:t_end_;
    
    Lag_full = linspace(config.t_start, config.t_end, n_lags);
    Lag_X = Lag_full(idx);
    Lag_Y = Lag_full(idx);
    % 
    % Lag_X = linspace(t_start_, t_end_, n_lags);
    % Lag_Y = linspace(t_start_, t_end_, n_lags);

    h.fig = figure;
    h.ax  = axes(h.fig);
 
    h.im = imagesc(h.ax, Lag_X, Lag_Y, corr_mat(idx, idx));
    
    set(h.ax,'YDir','normal');
    
    colormap(redbluecmap);
    colorbar;
    caxis([-1 1]);
    
    xlabel(x_lab);
    ylabel(y_lab);
    title(sprintf('\nCondition %d: \nDirection %d', config.cond, config.dir));
    

   %xlim(h.ax, [Lag_X(1) Lag_X(end)]);
   %ylim(h.ax, [Lag_Y(1) Lag_Y(end)]);
    
    axis(h.ax,'xy');  
    
    hold on
    % % Linea diagonale dagli stessi limiti ma ho  problemi di padding
    % plot(h.ax, [Lag_X(1) Lag_X(end)], ...
    %        [xlims ylims], ...
    %        '--','Color',[.5 .5 .5],'LineWidth',1.2);

    xlims = xlim;
    ylims = ylim;
    plot(xlims, ylims, '--', 'Color',[.5 .5 .5], 'LineWidth',1.2)
    hold off
end