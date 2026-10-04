function [c_m,h, stats] = LL_plot_detail(corr_mat, config, t_start_y,t_end_y, t_start_x, t_end_x)
    
    % ------------------------------------------------------------
    % INPUT:
    % corr_mat : matrice lag-lag completa [n_lags x n_lags]
    % quadrata e uniformemente campionata nel tempo
    % config   : struttura con t_start, t_end, soggetti ecc.
    % eventuali limiti temporali per ritagliare la matrice
    %
    % OUTPUT:
    % c_m   : sottomatrice ritagliata
    % h     : handle grafico (figura, assi, image)
    % stats : (opzionale) statistiche direzionali
    % ------------------------------------------------------------



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
        t_start_x = config.t_start;
        t_end_x   = config.t_end;
        t_start_y = config.t_start;
        t_end_y   = config.t_end;
    else
        t_start_x= t_start_x;
        t_end_x   = t_end_x;
        t_start_y= t_start_y;
        t_end_y   = t_end_y;
    end
    t_start_x = max(config.t_start, min(config.t_end, t_start_x));
    t_end_x   = max(config.t_start, min(config.t_end, t_end_x));
    t_start_y = max(config.t_start, min(config.t_end, t_start_y));
    t_end_y   = max(config.t_start, min(config.t_end, t_end_y));


    % map time to matrix index (n.b.! assume a linear relationship between
    % mapping and time)
    t_start_x_ = floor(1 + (t_start_x - config.t_start) / full_length * (n_lags - 1));
    t_end_x_   = ceil(1 + (t_end_x  - config.t_start) / full_length * (n_lags - 1))
    t_start_x_= max(1, min(n_lags, t_start_x_));
    t_end_x_  = max(1, min(n_lags, t_end_x_));
    t_start_y_ = floor(1 + (t_start_y - config.t_start) / full_length * (n_lags - 1));
    t_end_y_   = ceil(1 + (t_end_y  - config.t_start) / full_length * (n_lags - 1))
    t_start_y_= max(1, min(n_lags, t_start_y_));
    t_end_y_  = max(1, min(n_lags, t_end_y_))
   
   
    %y_lab = ['Lag (ms) ', upper(config.y_subject)];
    %x_lab = ['Lag (ms) ', upper(config.x_subject)];
    x_lab = ['Delayed population time lag (ms)'];
    y_lab = ['Leading population time lag (ms)'];
  

    idx_x= t_start_x_:t_end_x_;
    idx_y= t_start_y_:t_end_y_;
 
    Lag_full = linspace(config.t_start, config.t_end, n_lags);
    Lag_X = Lag_full(idx_x);
    Lag_Y = Lag_full(idx_y);
    % 
    % Lag_X = linspace(t_start_, t_end_, n_lags);
    % Lag_Y = linspace(t_start_, t_end_, n_lags);

    h.fig = figure;
    h.ax  = axes(h.fig);
 
    h.im = imagesc(h.ax, Lag_X, Lag_Y, corr_mat(idx_y, idx_x));
    c_m=corr_mat(idx_y, idx_x)

    % ------------------------------------------------------------
    % Calcolo statistiche solo se richiesto
    % ------------------------------------------------------------

    if nargout > 2
        stats = LL_directional_stats(c_m, 'all');
    end
    n_lags_x = size(c_m, 1);
    n_lags_y = size(c_m, 2);

    set(h.ax,'YDir','normal');
    
    colormap(redbluecmap);
    colorbar;
    caxis([-1 1]);
    
    xlabel(x_lab);
    ylabel(y_lab);
    %title(sprintf('\nCondition %d: \nDirection %d', config.cond, config.dir));
     
    title({'Lag-lag correlation map', ...
       'Leading-to-delayed temporal asymmetry'})

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