# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 13:08:20 2025

@author: loren
"""

import numpy as np
import matplotlib
#matplotlib.use("Agg") 
import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = r'C:\Users\loren\Downloads\ffmpeg-8.0-essentials_build\ffmpeg-8.0-essentials_build\bin\ffmpeg.exe'
from matplotlib.animation import FFMpegWriter
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D # noqa: F401
from pathlib import Path
import sys


# Add decoder path if needed
external_path = Path(r"J:\AI_PhD_Neuro_CNR\Empirics\GIT_stuff\AI_for_all\Neuro_Bridge\src\neurobridge\eval")
if str(external_path) not in sys.path:
    sys.path.append(str(external_path))
    
from knn_decoder import knn_decode_pos, r2_pos, mederr_pos

'''
function for both static and video plot

plt.subplot(2, 1, 1)
fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)

più elastico ...gs
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1], wspace=0.35, hspace=0.35)
x_series = fig.add_subplot(gs[0, 0])  #
ax_raster = fig.add_subplot(gs[1, 0])  # 
ax_embed3d = fig.add_subplot(gs[:, 1], projection='3d')

inpputs:
    dictionary of subject dictionaries
    subject
        |_X_used: data used to train model
        |_X_hat: manifold generated
        |_y: labels which guided the process
outputs

'''


def make_video_achilles(data_dict,
                        DT=0.025,
                        T_START_S=5.0,
                        T_END_S=15.0,
                        K_NEIGHBORS=5,
                        title_scatter='Manifold',
                        title_track='Decoded kNN',
                        animate=True,
                        save_path=None,
                        show=True):
    
    emb = np.asarray(data_dict["X_hat"])
    lab = np.asarray(data_dict["y"])
    spikes_full = np.asarray(data_dict["X"])
    
    if spikes_full.ndim != 2:
        raise ValueError("spikes_full must be 2D.")
    if spikes_full.shape[0] < spikes_full.shape[1]:
            spikes_full = spikes_full.T
    
    T_total = min(emb.shape[0], lab.shape[0], spikes_full.shape[0])
    emb = emb[:T_total]
    lab = lab[:T_total]
    spikes_full = spikes_full[:T_total]
    t_axis = np.arange(T_total) * DT


    pos_1d = lab.reshape(-1, 1) if lab.ndim == 1 else lab[:, :1]
    
    
    pos_hat_full = knn_decode_pos(
    emb_train=emb,
    y_pos_train=pos_1d,
    emb_eval=emb,
    k=K_NEIGHBORS,
    metric='cosine')
    
    i0 = int(T_START_S / DT)
    i1 = int(T_END_S / DT)
    i0 = max(0, min(i0, T_total - 2))
    i1 = max(i0 + 2, min(i1, T_total))
    
    
    Z = emb[i0:i1]
    lab_w = lab[i0:i1]
    pos_true = (pos_1d[i0:i1, 0] * 100.0)
    pos_hat = (pos_hat_full[i0:i1, 0] * 100.0)
    spikes = spikes_full[i0:i1]
    t_axis = t_axis[i0:i1] - t_axis[i0]
    
    
    # ---------------- EMBEDDING 3D (NO PCA; pad se d<3) ----------------
    d = Z.shape[1]
    Z_vis3 = np.column_stack([Z, np.zeros((Z.shape[0], 3 - d))]) if d < 3 else Z[:, :3]
    
    # 3D per TUTTO l’embedding (sfondo “pieno”)
    d_all = emb.shape[1]
    Z_all3 = np.column_stack([emb, np.zeros((emb.shape[0], 3 - d_all))]) if d_all < 3 else emb[:, :3]
    
    # ---------------- RASTER z-score ----------------
    S = spikes.astype(float).copy()
    S -= S.mean(axis=0, keepdims=True)
    S /= (S.std(axis=0, keepdims=True) + 1e-9)
    extent = [0.0, float(t_axis[-1]), 0, S.shape[1]]
    
    # ---------------- FIGURA E LAYOUT ----------------
    fig = plt.figure(figsize=(14, 6), dpi=150)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1], wspace=0.35, hspace=0.35)
    
    # (1) TRACK (sopra a sinistra): True vs Decoded con colori distinti
    ax_series = fig.add_subplot(gs[0, 0])
    ax_series.plot(t_axis, pos_true, ls="--", lw=1.6, color="gray", alpha=0.95)
    #ax_series.plot(t_axis, pos_hat,  lw=1.8, color="#6235E0", label="Decoded (CEBRA+kNN)", zorder=3)
   
    ax_series.set_title("Linear-track position")
    ax_series.set_xlabel("Time [s]")
    ax_series.set_ylabel("Position [cm]")
    line_true, = ax_series.plot(t_axis, pos_true, ls="--", lw=1.6, color="gray", alpha=0.95)
    line_reveal, = ax_series.plot([], [], lw=1.8, color="#6235E0", zorder=3)
    ax_series.legend(
        handles=[line_true, line_reveal],
        labels=["True", title_track],
        loc="upper center", bbox_to_anchor=(0.5, 1.22),
        ncol=2, frameon=False, borderaxespad=0.
    )
    ts_cursor = ax_series.axvline(float(t_axis[0]), color='k', lw=1.0, alpha=0.9)
    
    
    # (2) RASTER (sotto a sinistra)
    ax_raster = fig.add_subplot(gs[1, 0])
    S = spikes.astype(float).copy()
    S -= S.mean(axis=0, keepdims=True)
    S /= (S.std(axis=0, keepdims=True) + 1e-9)
    #extent = [0.0, float(t_axis[-1]), 0, S.shape[1]]
    tlim = ax_series.get_xlim()
    extent = [tlim[0], tlim[1], 0, S.shape[1]]
    ax_raster.imshow(S.T, aspect='auto', origin='lower',
                     cmap='gray_r', vmin=-2, vmax=2, extent=extent)
    ax_raster.set_xlim(tlim)
    #ax_raster.set_xlim(ax_series.get_xlim())
    ax_raster.set_title("Neural raster (z-scored)")
    ax_raster.set_xlabel("Time [s]")
    ax_raster.set_ylabel("Neurons")
    ras_cursor = ax_raster.axvline(0.0, color='k', lw=1.0, alpha=0.9)
    
# (3) MANIFOLD 3D (destra): sfondo pieno + overlay finestra, colori cool_r/summer_r
    ax_embed3d = fig.add_subplot(gs[:, 1], projection='3d')
    ax_embed3d.set_title(title_scatter)
    ax_embed3d.set_xlabel("z1"); ax_embed3d.set_ylabel("z2"); ax_embed3d.set_zlabel("z3")
    
    # Maschere left/right su TUTTO (col 2 = left, col 1 = right)
    idx_left_all = (lab[:, 2] == 1) if lab.ndim >= 2 and lab.shape[1] >= 3 else np.zeros(len(lab), dtype=bool)
    idx_right_all = (lab[:, 1] == 1) if lab.ndim >= 2 and lab.shape[1] >= 3 else np.zeros(len(lab), dtype=bool)
    
    # Sfondo pieno (alpha basso)
    plotted_bg = False
    if np.any(idx_left_all):
        ax_embed3d.scatter(
            Z_all3[idx_left_all, 0], Z_all3[idx_left_all, 1], Z_all3[idx_left_all, 2],
            c=lab[idx_left_all, 0], cmap="cool_r", s=0.6, alpha=0.25, label=""
        )
        plotted_bg = True
    if np.any(idx_right_all):
        ax_embed3d.scatter(
            Z_all3[idx_right_all, 0], Z_all3[idx_right_all, 1], Z_all3[idx_right_all, 2],
            c=lab[idx_right_all, 0], cmap="summer_r", s=0.6, alpha=0.25, label=""
        )
        plotted_bg = True
    if not plotted_bg:
        ax_embed3d.scatter(
            Z_all3[:, 0], Z_all3[:, 1], Z_all3[:, 2],
            c=lab[:, 0] if lab.ndim >= 2 else None, cmap="viridis",
            s=0.6, alpha=0.25, label="All"
        )
    
    # Overlay finestra (più marcato)
    idx_left_w = (lab_w[:, 2] == 1) if lab_w.ndim >= 2 and lab_w.shape[1] >= 3 else np.zeros(len(lab_w), dtype=bool)
    idx_right_w = (lab_w[:, 1] == 1) if lab_w.ndim >= 2 and lab_w.shape[1] >= 3 else np.zeros(len(lab_w), dtype=bool)
    
    if np.any(idx_left_w):
        ax_embed3d.scatter(
            Z_vis3[idx_left_w, 0], Z_vis3[idx_left_w, 1], Z_vis3[idx_left_w, 2],
            c=lab_w[idx_left_w, 0], cmap="cool_r", s=2.0, alpha=0.95, label="", zorder=3
        )
    if np.any(idx_right_w):
        ax_embed3d.scatter(
            Z_vis3[idx_right_w, 0], Z_vis3[idx_right_w, 1], Z_vis3[idx_right_w, 2],
            c=lab_w[idx_right_w, 0], cmap="summer_r", s=2.0, alpha=0.95, label="", zorder=3
        )
    if (not np.any(idx_left_w)) and (not np.any(idx_right_w)):
        ax_embed3d.scatter(
            Z_vis3[:, 0], Z_vis3[:, 1], Z_vis3[:, 2],
            c=lab_w[:, 0] if lab_w.ndim >= 2 else None, cmap="viridis",
            s=2.0, alpha=0.95, label="Window", zorder=3
        )

        # animated trajectory + moving dot
    traj3, = ax_embed3d.plot([], [], [], lw=1.6, alpha=0.9, color='k')
    dot3,  = ax_embed3d.plot([], [], [], 'o', ms=6, color='k')
    ax_embed3d.legend(frameon=False, loc="upper left")
    ax_embed3d.axis("off")
    
    plt.tight_layout()
    
    if animate:
        T_win = len(t_axis)
        def init():
            line_reveal.set_data([], [])
            dot3.set_data([], [])
            dot3.set_3d_properties([])
            ts_cursor.set_xdata([t_axis[0], t_axis[0]])
            ras_cursor.set_xdata([t_axis[0], t_axis[0]])
            return line_reveal, ts_cursor, ras_cursor, dot3
            
            
        def update(f):
            line_reveal.set_data(t_axis[:f+1], pos_hat[:f+1])
            ts_cursor.set_xdata([t_axis[f], t_axis[f]])
            ras_cursor.set_xdata([t_axis[f], t_axis[f]])
            dot3.set_data([Z_vis3[f, 0]], [Z_vis3[f, 1]])
            dot3.set_3d_properties([Z_vis3[f, 2]])
            return line_reveal, ts_cursor, ras_cursor, dot3
            
        print(f"Number of animation frames: {T_win}")
  
        ani = FuncAnimation(fig, update, frames=T_win, init_func=init,
                                interval=100, blit=False, cache_frame_data=False)
            
            
        if save_path is not None:
                from matplotlib.animation import FFMpegWriter

                writer = FFMpegWriter(fps=30, metadata=dict(artist='NeuroBridge'), bitrate=1800)

                ani.save(str(save_path), writer=writer, dpi=150)
                
            
            
        if show:
            plt.show()
        else:
            plt.close(fig)
            
    
    return fig
     
'''
#FIGURE SEPARATE

fig, ax = plt.subplots()
# oppure
fig = plt.figure()
ax = fig.add_subplot(111)

# 1 track pos
plt.plot(t_axis_c, pos_true_c, '--', color='gray', label="True")
plt.plot(t_axis_c, pos_hat_c, color="#6235E0", label="Decoded")
plt.legend(); plt.xlabel("Time [s]"); plt.ylabel("Position [cm]")

#2. Solo raster
plt.imshow(S.T, aspect='auto', origin='lower', cmap='gray_r', vmin=-2, vmax=2,
           extent=[0.0, t_axis_c[-1], 0, S.shape[1]])
plt.xlabel("Time [s]"); plt.ylabel("Neurons")

#3. Solo manifold
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Z_all3[:,0], Z_all3[:,1], Z_all3[:,2], c=lab[:,0], cmap="viridis", alpha=

    '''