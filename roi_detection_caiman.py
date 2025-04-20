#!/usr/bin/env python

"""
Complete demo pipeline for processing two photon calcium imaging data using the
CaImAn batch algorithm. The processing pipeline included motion correction,
source extraction and deconvolution. The demo shows how to construct the
params, MotionCorrect and cnmf objects and call the relevant functions. You
can also run a large part of the pipeline with a single method (cnmf.fit_file)
See inside for details.

Demo is also available as a jupyter notebook (see demo_pipeline.ipynb)
Dataset couresy of Sue Ann Koay and David Tank (Princeton University)

This demo pertains to two photon data. For a complete analysis pipeline for
one photon microendoscopic data see demo_pipeline_cnmfE.py

copyright GNU General Public License v2.0
authors: @agiovann and @epnev
"""

import cv2
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import tifffile as tiff
import time
import warnings
from utils import load_hdf5_as_dict
from IPython import embed

# import caiman as cm
from caiman import load, load_memmap, stop_server, load_movie_chain, concatenate
from caiman.cluster import setup_cluster
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.source_extraction import cnmf as o_cnmf
from caiman.summary_images import local_correlations_movie_offline


def source_extraction(file_name, save_dir, visual_check=True, parallel=True, corr_map=False):
    # start a cluster
    if parallel:
        c, dview, n_processes = setup_cluster(
            backend='multiprocessing', n_processes=None, single_thread=False
        )
    else:
        dview = None
        n_processes = 1

    # File Name
    f_names = [file_name]
    cmap = 'gray'

    # SETTINGS
    params_dict = {
        'data': {
            'fnames': f_names,
            'fr': 3,
            'decay_time': 5
        },
        'init': {
            'K': 100,  # expected # of neurons per patch
            'gSig': [2, 2],  # expected half size of neurons in px
            # 'method_init': 'corr_pnr',   # correlation-based initialization, patching should be avoided here
            'method_init': 'greedy_roi',
            'min_corr': 0.8,  # min local correlation for seed
            'min_pnr': 10,  # min peak-to-noise ratio for seed
            'ssub': 2,  # spatial subsampling during initialization (use every 2nd pixel → half resolution)
            'tsub': 2,   # temporal subsampling during initialization (average every 2 frames)
            'nb': 2,  # global background order
            'normalize_init': True,  # z score data, do not use with CNMF-E Background Ring Model
        },
        'online': {
            'ring_CNN': False  # CNMF-E Background Ring Model, if False, use global low-rank background modeling
        },
        'patch': {
            'n_processes': None,
            'rf': None,  # half size of each patch (should be ≥ 2× gSig)
            'stride': None  # overlap between patches ( 1- stride/rf), (typically 50% of rf)
            # 'rf': 16,  # half size of each patch (should be ≥ 2× gSig)
            # 'stride': 8  # overlap between patches ( 1- stride/rf), (typically 50% of rf)
        },
        'merging': {
            'merge_thr': 0.8  # merging threshold, max correlation allowed
        },
        'temporal': {
            'p': 1,  # order of the autoregressive system
        },
        'quality': {
            'SNR_lowest': 1.0,  # minimum required trace SNR. Traces with SNR below this will get rejected
            'min_SNR': 2.5,  # peak SNR for accepted components (if above this, accept)
            'rval_lowest': 0.2,  # minimum required space correlation. Components with correlation below this will get rejected
            'rval_thr': 0.8,  # spatial footprint consistency: space correlation threshold (if above this, accept)
            'use_cnn': True,  # use the CNN classifier (prob. of component being a neuron)
            'min_cnn_thr': 0.9,   # Only components with CNN scores ≥ thr are accepted as likely real neurons.
            'cnn_lowest': 0.1  # Components scoring < lowest are considered garbage and won’t be touched even during manual curation or re-evaluation.
        },
        'general': {
            'use_cuda': True
        },
    }

    opts = params.CNMFParams(params_dict=params_dict)

    # 2. Run full CNMF pipeline
    print('\n==== RUN CNMF =====\n')
    cnm = cnmf.CNMF(n_processes=n_processes, params=opts, dview=dview)
    cnm = cnm.fit_file()

    # 3. Evaluate components
    # the components are evaluated in three ways:
    #   a) the shape of each component must be correlated with the data
    #   b) a minimum peak SNR is required over the length of a transient
    #   c) each shape passes a CNN based classifier (this will pick up only neurons
    #           and filter out active processes)
    # A component has to exceed ALL low thresholds as well as ONE high threshold to be accepted.

    print('\n==== EVALUATING COMPONENTS =====\n')
    if cnm.estimates.A.shape[-1] <= 0:
        print('\n==== WARNING: NO COMPONENTS FOUND =====\n')
        exit()

    # load memory mapped file
    Yr, dims, T = load_memmap(cnm.mmap_file)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    mean_image = np.mean(images, axis=0)
    # sd_image = np.std(images, axis=0)

    print("Dims:", dims, "Frames:", T)
    cnm.estimates.evaluate_components(images, cnm.params, dview=dview)

    # 4. Visualize components
    if visual_check:
        if corr_map:
            print('\n==== VISUAL VALIDATION =====\n')
            # Plotting contours
            # Compute Pixel Correlation Matrix (px and its 8 neighbors)
            lc = local_correlations_movie_offline(
                f_names[0],
                remove_baseline=True,
                swap_dim=False,
                window=500,
                stride=8,
                winSize_baseline=200,
                quantil_min_baseline=10,
                dview=dview
            )
            Cn = lc.max(axis=0)

        # Get ROI centers
        component_centers = cnm.estimates.center
        good_idx = cnm.estimates.idx_components

        # Compare Accepted and Rejected Components
        save_contour_plot(cnm, mean_image, f'{save_dir}/caiman_mean_img_contour_plot.jpg', cmap=cmap)
        save_roi_centers_plot(component_centers[good_idx], mean_image, file_dir=f'{save_dir}/caiman_mean_img_rois_center.jpg', marker_size=30, cmap=cmap)

        if corr_map:
            save_contour_plot(cnm, Cn, f'{save_dir}/corr_contour_plot.jpg', cmap=cmap)
            save_roi_centers_plot(component_centers[good_idx], Cn, file_dir=f'{save_dir}/caiman_corr_rois_center.jpg', marker_size=30, cmap=cmap)

        # # play movie with results (original, reconstructed (A·C + b), amplified residual)
        # v = cnm.estimates.play_movie(
        #     images,
        #     q_min=1,
        #     q_max=99,
        #     include_bck=True,
        #     magnification=3,
        #     gain_res=1,
        #     gain_color=1,
        #     thr=0,
        #     use_color=False,
        #     display=False,
        # )
        # v.save(f'{save_dir}/caiman_movie.avi')
        # print('Movie Stored to Disk')
        # write_video_from_array(v, f'{save_dir}/movie.mp4', fps=10)

    print('\n==== SAVING RESULTS =====\n')
    # 5. Save results (optional)
    cnm.save(f'{save_dir}/cnmf_full_pipeline_results.hdf5')
    # Save manual curation
    # cnm.save(f'{save_dir}/cnmf_curated.hdf5')
    # Save the movie overlay
    if corr_map:
        np.save(f'{save_dir}/caiman_local_correlation_map.npy', Cn)

    # 6. Save ROI Traces

    # Export de-noised traces (C)
    # raw_traces = cnm.estimates.A.T @ Yr
    C = cnm.estimates.C  # shape (n_neurons, n_frames)
    pd.DataFrame(C.T).to_csv(f"{save_dir}/caiman_ca_traces.csv", index=False)

    # Export de-convolved spikes (S)
    # S = cnm.estimates.S
    # pd.DataFrame(S.T).to_csv(os.path.join(output_dir, "S_traces_spikes.csv"), index=False)

    # Export component centers (x, y)
    component_centers = cnm.estimates.center
    df_centers = pd.DataFrame(component_centers, columns=["y", "x"])  # CaImAn uses (row, col)
    df_centers.to_csv(f'{save_dir}/caiman_roi_centers.csv', index=False)

    print('\n==== CAIMAN FINISHED =====\n')
    if parallel:
        # Stop the cluster
        stop_server(dview=dview)


def inspect_caiman_results(file_dir, bg_image):
    # from caiman.source_extraction.cnmf.cnmf import load_CNMF
    cnm = cnmf.load_CNMF(file_dir)
    cnm.estimates.view_components(img=bg_image, idx=cnm.estimates.idx_components, cmap='viridis')


def save_roi_centers_plot(centers, bg_image, file_dir, marker_size=20, cmap='gray'):
    fig = plt.figure()
    fig.set_size_inches(15, 10)
    plt.imshow(bg_image, cmap=cmap, vmin=np.min(bg_image), vmax=np.max(bg_image)/1.5)
    plt.scatter(centers[:, 1], centers[:, 0], color='red', marker='o', s=marker_size, alpha=0.5)
    plt.axis('off')  # remove axes and ticks
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # remove padding/margins
    plt.savefig(file_dir, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def save_contour_plot(cnm, bg_image, file_dir, cmap):
    cnm.estimates.plot_contours(img=bg_image, idx=cnm.estimates.idx_components, cmap=cmap)
    fig = plt.gcf()
    fig.set_size_inches(15, 10)
    plt.axis('off')  # remove axes and ticks
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # remove padding/margins
    plt.savefig(file_dir, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def write_video_from_array(v, output_path, fps=10):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    n_frames, h, w = v.shape
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MPEG-4 codec for .mp4
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h), isColor=True)

    for frame in v:
        # Normalize to 0–255, convert to 8-bit
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
        frame = frame.astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # convert to 3 channels
        writer.write(frame)

    writer.release()
    print(f"Saved video to {output_path}")


def motion_correction(file_name, pw_rigid, output_path, display_images=False):
    # First setup some parameters for data and motion correction
    # dataset dependent parameters
    fnames = [file_name]
    fr = 3  # imaging rate in frames per second
    decay_time = 5.0  # length of a typical transient in seconds
    dxy = (2., 2.)  # spatial resolution in x and y in (um per pixel)
    max_shift_um = (12., 12.)  # maximum shift in um
    patch_motion_um = (100., 100.)  # patch size for non-rigid correction in um

    # motion correction parameters
    # pw_rigid = True  # flag to select rigid vs pw_rigid motion correction

    # maximum allowed rigid shift in pixels
    max_shifts = [int(a / b) for a, b in zip(max_shift_um, dxy)]

    # start a new patch for pw-rigid motion correction every x pixels
    strides = tuple([int(a / b) for a, b in zip(patch_motion_um, dxy)])

    # overlap between patches (size of patch in pixels: strides+overlaps)
    overlaps = (24, 24)

    # maximum deviation allowed for patch with respect to rigid shifts
    max_deviation_rigid = 3

    # size of filter, change this one if algorithm does not work
    gSig_filt = (3, 3)

    # mc_dict = {
    #     'fnames': fnames,
    #     'fr': fr,
    #     'decay_time': decay_time,
    #     'dxy': dxy,
    #     'pw_rigid': pw_rigid,
    #     'max_shifts': max_shifts,
    #     'strides': strides,
    #     'overlaps': overlaps,
    #     'max_deviation_rigid': max_deviation_rigid,
    #     # 'gSig_filt': gSig_filt,
    #     'border_nan': 'copy'
    # }

    params_dict = {
        'data': {
            'fnames': fnames,
            'fr': fr,
            'decay_time': decay_time,
            'dxy': dxy
        },
        'motion': {
            'pw_rigid': pw_rigid,
            'max_shifts': max_shifts,
            'strides': strides,
            'overlaps': overlaps,
            'max_deviation_rigid': max_deviation_rigid,
            'border_nan': 'copy'
        },
    }

    opts = params.CNMFParams(params_dict=params_dict)
    # play the movie (optional)
    # playing the movie using opencv. It requires loading the movie in memory.
    # To close the video press q

    if display_images:
        m_orig = load_movie_chain(fnames)
        ds_ratio = 0.2
        moviehandle = m_orig.resize(1, 1, ds_ratio)
        moviehandle.play(q_max=99.5, fr=60, magnification=2)

    # start a cluster for parallel processing
    c, dview, n_processes = setup_cluster(
        backend='multiprocessing', n_processes=None, single_thread=False)

    # MOTION CORRECTION
    # first we create a motion correction object with the specified parameters
    mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
    # note that the file is not loaded in memory

    # Run motion correction using NoRMCorre
    mc.motion_correct(save_movie=True)

    # compare with original movie
    if display_images:
        m_orig = load_movie_chain(fnames)
        m_els = load(mc.mmap_file)
        ds_ratio = 0.2
        moviehandle = concatenate([m_orig.resize(1, 1, ds_ratio) - mc.min_mov * mc.nonneg_movie,
                                      m_els.resize(1, 1, ds_ratio)], axis=2)
        moviehandle.play(fr=60, q_max=99.5, magnification=2)  # press q to exit

    # Store corrected video to disk
    # Load the corrected memory-mapped file
    corrected_movie = load(mc.mmap_file)

    # Normalize and convert to uint16 (if needed)
    corrected_array = corrected_movie - corrected_movie.min()
    corrected_array /= corrected_array.max()
    corrected_array = (corrected_array * 65535).astype(np.uint16)

    # Save as TIFF stack
    # output_path = f'{file_name[:-4]}_motion_corrected.tif'
    tiff.imwrite(
        output_path,
        corrected_array,
        dtype=np.uint16,
        # bigtiff=False,
        imagej=True,
        # metadata=None,
        photometric='minisblack',
        # planarconfig=None,
        # compress=None,
        # tile=None,
        # resolution=None,
        # description=None
    )

    stop_server(dview=dview)


def run_OnACID(file_dir, parallel=True):

    print('\n==== STARTING CAIMAN =====\n')
    if parallel:
        c, dview, n_processes = setup_cluster(backend='multiprocessing', n_processes=None, single_thread=False)
    else:
        dview = None

    fname = [file_dir]

    fr = 3  # frame rate (Hz)
    decay_time = 5  # approximate length of transient event in seconds
    gSig = [3, 3]  # expected half size of neurons
    p = 1  # order of AR indicator dynamics
    min_SNR = 1  # minimum SNR for accepting candidate components
    thresh_CNN_noisy = 0.65  # CNN threshold for candidate components
    gnb = 2  # number of background components
    init_method = 'cnmf'  # initialization method

    # set up CNMF initialization parameters
    init_batch = 400  # number of frames for initialization
    patch_size = 32  # size of patch
    stride = 8  # amount of overlap between patches
    K = 10  # max number of components in each patch
    use_CNN = True

    params_dict = {'fr': fr,
                   'fnames': fname,
                   'decay_time': decay_time,
                   'gSig': gSig,
                   'p': p,
                   'min_SNR': min_SNR,
                   'nb': gnb,
                   'init_batch': init_batch,
                   'init_method': init_method,
                   'rf': patch_size // 2,
                   'stride': stride,
                   'sniper_mode': True,
                   'thresh_CNN_noisy': thresh_CNN_noisy,
                   'K': K}
    opts = params.CNMFParams(params_dict=params_dict)
    # fit with online object
    print('\n==== RUNNING OnACID =====\n')
    cnm = o_cnmf.online_cnmf.OnACID(params=opts, dview=dview)
    cnm.fit_online()
    print('\n==== FINISHED OnACID =====\n')
    # plot contours
    print(f'Number of components: {str(cnm.estimates.A.shape[-1])}')
    Cn = load(fname[0], subindices=slice(0, 500)).local_correlations(swap_dim=False)
    # cnm.estimates.plot_contours(img=Cn)


    # pass through the CNN classifier with a low threshold (keeps clearer neuron shapes and excludes processes)
    if use_CNN:
        # threshold for CNN classifier
        opts.set('quality', {'min_cnn_thr': 0.05})
        cnm.estimates.evaluate_components_CNN(opts)
        cnm.estimates.plot_contours(img=Cn, idx=cnm.estimates.idx_components)

    embed()
    exit()

    # plot results
    cnm.estimates.view_components(img=Cn, idx=cnm.estimates.idx_components)

    if parallel:
        c.close()


def batch_motion_correction(file_dir, dual_pass=False):
    tif_files = [f for f in os.listdir(file_dir) if f.endswith('.tif')]
    k = 0
    if dual_pass:
        pw_rigid = False
    else:
        pw_rigid = True

    for f in tif_files:
        k += 1
        f_dir = f'{file_dir}/{f}'
        motion_correction(
            file_name=f_dir,
            pw_rigid=pw_rigid,
            display_images=False,
            output_path=f'{f_dir[:-4]}_motion_corrected.tif'
        )

        if dual_pass:
            pw_rigid = True
            motion_correction(
                file_name=f'{f_dir[:-4]}_motion_corrected.tif',
                pw_rigid=pw_rigid,
                display_images=False,
                output_path=f'{f_dir[:-4]}_motion_corrected.tif'
            )
        print(f'FINISHED {k}/{len(tif_files)}')


def batch_source_extraction(file_dir, save_dir):
    tif_files = [f for f in os.listdir(file_dir) if f.endswith('.tif')]
    k = 0

    for f in tif_files:
        t0 = time.perf_counter()
        k += 1
        f_dir = f'{file_dir}/{f}'

        # Create Output Dir
        sw_dir = f'{save_dir}/sw_{k:02}'
        os.makedirs(sw_dir, exist_ok=True)
        source_extraction(
            file_name=f_dir,
            save_dir=sw_dir,
            visual_check=True,
            parallel=True,
            corr_map=True
        )
        t1 = time.perf_counter()
        print(f'FINISHED {k}/{len(tif_files)}, this took {(t1 - t0)/60:.3f} minutes')



def main():
    # warnings.filterwarnings("ignore", category=FutureWarning)
    file_dir = 'F:/WorkingData/Tec_Data/Neuropil_RTe_Ca_imaging/tiff_recordings/motion_corrected'
    save_dir = 'F:/WorkingData/Tec_Data/Neuropil_RTe_Ca_imaging/caiman_output'
    # Batch Motion Correction
    # batch_motion_correction(file_dir, dual_pass=False)
    # exit()

    # Batch ROI Detection and Source Extraction
    batch_source_extraction(file_dir, save_dir)

    # motion_correction(
    #     file_name='D:/WorkingData/RoiDetection/test/rec/sw_25.tif',
    #     display_images=False
    # )

    # run_OnACID(file_dir='D:/WorkingData/RoiDetection/test/rec/sw_25_motion_corrected.tif', parallel=False)

    # source_extraction(
    #     file_name='D:/WorkingData/RoiDetection/test/rec/sw_25_motion_corrected.tif',
    #     save_dir='D:/WorkingData/RoiDetection/test/rec/caiman_output',
    #     visual_check=True,
    #     parallel=False,
    #     corr_map=False
    # )


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    t1 = time.perf_counter()
    print(f'FINISHED {(t1 - t0)/60:.2f} minutes')

    import timeit
    # execution_time = timeit.timeit("main()", globals=globals(), number=1)
    # print(f"Execution time: {execution_time:.4f} seconds")
