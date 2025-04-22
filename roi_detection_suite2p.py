
import suite2p
import numpy as np
import os
import time
import shutil
import matplotlib.pyplot as plt
from tifffile import imread
from utils import norm_min_max as nmm
import tifftools
from IPython import embed


def suite2p_registering(rec_path, rec_name, non_rigid=0, f_batch_size=300):
    # How to load and read ops.npy files:
    #  np.load(p, allow_pickle=True).item()

    t0 = time.time()
    # Set directories

    reg_suite2_path = f'{rec_path}/suite2p/plane0/reg_tif/'
    print('')
    print('-------- INFO --------')

    # Settings:
    if non_rigid:
        print('Non-Rigid Registration selected!')
    else:
        non_rigid = False
        print('Rigid Registration selected!')
    print('')

    # Store metadata to HDD
    # metadata_df.to_csv(f'{rec_path}/metadata.csv', index=False)

    ops = suite2p.default_ops()  # populates ops with the default options
    ops['tau'] = 1.25  # not important for registration
    ops['fs'] = 2  # not important for registration

    ops['nimg_init'] = 500  # (int, default: 200) how many frames to use to compute reference image for registration
    ops['batch_size'] = f_batch_size  # (int, default: 200) how many frames to register simultaneously in each batch.
    # Depends on memory constraints - it will be faster to run if the batch is larger, but it will require more RAM.

    ops['maxregshift'] = 0.2 # (float, default: 0.1) the maximum shift as a fraction of the frame size.
    # If the frame is Ly pixels x Lx pixels, then the maximum pixel shift in pixels will be max(Ly,Lx) * ops[‘maxregshift’].

    ops['smooth_sigma'] = 1.15 #  (float, default: 1.15) standard deviation in pixels of the gaussian used to smooth
    # the phase correlation between the reference image and the frame which is being registered. A value of >4 is recommended
    # for one-photon recordings (with a 512x512 pixel FOV).

    ops['smooth_sigma_time'] = 0  # (float, default: 0) standard deviation in time frames of the gaussian used to smooth
    # the data before phase correlation is computed. Might need this to be set to 1 or 2 for low SNR data.

    ops['two_step_registration'] = False  #  (bool, default: False) whether or not to run registration twice (for low SNR data).
    # keep_movie_raw must be True for this to work.

    ops['keep_movie_raw'] = False  # (bool, default: False) whether or not to keep the binary file of the non-registered frames.
    # You can view the registered and non-registered binaries together in the GUI in the “View registered binaries” view if you set this to True.

    ops['reg_tif'] = True  # store reg movie as tiff file

    ops['pad_fft'] = False  # (bool, default: False) Specifies whether to pad image or not during FFT portion of registration.

    # NON RIGID SETTINGS
    ops['nonrigid'] = non_rigid  # (bool, default: True) whether or not to perform non-rigid registration,
    # which splits the field of view into blocks and computes registration offset in each block separately.

    ops['block_size'] = [64, 64]  # (two ints, default: [128,128]) size of blocks for non-rigid reg, in pixels.
    # HIGHLY recommend keeping this a power of 2 and/or 3 (e.g. 128, 256, 384, etc) for efficient fft

    ops['snr_thresh'] = 1.5  # (float, default: 1.2) how big the phase correlation peak has to be relative to the noise
    # in the phase correlation map for the block shift to be accepted. In low SNR recordings like one-photon,
    # I’d recommend a larger value like 1.5, so that block shifts are only accepted if there is significant SNR in the phase correlation.

    ops['maxregshiftNR'] = 5.0  # (float, default: 5.0) maximum shift in pixels of a block relative to the rigid shift

    ops['roidetect'] = False  # (bool, default: True) whether or not to run ROI detect and extraction

    db = {
        'data_path': [rec_path],
        'save_path0': rec_path,
        'tiff_list': [rec_name],
        'subfolders': [],
        'fast_disk': rec_path,
        'look_one_level_down': False,
    }

    # Store suite2p settings
    # pd.DataFrame(ops.items(), columns=['Parameter', 'Value']).to_csv(
    #     f'{rec_path}/{rec_name}_reg_settings_ops.csv', index=False)
    # pd.DataFrame(db.items(), columns=['Parameter', 'Value']).to_csv(
    #     f'{rec_path}/{rec_name}_reg_settings_db.csv', index=False)

    # Run suite2p pipeline in terminal with the above settings
    output_ops = suite2p.run_s2p(ops=ops, db=db)
    print('---------- REGISTRATION FINISHED ----------')
    print('---------- COMBINING TIFF FILES ----------')

    # Load registered tiff files
    f_list = sorted(os.listdir(reg_suite2_path))
    # print('FOUND REGISTERED SINGLE TIFF FILES:')
    # print(f_list)
    # Load first tiff file
    im_combined = tifftools.read_tiff(f'{reg_suite2_path}{f_list[0]}')

    # Combine tiff files to one file
    for k, v in enumerate(f_list):
        if k == 0:
            continue
        else:
            im_dummy = tifftools.read_tiff(f'{reg_suite2_path}{v}')
            im_combined['ifds'].extend(im_dummy['ifds'])
    # Store combined tiff file
    tifftools.write_tiff(im_combined, f'{rec_path}/Registered_{rec_name}')
    t1 = time.time()

    # Delete all temporary files
    shutil.rmtree(f'{rec_path}/suite2p/')

    print('----------------------------------------')
    print('Stored Registered Tiff File to HDD')
    print(f'This took approx. {np.round((t1 - t0) / 60, 2)} min.')


def run_suite2p_pipeline(tiff_path, output_folder):
    """
    Parameters explained:
    threshold_scaling: 
    (float, default: 1.0) this controls the threshold at which to detect ROIs (how much the ROIs 
    have to stand out from the noise to be detected). if you set this higher, then fewer ROIs will be detected, and if 
    you set it lower, more ROIs will be detected.
    
    denoise: 
    (bool, default: False) Wether or not binned movie should be denoised before cell detection in sparse_mode.
    If True, make sure to set ops['sparse_mode'] is also set to True.
    """

    # ========== STEP 1: Set up Suite2p ==========
    # Make sure output dir exists
    os.makedirs(output_folder, exist_ok=True)
    ops = suite2p.default_ops()
    ops.update({
        # 'data_path': [os.path.dirname(tiff_path)],  # needed even if unused
        'data_path': [tiff_path],  # scans for tiffs
        # 'file_list': [tiff_path],
        'save_path0': output_folder,
        'nchannels': 1,  # or 2 if it's dual-channel
        'functional_chan': 1,  # usually 1 if GCaMP is first channel
        'nplanes': 1,
        'look_one_level_down': False,
        'fs': 1,
        'tau': 3,
        'threshold_scaling': 0.9,  #  this controls the threshold at which to detect ROIs (how much the ROIs have to stand out from the noise to be detected).
        # 'nbinned': 5000,  # maximum number of binned frames to use for ROI detection (default 5000).
        'batch_size': 500,
        'allow_overlap': True,
        'high_pass': 100,  # running mean subtraction across time with window of size ‘high_pass’.
        'smooth_masks': True,  # whether to smooth masks in final pass of cell detection. This is useful especially if you are in a high noise regime.
        'do_registration': 0,
        'denoise': True,
        'max_iterations': 20,  # how many iterations over which to extract cells
        'diameter': 3,             # Adjust based on measured soma size
        'connected': 1,            # Enforce spatial connectivity
        'max_overlap': 0.6,        # Allow moderate ROI overlap, 1: keep all rois
        'spatial_scale': 0,        # 0: Let Suite2p determine optimal scale
        'sparse_mode': False,        # Enable for sparse activity datasets
        'neuropil_extract': True,  # Whether or not to extract signal from neuropil. If False, Fneu is set to zero.
        'spatial_hp_cp': 25,
        'spikedetect': False,  # Whether or not to run spike_deconvolution
        # 'cellpose_run': True,
        # 'cellpose': True,  # Enable CellPose-based detection, combine functional and cellpose detection

        # Only use cellpose for roi detection (anatomical_only > 1)
        'anatomical_only': 0,
        # 1: Will find masks on max projection image divided by mean image.
        # 2: Will find masks on mean image
        # 3: Will find masks on enhanced mean image
        # 4: Will find masks on maximum projection image

    })

    # Settings for only anatomical cellpose detection
    if ops['anatomical_only']:
    # if ops['anatomical_only']:
        # print('==== USING CELLPOSE ====')
        ops.update({
            'diameter': 4,  # 0: estimate diameter
            'cellprob_threshold': 0,  # Decrease this threshold if cellpose is not returning as many ROIs as you’d expect.
            'flow_threshold': 4.0,  # Increase this threshold if cellpose is not returning as many ROIs as you’d expect.
            'spatial_hp_cp': 200,  # Window for spatial high-pass filtering of image to be used for cellpose.
        })

    print('')
    print("==== Running Suite2p... ====")
    print('')

    # RUN SUITE2P
    suite2p.run_s2p(ops=ops)

    print('')
    print('==== FINISHED ====')
    print('')


def check_detection(output_folder, cmap='gray'):
    import numpy.ma as ma
    plane_path = os.path.join(output_folder, 'suite2p', 'plane0')
    f = np.load(os.path.join(plane_path, 'F.npy'))
    f_neu = np.load(os.path.join(plane_path, 'Fneu.npy'))
    is_cell = np.load(os.path.join(plane_path, 'iscell.npy'))[:, 0].astype(bool)
    stat = np.load(os.path.join(plane_path, 'stat.npy'), allow_pickle=True)
    ops_out = np.load(os.path.join(plane_path, 'ops.npy'), allow_pickle=True).item()
    mean_img = ops_out['meanImg']

    # Select good cells
    f_cells = f[is_cell]
    f_neu_cells = f_neu[is_cell]
    stat_cells = [stat[i] for i, is_cell in enumerate(is_cell) if is_cell]
    neuropil_ratio = 0.7
    f_corrected = f_cells - (neuropil_ratio * f_neu_cells)

    im = np.zeros((ops_out['Ly'], ops_out['Lx']))
    # im = nmm(mean_img.copy())
    ncells = len(stat_cells)

    roi_centers = list()
    for n in range(0, ncells):
        x_center = int(stat_cells[n]['med'][0])
        y_center = int(stat_cells[n]['med'][1])
        roi_centers.append((x_center, y_center))

        ypix = stat_cells[n]['ypix'][~stat_cells[n]['overlap']]
        xpix = stat_cells[n]['xpix'][~stat_cells[n]['overlap']]
        im[ypix, xpix] = n + 1

    roi_centers = np.array(roi_centers)
    threshold = 0.1  # set your desired threshold
    # Mask the overlay image: only show values > threshold
    masked_overlay = ma.masked_where(im <= threshold, im)

    fig, axs = plt.subplots()
    fig.set_size_inches(5, 5)
    axs.imshow(nmm(mean_img), cmap=cmap, vmin=0, vmax=0.5)
    axs.imshow(masked_overlay, cmap='jet', alpha=0.4)
    plt.axis('off')  # remove axes and ticks
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # remove padding/margins
    plt.savefig(f'{output_folder}/suite2p/sp2_rois_shape_mean_image.jpg', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    fig, axs = plt.subplots()
    fig.set_size_inches(5, 5)
    axs.imshow(nmm(mean_img), cmap=cmap, vmin=0, vmax=0.5)
    axs.scatter(roi_centers[:, 1], roi_centers[:, 0], color='red', marker='o', s=20, alpha=0.5)
    plt.axis('off')  # remove axes and ticks
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # remove padding/margins
    plt.savefig(f'{output_folder}/suite2p/sp2_rois_center_mean_image.jpg', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # k = 0
    # fig, axs = plt.subplots(3, 1)
    # axs[0].plot(nmm(f_cells[k, :]), 'k')
    # axs[1].plot(nmm(f_neu_cells[k, :]), 'k')
    # axs[2].plot(nmm(f_corrected[k, :]), 'k')
    # plt.show()


def do_registration(rec_dir):
    # Get all Files in directory
    file_list = os.listdir(rec_dir)
    tiff_files = [s for s in file_list if
                  (s.endswith('.tif') or s.endswith('.tiff') or s.endswith('.TIF') or s.endswith('.TIFF'))]
    print('')
    print('FOUND THE FOLLOWING FILES:')
    for i in tiff_files:
        print(i)
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('')

    # Movement Artifacts Registration
    batch_size = 300
    method = 0  # 0: Rigid, 1: Non-Rigid
    suite2p_registering(rec_path=rec_dir, rec_name=tiff_files[0], non_rigid=method, f_batch_size=batch_size)


def run_suite2p_detection():
    rec_dir = 'D:/WorkingData/RoiDetection/test/rec'

    # Registration
    # do_registration(rec_dir)

    # CELL DETECTION
    file_list = os.listdir(rec_dir)
    tiff_files = [s for s in file_list if
                  (s.endswith('.tif') and s.startswith('Registered'))]

    if len(tiff_files) <= 0:
        print('COULD NOT FIND A REGISTERED TIF RECORDING')
        print('WILL USE ORIGINAL RECORDING')
        tiff_files = [s for s in file_list if
                      (s.endswith('.tif') or s.endswith('.tiff') or s.endswith('.TIF') or s.endswith('.TIFF'))]
    if len(tiff_files) <= 0:
        print('ERROR: NO TIF FILE FOUND')
        exit()

    print(f'FOUND RECORDING: {tiff_files[0]}')
    print('')

    # RUN ROI DETECTION
    run_suite2p_pipeline(
        tiff_path=f'{rec_dir}/{tiff_files[0]}',
        output_folder=rec_dir,
    )


def validate_results(rec_dir):
    # rec_dir = 'D:/WorkingData/RoiDetection/test/rec'
    check_detection(output_folder=rec_dir, cmap='gray')


def batch_suite2p(save_dir):
    import time
    sw_list = os.listdir(save_dir)
    k = 0
    for sw in sw_list:
        t0 = time.perf_counter()
        k += 1
        sw_dir = f'{save_dir}/{sw}'
        run_suite2p_pipeline(
            tiff_path=sw_dir,
            output_folder=sw_dir,
        )

        validate_results(sw_dir)
        t1 = time.perf_counter()
        print(f'FINISHED {k}/{len(sw)}, this took {(t1 - t0)/60:.3f} minutes')
        t1 = time.perf_counter()
        print(f'This took: {(t1-t0)/60:.2f} minutes')



def main():
    # Batch Detection
    # file_dir = 'F:/WorkingData/Tec_Data/Neuropil_RTe_Ca_imaging/tiff_recordings/motion_corrected'
    base_dir = 'F:/WorkingData/Tec_Data/Neuropil_RTe_Ca_imaging/cell_detection'
    batch_suite2p(base_dir)

    # run_suite2p_detection()
    # validate_results(rec_dir='D:/WorkingData/RoiDetection/test/rec')


if __name__ == '__main__':
    import timeit
    execution_time = timeit.timeit("main()", globals=globals(), number=1)
    print(f"Execution time: {execution_time:.4f} seconds")


