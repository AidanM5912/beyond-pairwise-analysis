from utils import *
from multiprocessing import Pool
import h5py
import os
import psutil
from s3 import s3wrangler as wr
import sys
import numpy as np
import pickle
import glob
import zipfile
import persim

# --- global batching state ---
batch_edge_weights = []
batch_frame_indices = []
flush_interval = 100  #  tuneable
peak_memory_gb = -1.0

batch_triangle_units = []  # for (i, j, k) tracking



## create the structure containing all the edges and triplets for every time t
def create_simplicial_framework_from_data(data, null_model_flag, folder_javaplex, scaffold_outdir, do_zscore):
    global ts_simplicial
    
    # Create the ets and the triplets_ts
    ts_simplicial = simplicial_complex_mvts(
        data, null_model_flag, folder_javaplex, scaffold_outdir, do_zscore)
    # return(ts_simplicial)


#this function is depricated
# This function allows to save on .hd5 file the list of violating triangles when projected at the level of edges.
# Moreover, it saves on the standard output several global quantities (line 32):
# Time; Hyper complexity indic.; Hyper complexity FC; Hyper complexity CT;
# Hyper complexity FD; Hyper coherence; Average edge violation

def handle_output(result):
    """
    result is (summary, edge_weights, triangle_units, mem_gb) from launch_code_one_t.
    This runs in the main process, so it’s safe to write the HD5 files here.
    """
    summary, edge_weights, triangle_units, mem_gb = result

    global peak_memory_gb
    peak_memory_gb = max(peak_memory_gb, mem_gb)

    print(" ".join(str(el) for el in summary))

    if flag_edgeweight:
        # acumulate into the main-process batches
        batch_edge_weights.append(edge_weights)
        batch_triangle_units.append(triangle_units)
        batch_frame_indices.append(summary[0])

        # flush to disk once we hit flush_interval

        if len(batch_edge_weights) >= flush_interval:
            #print(f"[flush] Writing {len(batch_frame_indices)} frames…")
            # Write edge weights
            with h5py.File(f"{flag_edgeweight_fn}.hd5", "a") as f_out:
                for idx, frame in enumerate(batch_frame_indices):
                    rows = []
                    for (i, j), (w, c) in batch_edge_weights[idx].items():
                        rows.append([int(i), int(j), float(w), float(c)])
                    arr = np.asarray(rows, dtype=np.float32)
                    f_out.create_dataset(str(frame), data=arr, compression="gzip")

            # Write triangle units
            with h5py.File(f"{flag_edgeweight_fn}_triangles.hd5", "a") as f_tri:
                for idx, frame in enumerate(batch_frame_indices):
                    tri_arr = np.array(batch_triangle_units[idx], dtype=np.int32)
                    if tri_arr.ndim != 2 or tri_arr.shape[1] != 3:
                        raise RuntimeError(f"Triangle array at frame {frame} is malformed: shape {tri_arr.shape}")
                    f_tri.create_dataset(str(frame), data=tri_arr, compression="gzip")

            # clear only these three lists—keeps RAM bounded by flush_interval
            batch_edge_weights.clear()
            batch_frame_indices.clear()
            batch_triangle_units.clear()


##Launch the bulk of the code for a single time point
def launch_code_one_t(t):
    global flag_edgeweight_fn
    global pd_outdir
    global prefix

    print("starting the simplical code for t = {0}".format(t))
    # Computing the simplicial filtration for the time t
    list_simplices_positive, list_violation_fully_coherence, hyper_coherence, list_filtration_scaffold = ts_simplicial.create_simplicial_complex(
        t)
    
    triangle_units = [list(simplices) for (simplices, _, _) in list_violation_fully_coherence]
    if any(len(tri) != 3 for tri in triangle_units):
        raise RuntimeError(f"Bad triangle shape in frame {t}: {triangle_units[:5]}")

    # Computing the persistence diagram using cechmate
    #print("Computing the persistence diagram")

    dgms1 = compute_persistence_diagram_cechmate(list_simplices_positive)
    # Maximum value that will be used to replace the inf term (important for the WS distance)
    #print("Computing the maximum filtration weight")
    max_filtration_weight = ts_simplicial.find_max_weight(t)
    # Replace the inf value of the persistence diagram with maximum weight
    #print("Cleaning the persistence diagram")
    dgms1_clean = clean_persistence_diagram_cechmate(
        dgms1, max_filtration_weight)
    pd_filename = os.path.join(pd_outdir, f"{prefix}PD1_{t}.pck")
    with open(pd_filename, "wb") as pf:
        pickle.dump(dgms1_clean, pf)
    print(f"[saved] {pd_filename}")


    # If flag is activated, compute the scaffold and save the list of generators on file
    # The function below uses jython (and the corresponding code: persistent_homology_calculation.py)
    
    if ts_simplicial.javaplex_path != False:
        print("computing scaffold!")
        # build absolute path to the Jython helper, based on this script's location
        script_dir = os.path.dirname(os.path.realpath(__file__))
        phy_path   = os.path.join(script_dir, 'persistent_homology_calculation.py')
        compute_scaffold(list_filtration_scaffold, dimension=1, directory=ts_simplicial.scaffold_outdir,
                         tag_name_output='_{0}'.format(t),
                         javaplex_path=ts_simplicial.javaplex_path, save_generators=True, verbose=False,
                         python_persistenthomologypath=phy_path)

    # Computing the hyper-complexity indicator as the Wasserstein distance with the empty space
    #print("Computing the hyper-complexity indicator")
    hyper_complexity = persim.sliced_wasserstein(dgms1_clean, np.array([]))

    # Since the signs of the persistence diagram are flipped,
    # then Fully Coherent contributes identify points with birth and death <=0
    
    dgms1_complexity_FC = dgms1_clean[(
        dgms1_clean[:, 0] < 0) & (dgms1_clean[:, 1] <= 0)]
    # Coherent Transition contributes identify points with birth < 0 and death > 0
    dgms1_complexity_CT = dgms1_clean[(
        dgms1_clean[:, 0] < 0) & (dgms1_clean[:, 1] > 0)]
    # Fully Decoherence contributes identify points with birth > 0 and death > 0
    dgms1_complexity_FD = dgms1_clean[(
        dgms1_clean[:, 0] > 0) & (dgms1_clean[:, 1] > 0)]

    # Computing the Wasserstein distances
    print(" computing wasserstein distances")
    complexity_FC = persim.sliced_wasserstein(
        dgms1_complexity_FC, np.array([]))
    complexity_CT = persim.sliced_wasserstein(
        dgms1_complexity_CT, np.array([]))
    complexity_FD = persim.sliced_wasserstein(
        dgms1_complexity_FD, np.array([]))

    flag_violations_list = np.array(
        list_violation_fully_coherence, dtype="object")[:, 2]
    # Average edge violation
    print("average edge violation")
    avg_edge_violation = np.mean(flag_violations_list)

    n_ROI = ts_simplicial.num_ROI
    # From the magnitude of the list of violating triangles $\Delta_v$,
    # we compute the downward projection at the level of edges
    print("computing the edge weights")
    edge_weights = compute_edgeweight(list_violation_fully_coherence, n_ROI)

    # Report the results in a vector and print everything 
    # (except the downward projections) on standard output

    ###modiying for batching!!
    #results = [t, hyper_complexity, complexity_FC, complexity_CT, complexity_FD, hyper_coherence, avg_edge_violation, edge_weights]
    #return(results)

    # at end of launch_code_one_t, before you return:
    if any(len(tri) != 3 for tri in triangle_units):
        raise RuntimeError(f"Bad triangle shape in frame {t}: {triangle_units[:5]}")

    # Update this worker’s peak memory tracker
    process = psutil.Process(os.getpid())
    mem_gb = process.memory_info().rss / 1e9


    # Pack summary stats exactly as before
    summary = [t, hyper_complexity, complexity_FC, complexity_CT, complexity_FD, hyper_coherence, avg_edge_violation]

    # Return a 3-tuple instead of just the summary:
    # 1) summary list for printing
    # 2) edge_weights dict for HDF5
    # 3) triangle_units list for HDF5
    return (summary, edge_weights, triangle_units, mem_gb)



    

def process_data(s3_input_path, local_input_file):
    """
    Download a file from S3 (using your s3wrangler utility) 
    and return the local path to it (or None on error).
    """
    
    print(f"Downloading {s3_input_path} → {local_input_file} …")
    try:
        wr.download(s3_input_path, local_input_file)
    except Exception as e:
        print(f" Error downloading {s3_input_path}: {e}", file=sys.stderr)
        return None

    if not os.path.exists(local_input_file):
        print(f" Expected file {local_input_file} not found after download", file=sys.stderr)
        return None

    return local_input_file

def push_to_s3(local_path, s3_path):
    try:
        wr.upload(local_path, s3_path)
        print(f" Uploaded {local_path} → {s3_path}")
    except Exception as e:
        print(f" Failed to upload {local_path} → {s3_path}: {e}", file=sys.stderr)


############# MAIN CODE #############
if len(sys.argv) <= 1:
    print(
        "******************************************************************************\n"
        "**                                                                          **\n"
        "**              Computation of the higher-order indicators                  **\n"
        "**               starting from a multivariate time series                   **\n"
        "**                                                                          **\n"
        "**                                                                          **\n"
        "**  <filename_multivariate_series> file containing the multiv. time series  **\n"
        "**                         Format currently accepted:                       **\n"
        "**        .txt:  where columns represents the independent time series       **\n"
        "**        .mat:  where rows are ROI, and columns are the time instants      **\n"
        "**                                                                          **\n"
        "**                                                                          **\n"
        "**                     ----   Optional Variables  ----                      **\n"
        "**                                                                          **\n"
        "**    <-t t0 T> restricts the output of the higher-order indicators         **\n"
        "**                  only for the time interval [t0,T]                       **\n"
        "**                                                                          **\n"
        "**   <-p #core> represents the number of cores used for the computation of  **\n"
        "**                     the higher-order indicators                          **\n"
        "**                                                                          **\n"
        "**     <-n > computes the higher-order indicators for the null model        **\n"
        "**           constructed by independently reshuffling each signal           **\n"
        "**                                                                          **\n"
        "**   <-s <filename>> saves on filename.hdf5 the weighted network obtained   **\n"
        "**    when projecting the magnitude of the list of violations on a graph    **\n"
        "**                                                                          **\n"
        "**    <-j -path_javaplex -outdir> launch the jython code for computing the  **\n"
        "**    homological scaffold and save it in the in the folder 'outdir', it    **\n"
        "**    relies on javaplex and requires a lot of RAM for this computation     **\n"
        "**                                                                          **\n"
        "**      OUTPUT: by default the algorithm returns the following info:        **\n"
        "** Time; Hyper complexity indic.; Hyper complexity FC; Hyper complexity CT; **\n"
        "**        Hyper complexity FD; Hyper coherence; Average edge violation      **\n"
        "**                                                                          **\n"
        "******************************************************************************\n"
        "Usage: %s <filename_multivariate_series>   [-t t0 T] [-p #core] [-n] [-s <filename>] [-j <path_javaplex> <name_outdir>]\n\n" % sys.argv[0]);
    exit(1)

if __name__ == "__main__":
    # change working directory to script location
    script_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(script_dir)

    # Parse command‑line args
    (path_file, t_init, t_end, t_total, ncores,
     null_model_flag, flag_edgeweight, flag_edgeweight_fn,
     folder_javaplex, scaffold_outdir, do_zscore,
     output_s3_path) = parse_input(sys.argv)

    # Determine prefix for null model runs and prepare pd output dir
    prefix    = "nullmodel_" if null_model_flag else ""
    pd_outdir = f"{prefix}pd_diagrams"
    os.makedirs(pd_outdir, exist_ok=True)

    # Download data from S3 if needed
    if path_file.startswith("s3://"):
        local = os.path.basename(path_file)
        got   = process_data(path_file, local)
        if got is None:
            sys.exit(1)
        path_file = local

    # adjust output paths for null model
    if null_model_flag:
        if flag_edgeweight_fn:
            flag_edgeweight_fn = f"{prefix}{flag_edgeweight_fn}"
        if scaffold_outdir:
            scaffold_outdir = os.path.join(scaffold_outdir, "nullmodel_outputs")
            os.makedirs(scaffold_outdir, exist_ok=True)

    # empty existing .hd5 files if applicable
    if flag_edgeweight_fn:
        with h5py.File(f"{flag_edgeweight_fn}.hd5", "w"):
            pass

    if scaffold_outdir:
        os.makedirs(scaffold_outdir, exist_ok=True)

    # load input data
    data_TS = load_data(path_file)
    print("data loaded!", data_TS.shape, data_TS.dtype)
    sys.stdout.flush()

    # use full time range if not specified
    if t_init == 0 and t_end == 0:
        t_end   = data_TS.shape[1]
        t_total = list(range(t_init, t_end))

    # initialize multiprocessing pool
    pool = Pool(
        processes=ncores,
        initializer=create_simplicial_framework_from_data,
        initargs=(data_TS, null_model_flag, folder_javaplex, scaffold_outdir, do_zscore)
    )

    # dispatch jobs
    for t in t_total:
        pool.apply_async(launch_code_one_t, args=(t,), callback=handle_output)
    pool.close()
    pool.join()

    # flush any remaining frames to .hd5
    if flag_edgeweight and batch_edge_weights:
        with h5py.File(f"{flag_edgeweight_fn}.hd5", "a") as f_out:
            for idx, frame in enumerate(batch_frame_indices):
                arr = np.array(list(batch_edge_weights[idx].items()))
                arr = arr.flatten().reshape(-1, 4).astype(np.float32)
                f_out.create_dataset(str(frame), data=arr, compression="gzip")
        with h5py.File(f"{flag_edgeweight_fn}_triangles.hd5", "a") as f_tri:
            for idx, frame in enumerate(batch_frame_indices):
                tri_arr = np.array(batch_triangle_units[idx], dtype=np.int32)
                if tri_arr.shape[1] != 3:
                    raise RuntimeError(f"Triangle array at frame {frame} is malformed: shape {tri_arr.shape}")
                f_tri.create_dataset(str(frame), data=tri_arr, compression="gzip")
        batch_edge_weights.clear()
        batch_frame_indices.clear()
        batch_triangle_units.clear()

    print(f"[summary] peak memory usage: {peak_memory_gb:.2f} GB")

    # upload outputs to s3 if requested
    if output_s3_path:
        print(f"Uploading triangles and edgeweights to S3 at {output_s3_path} …")
        # Upload .hd5 edge weight matrix
        if flag_edgeweight and flag_edgeweight_fn:
            local_hd5 = f"{flag_edgeweight_fn}.hd5"
            tri_hd5 = f"{flag_edgeweight_fn}_triangles.hd5"
            if os.path.exists(local_hd5):
                push_to_s3(local_hd5, os.path.join(output_s3_path, os.path.basename(local_hd5)))
                print(f" {local_hd5} uploaded to S3 …")
            if os.path.exists(tri_hd5):
                push_to_s3(tri_hd5, os.path.join(output_s3_path, os.path.basename(tri_hd5)))
                print(f" {tri_hd5} uploaded to S3 …")

        # bundle and upload persistence diagrams
        pd_pattern = os.path.join(pd_outdir, f"{prefix}PD1_*.pck")
        pd_files = glob.glob(pd_pattern)
        print("found pd files:", pd_files)
        if pd_files:
            zip_name = f"{prefix}pd_diagrams.zip"
            print(f"bundling {len(pd_files)} files into {zip_name} …")
            with zipfile.ZipFile(zip_name, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for fn in pd_files:
                    zf.write(fn, arcname=os.path.basename(fn))
            push_to_s3(zip_name, os.path.join(output_s3_path, zip_name))

        # upload scaffold generator files
        gen_dir = os.path.join(scaffold_outdir, "gen")
        if os.path.isdir(gen_dir):
            for fname in os.listdir(gen_dir):
                local_path  = os.path.join(gen_dir, fname)
                remote_path = os.path.join(output_s3_path, "homo_scaff", "gen", fname)
                push_to_s3(local_path, remote_path)
            print("generators uploaded to S3!")

        # bundle and upload Betti numbers JSONs
        betti_dir = os.path.join(scaffold_outdir, "betti")
        if os.path.isdir(betti_dir):
            betti_jsons = [os.path.join(betti_dir, f)
                        for f in os.listdir(betti_dir) if f.endswith(".json")]
            if betti_jsons:
                betti_zip = f"{prefix}betti_numbers.zip"
                print(f"bundling {len(betti_jsons)} betti-number files into {betti_zip} …")
                with zipfile.ZipFile(betti_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for fn in betti_jsons:
                        zf.write(fn, arcname=os.path.basename(fn))
                push_to_s3(betti_zip, os.path.join(output_s3_path, betti_zip))
                print("betti numbers zip uploaded to S3!")

        # bundle and upload Betti curves JSONs
        betti_curves_dir = os.path.join(scaffold_outdir, "betti_curves")
        if os.path.isdir(betti_curves_dir):
            curves_jsons = [os.path.join(betti_curves_dir, f)
                            for f in os.listdir(betti_curves_dir) if f.endswith(".json")]
            if curves_jsons:
                curves_zip = f"{prefix}betti_curves.zip"
                print(f"bundling {len(curves_jsons)} betti-curve files into {curves_zip} …")
                with zipfile.ZipFile(curves_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for fn in curves_jsons:
                        zf.write(fn, arcname=os.path.basename(fn))
                push_to_s3(curves_zip, os.path.join(output_s3_path, curves_zip))
                print("betti curves zip uploaded to S3!")


        print("all uploads complete.")
