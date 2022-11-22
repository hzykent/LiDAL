import os
import glob
import numpy as np
import multiprocessing as mp
from k_means_constrained import KMeansConstrained
import pickle


def process_frame(file_path):
    
    # Load point data
    print('Processing {}'.format(file_path))
    raw_data = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    coords_p = raw_data[:, :3]
    
    # Constrained KMeans Clustering
    clf = KMeansConstrained(n_clusters = 20, size_min = int(coords_p.shape[0] / 20 * 0.95), size_max = int(coords_p.shape[0] / 20 * 1.05), n_init=1, max_iter=1, random_state=0)
    clf.fit_predict(coords_p)
    
    # Save sv label
    label_path = 'Processing_files/SK/super_voxel/KMeans/' + file_path.split('/')[-3] + '/' + file_path.split('/')[-1][:-3] + 'npy'
    np.save(label_path, clf.labels_)
    

if __name__ == "__main__":

    train_split = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']

    train_files = []
    if not os.path.exists('Processing_files/SK'):
        os.makedirs('Processing_files/SK')
    if not os.path.exists('Processing_files/SK/super_voxel'):
        os.makedirs('Processing_files/SK/super_voxel')
    if not os.path.exists('Processing_files/SK/super_voxel/KMeans'):
        os.makedirs('Processing_files/SK/super_voxel/KMeans')
    for i_folder in train_split:
        train_files += sorted(glob.glob('Semantic_kitti/dataset/sequences/{}'.format(i_folder) + '/velodyne/*.bin'))
        if not os.path.exists('Processing_files/SK/super_voxel/KMeans/' + i_folder):
            os.makedirs('Processing_files/SK/super_voxel/KMeans/' + i_folder)

    
    # Calculate super voxels
    # multi-processing
    pf_pool = mp.Pool(processes=20)
    pf_pool.map(process_frame, train_files)
    pf_pool.close()
    pf_pool.join()

    # Extract infos for super voxels
    sv_files = []
    for i_folder in train_split:
        sv_files += sorted(glob.glob('Processing_files/SK/super_voxel/KMeans/{}/*.npy'.format(i_folder)))

    sv_id_count = 0
    id2sv = []
    for file_path in sv_files:
        
        # Load super voxel labels
        sv_label = np.load(file_path)
        
        # Calculate sv2point
        sv2point = []
        for sv_l in np.unique(sv_label):
            p_ids = np.where(sv_label == sv_l)[0]
            sv2point += [p_ids]
        
        # sv_id
        sv_id = np.arange(len(sv2point)) + sv_id_count
        sv_id_count += len(sv2point)
        
        # Save
        save_path = file_path[:-3] + 'pickle'
        with open(save_path, 'wb') as fjson:
            pickle.dump((sv_id, sv2point), fjson) 
        
        # Construct mapping from sv_id to super voxels    
        id2sv += [(file_path.split('/')[-2], file_path.split('/')[-1][:-4], id) for id in np.arange(len(sv2point))]

    with open('Processing_files/SK/super_voxel/KMeans/id2sv.pickle', 'wb') as fjson:
        pickle.dump(id2sv, fjson)
