import os
import glob
import numpy as np
import multiprocessing as mp
import utils.pypcd as pypcd
import pickle


def process_frame(file_path):
    
    # Load point data
    raw_data = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    coords_p = raw_data[:, :3]
    
    # Save as pcd
    coords_arr = coords_p.ravel().view(dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    pc = pypcd.PointCloud.from_array(coords_arr)
    pcd_path = file_path[:-3] + 'pcd'
    pc.save(pcd_path)
    
    # Supervoxel clustering
    os.system("./pcl_related/supervoxel_clustering -p " + pcd_path + " -o " + pcd_path)
    pc_label = pypcd.PointCloud.from_path(pcd_path)
    sv_label = np.array(pc_label.pc_data['label'], dtype=np.int)
    
    # Save sv label
    label_path = 'Processing_files/SK/super_voxel/VCCS/' + pcd_path.split('/')[-3] + '/' + pcd_path.split('/')[-1][:-3] + 'npy'
    np.save(label_path, sv_label)
    
    # Clear
    os.system("rm " + pcd_path)
    
    

if __name__ == "__main__":

    train_split = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']

    train_files = []
    if not os.path.exists('Processing_files/SK'):
        os.makedirs('Processing_files/SK')
    if not os.path.exists('Processing_files/SK/super_voxel'):
        os.makedirs('Processing_files/SK/super_voxel')
    if not os.path.exists('Processing_files/SK/super_voxel/VCCS'):
        os.makedirs('Processing_files/SK/super_voxel/VCCS')
    for i_folder in train_split:
        train_files += sorted(glob.glob('Semantic_kitti/dataset/sequences/{}'.format(i_folder) + '/velodyne/*.bin'))
        if not os.path.exists('Processing_files/SK/super_voxel/VCCS/' + i_folder):
            os.makedirs('Processing_files/SK/super_voxel/VCCS/' + i_folder)
    
    # Calculate super voxels
    # multi-processing
    pf_pool = mp.Pool(processes=12)
    pf_pool.map(process_frame, train_files)
    pf_pool.close()
    pf_pool.join()

    # Extract infos for super voxels
    sv_files = []
    for i_folder in train_split:
        sv_files += sorted(glob.glob('Processing_files/SK/super_voxel/VCCS/{}/*.npy'.format(i_folder)))

    sv_id_count = 0
    id2sv = []
    for file_path in sv_files:
        
        # Load super voxel labels
        sv_label = np.load(file_path)
        
        # Calculate sv2point
        sv2point = []
        for sv_l in np.unique(sv_label):
            if sv_l != 0:
                # Prune < 100 points
                p_ids = np.where(sv_label == sv_l)[0]
                if len(p_ids) > 100:
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

    with open('Processing_files/SK/super_voxel/VCCS/id2sv.pickle', 'wb') as fjson:
        pickle.dump(id2sv, fjson)