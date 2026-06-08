"""Data loader for loading data and do POD or streaming SVD or balanced POD to generate reduced subspaces or preliminarily reduce the dimensionality of the FOM before official training."""
import numpy as np

class FOMDataloader:
    def __init__(self, params):
        self.params = params
       
    def get_traj(self, which_traj, which_snapshot, is_scaled=False, is_template_fitted=False):
        """Return the scaled data (R)q (N, T, M) of trajectories"""
        which_traj = np.atleast_1d(which_traj)
        which_snapshot = np.atleast_1d(which_snapshot)
        if is_scaled:
            if is_template_fitted:
                fname_Q = self.params.fname_traj_fitted_scaled
            else:
                fname_Q = self.params.fname_traj_scaled
        else:
            if is_template_fitted:
                fname_Q = self.params.fname_traj_fitted
            else:
                fname_Q = self.params.fname_traj
        num_traj, num_snapshot = len(which_traj), len(which_snapshot)
                
        Q0 = np.load(fname_Q % ("fom", which_traj[0]), mmap_mode='r') # shape (n_states, n_snapshots)
        num_states = Q0.shape[0]
        Q = np.empty((num_states, num_snapshot, num_traj))
        for i, traj_idx in enumerate(which_traj):
            Q[:,:,i] = np.load(fname_Q % ("fom", traj_idx), mmap_mode='r')[:, which_snapshot]
        
        return Q.squeeze() # shape (N, T, M) or (N, T)
    
    def get_shift_amount(self, which_traj, which_snapshot):
        """Return the shift amount c (T, M)"""
        which_traj = np.atleast_1d(which_traj)
        which_snapshot = np.atleast_1d(which_snapshot)
        fname_shift_amount = self.params.fname_shift_amount
        num_traj, num_snapshot = len(which_traj), len(which_snapshot)
        c = np.empty((num_snapshot, num_traj))
        
        for i, traj_idx in enumerate(which_traj):
            c[:,i] = np.load(fname_shift_amount % ("fom", traj_idx))[which_snapshot]
            
        return c.squeeze() # shape (T, M) or (T,)
    
    def get_time(self, which_snapshot):
        """Return the time vector (T, M)"""
        which_snapshot = np.atleast_1d(which_snapshot)
        fname_time = self.params.fname_time
        time = np.load(fname_time)[which_snapshot]
        return time