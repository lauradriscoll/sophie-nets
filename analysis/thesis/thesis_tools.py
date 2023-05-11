import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os 

from task import rules_dict


def get_interp_filename(ri_set, epoch_list, t_set, ruleset):
  rule1, rule2 = np.array(rules_dict[ruleset])[ri_set]
  filename = rule1+'_'+rule2+'_'+'_'.join(epoch_list)+'_x'+str(t_set[0])+'_x'+str(t_set[1])
  return filename


"""
Gets the fp_struct for the fixed points with the specified inputs.
The specification specifies the dynamics at step_i
  interpolated between the inputs at ri_set | epoch_list | trial_num
"""
def get_fp_struct(ri_set, epoch_list, trial_num, step_i, ruleset, m, q_tol_name,
              q_thresh = 1e-6,
              step_file = 'interp_tasks_small_init_stim'):
    f = get_interp_filename(ri_set, epoch_list, trial_num, ruleset)    
    tasks_str = '_'.join([rules_dict[ruleset][ri_set[0]],rules_dict[ruleset][ri_set[1]]])
    save_dir = os.path.join(m,step_file,tasks_str)
    fp_struct = np.load(os.path.join(save_dir,q_tol_name,f+'_step_'+str(step_i)+'.npz'))
    return fp_struct 

"""
Gets the indices of the fixed points below a particular threshold
"""
def get_fps_below_qthresh_idx(ri_set, epoch_list, trial_num, step_i, ruleset, m, q_tol_name,
              q_thresh = 1e-6,
              step_file = 'interp_tasks_small_init_stim'):

  
  fp_struct = get_fp_struct(ri_set, epoch_list, trial_num, step_i, ruleset, m, q_tol_name, q_thresh, step_file)

  return  np.where(fp_struct['qstar'] < q_thresh)[0] # get indices of fixed points which have qstart below the threshold

"""
Gets the values of the fixed points below a certain threshold
"""
def get_xstar(ri_set, epoch_list, trial_num, step_i, ruleset, m, q_tol_name,
              q_thresh = 1e-6,
              step_file = 'interp_tasks_small_init_stim'):

  fp_struct = get_fp_struct(ri_set, epoch_list, trial_num, step_i, ruleset, m, q_tol_name, q_thresh, step_file)
  indices = get_fps_below_qthresh_idx(ri_set, epoch_list, trial_num, step_i, ruleset, m, q_tol_name, q_thresh, step_file)
  return fp_struct['xstar'][indices]

"""
Gets the eigenvalues of the Jacobian at the fixed poitns below a certain threshold.
"""
def get_eigenvals(ri_set, epoch_list, trial_num, step_i, ruleset, m, q_tol_name,
              q_thresh = 1e-6,
              step_file = 'interp_tasks_small_init_stim'):
  fp_struct = get_fp_struct(ri_set, epoch_list, trial_num, step_i, ruleset, m, q_tol_name, q_thresh, step_file)
  indices = get_fps_below_qthresh_idx(ri_set, epoch_list, trial_num, step_i, ruleset, m, q_tol_name, q_thresh, step_file)
  
  J_xstar = fp_struct['J_xstar'][indices]
  return np.linalg.eigvals(J_xstar)




"""
Runs PCA on all the fixed points discovered when interpolating between the
tasks specified by ri_set | epoch_list | trial num. 

n_components is the number of PCA dimenstions and n_interp is the number of interpolation steps.
"""
def get_pca(ri_set, epoch_list, trial_num, ruleset, m, q_tol_name, n_components = 3, n_interp = 20, q_thresh = 1e-6):

  # Collect the results of get_xstar() for each step_i
  results = []
  for ti in range(len(ri_set)-1):  
    for step_i in range(n_interp + 1):
      xstar_i = get_xstar(ri_set[ti:ti+2], epoch_list[ti:ti+2], trial_num[ti:ti+2], step_i, ruleset, m, q_tol_name, q_thresh = q_thresh)
      results.append(xstar_i)

  # Concatenate the results into a single numpy array
  data = np.concatenate(results, axis=0)

  # Apply PCA to the concatenated data
  pca = PCA(n_components=n_components)
  pca.fit(data)

  return pca

"""
Plots the fixed points discovered when interpolating between tasks.
The first dimension is the "interpolation dimension" and separates each interpolated step. 
The remaining 2 dimensions are from PCA. 
"""
def plot_interp_pca(ri_set, epoch_list, trial_num, ruleset, m, q_tol_name, q_thresh = 1e-6, n_interp = 20):
  
  pca = get_pca(ri_set, epoch_list, trial_num, ruleset, m, q_tol_name, n_components = 2, n_interp = 20, q_thresh = q_thresh)

  fig_width = 10
  fig_height = 12
  fig = plt.figure(figsize=(fig_width,fig_height))
  ax = fig.add_subplot(111, projection='3d')

  cmap_grad = plt.get_cmap('plasma')
  s = 200

  for ti in range(len(ri_set)-1):
    for step_i in range(n_interp + 1):
      step_i_frac = step_i/n_interp
      c = cmap_grad((step_i/n_interp + ti)/(len(ri_set)-1))
      
      eigs = get_eigenvals(ri_set[ti:ti+2], epoch_list[ti:ti+2], trial_num[ti:ti+2], step_i, ruleset, m, q_tol_name, q_thresh = q_thresh)
      transformed_data = pca.transform(get_xstar(ri_set[ti:ti+2], epoch_list[ti:ti+2], trial_num[ti:ti+2], step_i, ruleset, m, q_tol_name, q_thresh = q_thresh))

      for i in range(len(transformed_data)):
        if np.abs(np.max(eigs[i])) > .99:
          facecolors = 'w'
        else:
          facecolors = c
        
        ax.scatter(step_i_frac + ti, transformed_data[i][0], transformed_data[i][1],  
          'o', s = s, edgecolors = c, facecolors = facecolors, alpha = .8)

  return ax

def plot_interp_pca1D(ri_set, epoch_list, trial_num, ruleset, m, q_tol_name, q_thresh = 1e-6, n_interp = 20):
  
  pca = get_pca(ri_set, epoch_list, trial_num, ruleset, m, q_tol_name,  n_components = 1, n_interp = 20, q_thresh = q_thresh)

  fig_width = 10
  fig_height = 10
  fig = plt.figure(figsize=(fig_width,fig_height))
  ax = fig.add_subplot(111)

  cmap_grad = plt.get_cmap('plasma')
  s = 200

  for ti in range(len(ri_set)-1):
    for step_i in range(n_interp + 1):
      step_i_frac = step_i/n_interp
      c = cmap_grad((step_i/n_interp + ti)/(len(ri_set)-1))
      
      eigs = get_eigenvals(ri_set[ti:ti+2], epoch_list[ti:ti+2], trial_num[ti:ti+2], step_i, ruleset, m, q_tol_name, q_thresh = q_thresh)
      transformed_data = pca.transform(get_xstar(ri_set[ti:ti+2], epoch_list[ti:ti+2], trial_num[ti:ti+2], step_i, ruleset, m, q_tol_name, q_thresh = q_thresh))

      for i in range(len(transformed_data)):
        if np.abs(np.max(eigs[i])) > .99:
          facecolors = 'w'
        else:
          facecolors = c
        
        ax.scatter(step_i_frac + ti, transformed_data[i][0], s = s, edgecolors = c, facecolors = facecolors) 
          #'o', s = s, edgecolors = c, facecolors = facecolors, alpha = .8)

  return ax