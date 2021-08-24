import torch
import torch.optim as optim
import os
import shutil
import argparse
from tqdm import tqdm
import time, datetime
from collections import defaultdict
import pandas as pd
from src import config
from src.checkpoints import CheckpointIO
from src.utils.io import export_pointcloud
from src.utils.visualize import visualize_data
from src.utils.voxels import VoxelGrid
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(
    description='Extract meshes from occupancy process.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

out_dir = cfg['training']['out_dir']
generation_dir = os.path.join(out_dir, cfg['generation']['generation_dir'])
out_time_file = os.path.join(generation_dir, 'time_generation_full.pkl')
out_time_file_class = os.path.join(generation_dir, 'time_generation.pkl')


input_type = cfg['data']['input_type']
vis_n_outputs = cfg['generation']['vis_n_outputs']
if vis_n_outputs is None:
    vis_n_outputs = -1

# Dataset
dataset = config.get_dataset('test', cfg, return_idx=True)
print('test_split:', cfg['data']['test_split'])

# Model
model = config.get_model(cfg, device=device, dataset=dataset)

checkpoint_io = CheckpointIO(out_dir, model=model)
checkpoint_io.load(cfg['test']['model_file'])

# Generator
generator = config.get_generator(model, cfg, device=device)

# Determine what to generate
generate_mesh = cfg['generation']['generate_mesh']
generate_pointcloud = cfg['generation']['generate_pointcloud']

if generate_mesh and not hasattr(generator, 'generate_mesh'):
    generate_mesh = False
    print('Warning: generator does not support mesh generation.')

if generate_pointcloud and not hasattr(generator, 'generate_pointcloud'):
    generate_pointcloud = False
    print('Warning: generator does not support pointcloud generation.')


# Loader
test_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, num_workers=0, shuffle=False)

# Statistics
time_dicts = []

# Generate
# model.eval()

# Count how many models already created
model_counter = defaultdict(int)

for it, data in enumerate(tqdm(test_loader)):
    # Output folders
    mesh_dir = os.path.join(generation_dir, 'meshes')
    pointcloud_dir = os.path.join(generation_dir, 'pointcloud')
    in_dir = os.path.join(generation_dir, 'input')
    generation_vis_dir = os.path.join(generation_dir, 'vis')
    log_dir = os.path.join(generation_dir, 'log')

    # Get index etc.
    idx = data['idx'].item()

    try:
        model_dict = dataset.get_model_dict(idx)
    except AttributeError:
        model_dict = {'model': str(idx), 'category': 'n/a'}
    
    modelname = model_dict['model']
    category_id = model_dict.get('category', 'n/a')

    try:
        category_name = dataset.metadata[category_id].get('name', 'n/a')
    except AttributeError:
        category_name = 'n/a'

    if category_id != 'n/a':
        mesh_dir = os.path.join(mesh_dir, str(category_id))
        pointcloud_dir = os.path.join(pointcloud_dir, str(category_id))
        in_dir = os.path.join(in_dir, str(category_id))

        folder_name = str(category_id)
        if category_name != 'n/a':
            folder_name = str(folder_name) + '_' + category_name.split(',')[0]

        generation_vis_dir = os.path.join(generation_vis_dir, folder_name)

        log_dir = os.path.join(log_dir, str(category_id), modelname)
    # Create directories if necessary
    if vis_n_outputs >= 0 and not os.path.exists(generation_vis_dir):
        os.makedirs(generation_vis_dir)

    if generate_mesh and not os.path.exists(mesh_dir):
        os.makedirs(mesh_dir)

    if generate_pointcloud and not os.path.exists(pointcloud_dir):
        os.makedirs(pointcloud_dir)

    if not os.path.exists(in_dir):
        os.makedirs(in_dir)
    
    #record the loss curve for each test intance
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = SummaryWriter(log_dir)

    # Timing dict
    time_dict = {
        'idx': idx,
        'class id': category_id,
        'class name': category_name,
        'modelname': modelname,
    }
    time_dicts.append(time_dict)

    # Generate outputs
    out_file_dict = {}

    # Also copy ground truth
    if cfg['generation']['copy_groundtruth']:
        modelpath = os.path.join(
            dataset.dataset_folder, category_id, modelname, 
            cfg['data']['watertight_file'])
        out_file_dict['gt'] = modelpath

    #load pretrained model in every iteration
    checkpoint_io = CheckpointIO(out_dir, model=model)
    checkpoint_io.load(cfg['test']['model_file'])
    filename = os.path.join(out_dir, cfg['test']['model_file'])
    state_dict = torch.load(filename).get('model')

    def generate_mesh_func(modename, iter, is_final=False, th=0.4, suffix='th0.4'):
        # Generate
        generator.threshold = th
        model.eval()
        t0 = time.time()
        if cfg['generation']['sliding_window']:
            if it == 0:
                print('Process scenes in a sliding-window manner')
            out = generator.generate_mesh_sliding(data)
        else:
            out = generator.generate_mesh(data)
        time_dict['mesh'] = time.time() - t0

        # Get statistics
        try:
            mesh, stats_dict = out
        except TypeError:
            mesh, stats_dict = out, {}
        time_dict.update(stats_dict)

        # Write output
        if not is_final:
            if iter > 0:
                mesh_out_file = os.path.join(mesh_dir, '%s_iter%d_%s.off' % (modelname, iter, suffix))
            else:
                mesh_out_file = os.path.join(mesh_dir, '%s.off' % (modelname))
        else:
            mesh_out_file = os.path.join(mesh_dir, '%s_final_%s.off' % (modelname, suffix))
        mesh.export(mesh_out_file)
        out_file_dict['mesh'] = mesh_out_file

    # Generate results before test-time optimization (results of pretrained ConvONet)
    if generate_mesh:
        th = cfg['test']['threshold']
        generate_mesh_func(modelname, 0, th=th, suffix=f"th{th}")
    
    # Intialize training using pretrained model, and then optimize network parameters for each observed input.
    ft_onlyencoder = cfg['test_optim']['onlyencoder']
    print('ft only encoder', ft_onlyencoder)
    lr, lr_decay = cfg['test_optim']['learning_rate'], cfg['test_optim']['decay_rate']
    iter, n_iter, n_step = 0, cfg['test_optim']['n_iter'], cfg['test_optim']['n_step']
    batch_size = cfg['test_optim']['batch_size']
    thres_list = cfg['test_optim']['threshold']

    if ft_onlyencoder:
        print('only optimize encoder')
        optimizer = optim.Adam(model.encoder.parameters(), lr=lr)
    else:
        print('optimize encoder & decoder')
        optimizer = optim.Adam(model.parameters(), lr=lr)
    trainer = config.get_trainer(model, optimizer, cfg, device=device)

    scene_index = int(data["idx"])
    
    time_dict['net_optim'] = 0
    time_dict['net_optim_iter'] = 0
    for iter in range(0, n_iter):
        t0 = time.time()
        
        # load cropped patches into a batch
        dataset.split = "train" 
        crop_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, indices=[scene_index]), batch_size=1)
        crop_data_lst = list()
        for _ in range(batch_size):
            added = False
            while not added:
                try:
                    crop_data_lst += [data for data in crop_loader] 
                    added = True
                except Exception as e:
                    print(f"encountering error: {e}")
                    pass

        cat_fn = lambda the_key: torch.cat([d[the_key] for d in crop_data_lst], dim=0) if isinstance(crop_data_lst[0][the_key], torch.Tensor) else {_key: torch.cat([d[the_key][_key] for d in crop_data_lst], dim=0) for _key in crop_data_lst[0][the_key].keys()}
        crop_data = dict(pointcloud_crop=True, **{k: cat_fn(k) for k in 
                                                  ["points", "points.occ", "points.normalized", "inputs", "inputs.ind", "inputs.mask"]})
        dataset.split = "test"

        loss = trainer.sign_agnostic_optim_cropscene_step(crop_data, state_dict)
        logger.add_scalar('test_optim/loss', loss, iter)
        time_dict['net_optim'] += time.time() - t0
        time_dict['net_optim_iter'] += 1
        
        t = datetime.datetime.now()
        print('[It %02d] iter_ft=%03d, loss=%.4f, time: %.2fs, %02d:%02d' % (it, iter, loss, time.time() - t0, t.hour, t.minute))
        if (iter + 1) % n_step == 0:
            lr = lr * lr_decay
            print('adjust learning rate to', lr)
            for g in optimizer.param_groups:
                g['lr'] = lr
            trainer = config.get_trainer(model, optimizer, cfg, device=device)
            if generate_mesh:
                for th in thres_list:
                    generate_mesh_func(modelname, iter, th=th, suffix=f"th{th}")
    if generate_mesh:
        for th in thres_list:
            generate_mesh_func(modelname, n_iter, is_final=True, th=th, suffix=f"th{th}")

    if cfg['generation']['copy_input']:
        # Save inputs
        if input_type == 'voxels':
            inputs_path = os.path.join(in_dir, '%s.off' % modelname)
            inputs = data['inputs'].squeeze(0).cpu()
            voxel_mesh = VoxelGrid(inputs).to_mesh()
            voxel_mesh.export(inputs_path)
            out_file_dict['in'] = inputs_path
        elif input_type ==  'pointcloud_crop':
            inputs_path = os.path.join(in_dir, '%s.ply' % modelname)
            inputs = data['inputs'].squeeze(0).cpu().numpy()
            export_pointcloud(inputs, inputs_path, False)
            out_file_dict['in'] = inputs_path
        elif input_type == 'pointcloud' or 'partial_pointcloud':
            inputs_path = os.path.join(in_dir, '%s.ply' % modelname)
            inputs = data['inputs'].squeeze(0).cpu().numpy()
            export_pointcloud(inputs, inputs_path, False)
            out_file_dict['in'] = inputs_path

    # Copy to visualization directory for first vis_n_output samples
    c_it = model_counter[category_id]
    if c_it < vis_n_outputs:
        # Save output files
        img_name = '%02d.off' % c_it
        for k, filepath in out_file_dict.items():
            ext = os.path.splitext(filepath)[1]
            out_file = os.path.join(generation_vis_dir, '%02d_%s%s'
                                    % (c_it, k, ext))
            shutil.copyfile(filepath, out_file)

    model_counter[category_id] += 1

# Create pandas dataframe and save
time_df = pd.DataFrame(time_dicts)
time_df.set_index(['idx'], inplace=True)
time_df.to_pickle(out_time_file)

# Create pickle files  with main statistics
time_df_class = time_df.groupby(by=['class name']).mean()
time_df_class.to_pickle(out_time_file_class)

# Print results
time_df_class.loc['mean'] = time_df_class.mean()
print('Timings [s]:')
print(time_df_class)
