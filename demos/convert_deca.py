import os, sys
import numpy as np
import shutil
import cv2
import pickle
from scipy.io import savemat
import argparse
from tqdm import tqdm
import torch
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets 
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from decalib.utils.tensor_cropper import transform_points
from smplx.renderer import Renderer

def load_pickle_file(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data

def write_pickle_file(pkl_path, data_dict):
    with open(pkl_path, 'wb') as fp:
        pickle.dump(data_dict, fp, protocol=2)


########## Perspective Camera

if __name__ == "__main__":
    device = 'cuda'
    data_root = r'TestSamples'
    images_dir = os.path.join(data_root, 'heads')

    img = cv2.imread(os.path.join(images_dir, os.listdir(images_dir)[0]))
    H, W = img.shape[:2]

    longFOV = np.pi/4
    focal = max(H, W) / 2 / np.tan(longFOV/2)
    if(W>H):
        yfov = np.arctan(H / focal / 2) * 2
    else:
        yfov = longFOV

    testdata = datasets.TestData(images_dir, iscrop=True, face_detector='fan', sample_step=10)

    # run DECA
    deca_cfg.model.use_tex = False
    deca = DECA(config = deca_cfg, device=device)
    renderer = Renderer(deca.flame.faces, resolution=(W, H), orig_img=True, wireframe=False)

    all_vertices = []
    all_tform = []
    dict = {'frames': {}, 'canonical_vertices': None}
    for i in tqdm(range(len(testdata))):
        name = testdata[i]['imagename']
        images = testdata[i]['image'].to(device)[None,...]
        with torch.no_grad():
            codedict = deca.encode(images)

            tform = testdata[i]['tform'][None, ...]
            tform = torch.inverse(tform).transpose(1,2).to(device)
            original_image = testdata[i]['original_image'][None, ...].to(device)

            opdict, orig_visdict = deca.decode(codedict, focal=focal, render_orig=True, original_image=original_image, tform=tform)

            vertices = opdict['trans_verts3'][0].cpu().numpy()

            vertices[..., 0] -= W / 2
            vertices[..., 1] -= H / 2
            vertices[..., 1] = -vertices[..., 1]
            vertices[..., 2] = -vertices[..., 2]
            vertices[..., 2] -= np.mean(vertices[..., 2])

            img = renderer.renderPerspectiveCamera(
                cv2.imread(testdata[i]['imagepath']),
                vertices,
                focal=focal,
                yfov=yfov,
                color=[0.6, 0.3, 0.1],
                mesh_filename=None
            )

            cv2.imwrite(f'TestSamples/Perspective/haha_{i}.png', img)

            ## scale the vertices coordinate in [-1, 1], for each axis, z keep along with x.
            vertices[..., 0] = vertices[..., 0] / (W / 2)
            vertices[..., 1] = vertices[..., 1] / (H / 2)
            vertices[..., 2] = vertices[..., 2] / (W / 2)
            all_tform.append(tform[0])

            dict['frames'][name] = {"vertices" : vertices, "pose": codedict['pose'][0].cpu().numpy(), "exp": codedict['exp'][0].cpu().numpy()}


    ## Calculate the canonical face vertices.
    codedict['pose'][:] = 0.
    codedict['exp'][:] = 0.
    all_tform = torch.stack(all_tform, axis=0)
    mean_tform = torch.mean(all_tform, axis=0)[None, ...]
    opdict, orig_visdict = deca.decode(codedict, focal=focal, render_orig=True, original_image=original_image, tform=mean_tform)

    mean_vertices = opdict['trans_verts3'][0].cpu().numpy()
    mean_vertices[..., 0] -= W / 2
    mean_vertices[..., 1] -= H / 2
    mean_vertices[..., 1] = -mean_vertices[..., 1]
    mean_vertices[..., 2] = -mean_vertices[..., 2]
    mean_vertices[..., 2] -= np.mean(mean_vertices[..., 2])

    mean_img = renderer.renderPerspectiveCamera(
        cv2.imread(testdata[0]['imagepath']),
        mean_vertices,
        focal=focal,
        yfov=yfov,
        color=[0.6, 0.3, 0.1],
        mesh_filename=None
    )

    cv2.imwrite(f'TestSamples/Perspective/canonical.png', mean_img)

    ## scale the vertices coordinate in [-1, 1], for each axis, z keep along with x.
    mean_vertices[..., 0] = mean_vertices[..., 0] / W / 2
    mean_vertices[..., 1] = mean_vertices[..., 1] / H / 2
    mean_vertices[..., 2] = mean_vertices[..., 2] / W / 2
    
    dict['canonical_vertices'] = mean_vertices
    
    write_pickle_file('TestSamples/Perspective/deca.pkl', dict)
'''

########## Orthographic Camera
if __name__ == "__main__":
    device = 'cuda'
    data_root = r'TestSamples'
    images_dir = os.path.join(data_root, 'images')

    img = cv2.imread(os.path.join(images_dir, os.listdir(images_dir)[0]))
    H, W = img.shape[:2]

    longFOV = np.pi/4
    focal = max(H, W) / 2 / np.tan(longFOV/2)
    if(W>H):
        yfov = np.arctan(H / focal / 2) * 2
    else:
        yfov = longFOV

    testdata = datasets.TestData(images_dir, iscrop=True, face_detector='fan', sample_step=10)

    # run DECA
    deca_cfg.model.use_tex = False
    deca = DECA(config = deca_cfg, device=device)
    renderer = Renderer(deca.flame.faces, resolution=(W, H), orig_img=True, wireframe=False)

    all_vertices = []
    all_tform = []
    dict = {'frames': {}}
    for i in tqdm(range(len(testdata))):
        name = testdata[i]['imagename']
        images = testdata[i]['image'].to(device)[None,...]
        with torch.no_grad():
            codedict = deca.encode(images)

            tform = testdata[i]['tform'][None, ...]
            tform = torch.inverse(tform).transpose(1,2).to(device)
            original_image = testdata[i]['original_image'][None, ...].to(device)

            opdict, orig_visdict = deca.decode(codedict, focal=focal, render_orig=True, original_image=original_image, tform=tform)

            vertices = opdict['trans_verts3'][0].cpu().numpy()

            vertices[..., 0] -= W / 2
            vertices[..., 1] -= H / 2
            vertices[..., 1] = -vertices[..., 1]
            vertices[..., 2] = -vertices[..., 2]
            vertices[..., 2] -= np.mean(vertices[..., 2])

            img = renderer.renderOrthographicCamera(
                cv2.imread(testdata[i]['imagepath']),
                vertices,
                color=[0., 0., 0.],
                mesh_filename=None
            )

            cv2.imwrite(f'TestSamples/Orthographic/{name}_mask.png', img)

            dict['frames'][name] = {"pose": codedict['pose'][0].cpu().numpy(), "exp": codedict['exp'][0].cpu().numpy()}

    write_pickle_file('TestSamples/Orthographic/deca_test.pkl', dict)
    # write_pickle_file(os.path.join(data_root, 'deca.pkl'), dict)
'''