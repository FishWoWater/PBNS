import os
import sys
import numpy as np
from scipy import sparse
import tensorflow as tf
from scipy.spatial import cKDTree

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Layers import *

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from IO import *
from util import *
from values import *
import pickle as pkl

class Resizer:
    # Builds models
    def __init__(self, object, body, checkpoint=None, pose_mode=0):
        """
        Args:
        - object: name of the outfit/garment OBJ file (within the same folder as this file)
        - body: name of the body MAT file (within the same folder), as described by README.md
        - checkpoint: name of the checkpoint NPY file (optional)
        file names WITHOUT extensions ('.obj', '.mat', '.npy')
        """
        self.pose_mode = pose_mode
        self._object = object
        # body data
        # init simplified smpl (for shape only)
        if self.pose_mode == 0:
            self._read_body(body)
            self._init_smpl()
        elif self.pose_mode == 1:
            body_data = pkl.load(open(os.path.abspath(os.path.dirname(__file__)) + '/' + body, "rb"))
            self._init_std_smpl(body_data)
        else:
            body_data = pkl.load(open(os.path.abspath(os.path.dirname(__file__)) + '/' + body, "rb"))
            self._init_posed_smpl(body_data)
        # outfit data
        self._read_outfit()
        # build
        self._build()
        # load pre-trained
        # the code does not check if checkpoint, object and body are consistent
        if checkpoint is not None:
            print("Loading pre-trained model: " + checkpoint)
            self.load(checkpoint)
    
    def _read_body(self, body_mat):
        if not body_mat.endswith('.mat'): body_mat = body_mat + '.mat'
        body_data = loadInfo(os.path.abspath(os.path.dirname(__file__)) + '/' + body_mat)
        self._shape = body_data['shape']
        self._gender = body_data['gender']

    def _init_std_smpl(self, body_data):
        # _pose, _shape, aligned with outfit
        self._shape = body_data['shape']
        self._pose = body_data['pose'] if 'pose' in body_data else None
        self._gender = body_data['gender']

        print('prepare the SMPL model, in particular, the transformation matrix')
        self._J_regressor = tf.constant(
            np.array(body_data['J_regressor'].todense(),
                     dtype=np.float64)
        )
        self._weights = tf.constant(body_data['weights'], dtype=np.float64)
        self._posedirs = tf.constant(body_data['posedirs'], dtype=np.float64)
        self._v_template = tf.constant(body_data['v_template'], dtype=np.float64)
        self._shapedirs = tf.constant(body_data['shapedirs'], dtype=np.float64)
        f = body_data['f']

        self._kintree_table = body_data['kintree_table']
        id_to_col = {self._kintree_table[1, i]: i for i in range(self._kintree_table.shape[1])}
        self._parent = {
            i: id_to_col[self._kintree_table[0, i]]
            for i in range(1, self._kintree_table.shape[1])
        }
        # for compatibility with `self._body(shape)`
        self._smpl = {}
        self._smpl['v_template'] = body_data['v_template']
        self._smpl['shapedirs'] = body_data['shapedirs']
        self._smpl['body_weights'] = body_data['weights']
        self._smpl['faces'] = body_data['f'].astype(np.int32)
    
    def _init_smpl(self):
        # path
        mpath = os.path.abspath(os.path.dirname(__file__)) + '/smpl/model_[G].mat'
        mpath = mpath.replace('[G]', 'm' if self._gender else 'f')
        # load smpl
        self._smpl = loadInfo(mpath)

    def _init_posed_smpl(self, body_data):
        # path
        self._shape = np.zeros((10, ))
        self._gender = body_data['gender']
        self._smpl = {
            "v_template": body_data['v_template'], 
            'shapedirs': body_data['shapedirs'],
            'faces':    body_data['f'].astype(np.int32)
        }
        
    def _read_outfit(self):
        """ Outfit data """
        root = os.path.abspath(os.path.dirname(__file__))
        self._T, F = readOBJ(root + '/' + self._object + '.obj')
        self._F = quads2tris(F) # triangulate
        self._E = faces2edges(self._F)
        self._L = laplacianMatrix(self._F)
        # the first dimension should be the number of edges
        self._neigh_F = neigh_faces(self._F, self._E) # edges of graph representing face connectivity
        # blend shapes prior
        self._blendshapes_prior()
        """ Outfit config """
        # - 'layers': list of lists. Each list represents a layer. Each sub-list contains the indices of the vertices belonging to that layer.
        # - 'edge': per-vertex weights for edge loss
        # - 'bend': per-vertex weights for bending loss
        # - 'pin': pinned vertex indices
        N = self._T.shape[0] # n. verts
        self._config = {}        
        config_path = root + '/' + self._object + '_config.mat'
        if os.path.isfile(config_path):
            self._config = loadInfo(config_path)
        else:
            print("Outfit config file not found. Using default config.")
        # DEFAULT MISSING CONFIG FIELDS
        # layers
        if 'layers' not in self._config:
            self._config['layers'] = [list(range(N))]
        # edge
        if 'edge' not in self._config:
            self._config['edge'] = np.ones((N,), np.float32)
        # bend
        if 'bend' not in self._config:
            self._config['bend'] = np.ones((N,), np.float32)

        # convert per-vertex 'edge' weights to per-edge weights
        _edge_weights = np.zeros((len(self._E),), np.float32)
        for i,e in enumerate(self._E):
            _edge_weights[i] = self._config['edge'][e].mean()
        self._config['edge'] = _edge_weights
        # convert per-vertex 'bend' weights to per-hinge weights (hinge = adjacent faces)
        _bend_weights = np.zeros((len(self._neigh_F),), np.float32)
        for i,n_f in enumerate(self._neigh_F):
            # get common verts
            v = list(set(self._F[n_f[0]]).intersection(set(self._F[n_f[1]])))
            _bend_weights[i] = self._config['bend'][v].mean()
        self._config['bend'] = _bend_weights
        # setup the cloth's skinning weight
        if self.pose_mode == 1:
            # search the idx
            tree = cKDTree(self._body(self._shape))
            idx = tree.query(self._T, n_jobs=-1)[1]
            self._cloth_weights = self._smpl['body_weights'][idx]
        else:
            self._cloth_weights = None
    
    def _blendshapes_prior(self, it=100):
        tree = cKDTree(self._body(self._shape))
        idx = tree.query(self._T, n_jobs=-1)[1]
        self._BS0 = self._smpl['shapedirs'][idx]
        # smooth shapedirs
        if it:
            self._BS0 = self._BS0.reshape((-1, 3 * 10))
            for i in range(it):
                self._BS0 = self._L @ self._BS0
            self._BS0 = self._BS0.reshape((-1, 3, 10))

    def _rodrigues(self, r):
        theta = tf.norm(r + tf.random_normal(r.shape, 0, 1e-8, dtype=tf.float64), axis=(1, 2), keepdims=True)
        # avoid divide by zero
        r_hat = r / theta
        cos = tf.cos(theta)
        z_stick = tf.zeros(theta.get_shape().as_list()[0], dtype=tf.float64)
        m = tf.stack(
            (z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1], r_hat[:, 0, 2], z_stick,
             -r_hat[:, 0, 0], -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick), axis=1)
        m = tf.reshape(m, (-1, 3, 3))
        i_cube = tf.expand_dims(tf.eye(3, dtype=tf.float64), axis=0) + tf.zeros(
            (theta.get_shape().as_list()[0], 3, 3), dtype=tf.float64)
        A = tf.transpose(r_hat, (0, 2, 1))
        B = r_hat
        dot = tf.matmul(A, B)
        R = cos * i_cube + (1 - cos) * dot + tf.sin(theta) * m
        return R

    def _with_zeros(self, x):
        ret = tf.concat(
            # B x 3 x 4 cat B x 1 x 4 => B x 4 x 4
            (x, tf.tile(tf.constant([[[0.0, 0.0, 0.0, 1.0]]], dtype=tf.float64), (x.shape[0], 1, 1))),
            axis=1
        )
        return ret

    def _pack(self, x):
        # B x C x 4 x 1 => B x C x 4 x 4
        ret = tf.concat(
            (tf.zeros((x.get_shape().as_list()[0], x.get_shape().as_list()[1], 4, 3), dtype=tf.float64), x),
            axis=3
        )
        return ret

    def _body_tf_wpose(self, shape):
        batch_num = shape.shape[0]
        shape = tf.cast(shape, tf.float64)
        # shape = tf.constant(shape.astype(np.float64), dtype=tf.float64)
        # read pose from self
        v_shaped = tf.tensordot(shape, self._shapedirs, axes=[[1], [2]]) + self._v_template
        J = tf.matmul(self._J_regressor, v_shaped)
        pose = tf.tile(tf.expand_dims(self._pose, axis=0), (batch_num, 1))
        pose_cube = tf.reshape(pose, (-1, 1, 3))
        R_cube_big = tf.reshape(rodrigues(pose_cube), (batch_num, -1, 3, 3))

        R_cube = R_cube_big[:, 1:]
        I_cube = tf.expand_dims(tf.eye(3, dtype=tf.float64), axis=0) + \
                 tf.zeros((batch_num, R_cube.get_shape()[1], 3, 3), dtype=tf.float64)
        lrotmin = tf.squeeze(tf.reshape((R_cube - I_cube), (batch_num, -1, 1)), axis=2)
        # v_posed = v_shaped + tf.tensordot(self._posedirs, lrotmin, axes=[[2], [0]])
        v_posed = v_shaped + tf.tensordot(lrotmin, self._posedirs, axes=[[1], [2]])

        results = []
        results.append(
            self._with_zeros(tf.concat((R_cube_big[:, 0], tf.reshape(J[:, 0, :], (-1, 3, 1))), axis=2))
        )
        for i in range(1, self._kintree_table.shape[1]):
            results.append(
                tf.matmul(
                    results[self._parent[i]],
                    self._with_zeros(
                        tf.concat(
                            (R_cube_big[:, i], tf.reshape(J[:, i, :] - J[:, self._parent[i], :], (-1, 3, 1))),
                            axis=2
                        )
                    )
                )
            )
        stacked = tf.stack(results, axis=1)
        results = stacked - \
                  self._pack(
                      tf.matmul(
                          stacked,
                          tf.reshape(
                              tf.concat((J, tf.zeros((batch_num, 24, 1), dtype=tf.float64)), axis=2),
                              (batch_num, 24, 4, 1)
                          )
                      )
                  )
        self._body_transformation = results
        T = tf.tensordot(results, self._weights, axes=((1), (1)))
        T = tf.transpose(T, (0, 3, 1, 2))
        rest_shape_h = tf.concat(
            (v_posed, tf.ones((batch_num, v_posed.get_shape().as_list()[1], 1), dtype=tf.float64)),
            axis=2
        )
        v = tf.matmul(T, tf.reshape(rest_shape_h, (batch_num, -1, 4, 1)))
        v = tf.reshape(v, (batch_num, -1, 4))[:, :, :3]
        v = tf.cast(v, tf.float32)
        return v

    def _body(self, shape):
        return self._smpl['v_template'] + np.einsum('a,bca->bc', shape, self._smpl['shapedirs'])
            
    def _body_tf(self, shapes):
        # TODO: add transformation based on some specific pose
        return self._smpl['v_template'][None] + tf.einsum('ab,cdb->acd', shapes, self._smpl['shapedirs'])

    def _flat_resize_wpose(self, shape):
        batch_num = shape.shape[0]
        v_cloth_shaped = self._T[None] + np.einsum('ab,cdb->acd', (shape - self._shape[None]), self._BS0)
        # v_cloth_shaped = np.tile(v_cloth_shaped[None], (batch_num, 1, 1, 1))
        # import ipdb; ipdb.set_trace()
        v_cloth_shaped = tf.constant(v_cloth_shaped, dtype=tf.float64)
        # transform the cloth according to self._pose
        T = tf.tensordot(self._body_transformation, self._cloth_weights, axes=((1), (1)))
        T = tf.transpose(T, (0, 3, 1, 2))
        rest_shape_h = tf.concat(
            (v_cloth_shaped, tf.ones((batch_num, v_cloth_shaped.get_shape().as_list()[1], 1), dtype=tf.float64)),
            axis=2
        )
        v = tf.matmul(T, tf.reshape(rest_shape_h, (batch_num, -1, 4, 1)))
        v = tf.reshape(v, (batch_num, -1, 4))[:, :, :3]
        return v.numpy().astype(np.float32)

    def _flat_resize(self, shape):
        # TODO: add transformation based on some specific pose
        return self._T[None] + np.einsum('ab,cdb->acd', (shape - self._shape[None]), self._BS0)
        
    def _compute_edges(self, T):
        return np.sqrt(np.sum((T[:,self._E[:,0]] - T[:,self._E[:,1]]) ** 2, -1))
        
    def _compute_area(self, T):
        u = T[:,self._F[:,2]] - T[:,self._F[:,0]]
        v = T[:,self._F[:,1]] - T[:,self._F[:,0]]
        return np.linalg.norm(np.cross(u,v), axis=-1).sum(-1) / 2.0
        
    # Builds model
    def _build(self):
        # Shape MLP
        self._mlp = [
            FullyConnected((12, 32), act=tf.nn.selu, name='fc0'),
            FullyConnected((32, 32), act=tf.nn.selu, name='fc1'),
            FullyConnected((32, 32), act=tf.nn.selu, name='fc2'),
            FullyConnected((32, 32), act=tf.nn.selu, name='fc3')
        ]

        # Blend Shapes matrix
        shape = self._mlp[-1].w.shape[-1], self._T.shape[0], 3
        self._dBS = tf.Variable(tf.initializers.glorot_normal()(shape), name='dBS')
            
    # Returns list of model variables
    def gather(self):
        vars = [self._dBS]
        for l in self._mlp:
            vars += l.gather()
        return vars
    
    # loads pre-trained model
    def load(self, checkpoint):
        # checkpoint: path to pre-trained model
        # list vars
        vars = self.gather()
        # load vars values
        if not checkpoint.endswith('.npy'): checkpoint += '.npy'
        values = np.load(checkpoint, allow_pickle=True)[()]
        # assign
        for v in vars:
            try: 
                v.assign(values[v.name])
            except: print("Mismatch between model and checkpoint: " + v.name)
        try:
            self._BS0 = values['BS0']
        except:
            print("Missing BS0: expect misbehavior")
        
    def save(self, checkpoint):
        # checkpoint: path to pre-trained model
        print("\tSaving checkpoint: " + checkpoint)
        # get  TF vars values
        values = {v.name: v.numpy() for v in self.gather()}
        # save BS0
        values['BS0'] = self._BS0
        # save weights
        if not checkpoint.endswith('.npy'): checkpoint += '.npy'
        np.save(checkpoint, values)

    def __call__(self, X, tightness):
        # X : smpl shape (10,)
        # tightness: tightness (2,)
        """ Body """
        if self.pose_mode != 1: B = self._body_tf(X).numpy()
        else:   B = self._body_tf_wpose(X).numpy()
        """ NUMPY """
        # flat resize
        if self.pose_mode != 1: T = self._flat_resize(X.numpy())
        else:   T = self._flat_resize_wpose(X.numpy())
        # save the cloth
        # writeOBJ("cloth2.obj", T[0], self._F)
        # writeOBJ("human2.obj", B[0], self._smpl['faces'])
        # exit(0)
        # edge resize
        X_e = X.numpy()
        X_e[:,0] += tightness[:,0]
        X_e[:,1] += tightness[:,1]
        X_e[:,2:] = 0
        if self.pose_mode != 1: T_e = self._flat_resize(X_e)
        else:   T_e = self._flat_resize_wpose(X_e)
        # edges
        E = self._compute_edges(T_e)
        # areas
        A = self._compute_area(T_e)
        """ TENSORFLOW """
        # Pose MLP
        X = tf.concat((X, tightness), -1)
        for l in self._mlp:
            X = l(X)
        # Blend shapes
        self.D = tf.einsum('ab,bcd->acd', X, self._dBS)
        self.D = tf.cast(self.D, tf.float32)
        # final resize
        T = T + self.D
        return T, E, A, B
