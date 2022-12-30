import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch3d.transforms as tt

from ..utils import generate_healpix_grid



class ImplicitSO3(nn.Module):
    """
    Represents a distribution over SO(3) as an implicit function.
    Specifically, this is a fully connected network which takes as input
    a visual description vector and a query rotation, and outputs
    a single scalar which can be converted to a log likelihood via normalization.
    By querying many values we can build up a probability distribution over SO(3).
    The query format is a 3x3 rotation matrix, flattened and then positionally
    encoded in each component with multiple frequencies of sinusoids.
    Args:
        resnet_depth: The depth of feature extractor model (pretrained ResNet backbone).
        feat_dim: The length of the visual description vector, which
            is returned with the vision model at creation.
        num_fourier_comp: The number of positional encoding frequencies for
            the rotation query.  If zero, positional encoding is not used.
        mlp_layer_sizes: A list of the number of units in each layer of the MLP.
        so3_sampling_mode: 'random' or 'grid'.  This is only efficacious during
            training, and determines how the normalization is computed.  'random'
            corresponds to sampling uniformly over the space of rotations, which is
            simple and less constrained but results in inexact normalization.  'grid'
            uses equivolumetric grids generated using healpix as a starting point.
            These grids are stored so each size only needs to be computed once per run.
        num_train_queries: The number of queries to use during training, which
            populate SO(3) to approximate the normalization of the likelihoods.  If
            so3_sampling_mode is 'grid', the nearest grid size in log space is used.
        num_eval_queries: The number of queries to use during evaluation, which
            is always a grid sampling (for proper normalization).
    """
    def __init__(self, feature_extractor,feat_dim, num_fourier_comp=0, mlp_layer_sizes=[256]*4,
                 so3_sampling_mode='grid', num_train_queries=2**15, num_eval_queries=2**16):
        super(ImplicitSO3, self).__init__()
        self.feat_dim = feat_dim
        self.rotation_dim = 9 # Rotations are represented as flattened 3x3 orthonormal matrices.
        self.query_dim = self.rotation_dim
        self.rot_query_dim = self.rotation_dim if (num_fourier_comp == 0) else (self.rotation_dim * num_fourier_comp * 2)
        self.num_fourier_comp = num_fourier_comp
        self.so3_sampling_mode = so3_sampling_mode
        self.num_train_queries = num_train_queries
        self.num_eval_queries = num_eval_queries

        frequencies = torch.arange(self.num_fourier_comp, dtype=torch.float32)
        self.frequencies = torch.pow(2., frequencies)   

        self.grids = {} # populated on-demand.
        if so3_sampling_mode == 'grid':
            self._get_closest_avail_grid(self.num_train_queries)
        self._get_closest_avail_grid(self.num_eval_queries)

        # ResNet + MLP
        self.feature_extractor = feature_extractor
        self.FC_vis_emb = nn.Linear(self.feat_dim, mlp_layer_sizes[0])
        self.FC_pose_query_emb = nn.Linear(self.rot_query_dim, mlp_layer_sizes[0])
        self.mlp_layers = nn.ModuleList()
        for i in range(1, len(mlp_layer_sizes)):
            self.mlp_layers.append(nn.Linear(mlp_layer_sizes[i-1], mlp_layer_sizes[i]))
            self.mlp_layers.append(nn.ReLU())
        self.mlp_layers.append(nn.Linear(mlp_layer_sizes[-1], 1))

    def forward(self, input_):
        """
        Arguments:
            input_: Tuple of    Image of shape [BS, Ch, H, W]
                                query_rotation of shape [BS, N, len_rotation]/ [BS, len_rotation]
                                    or [BS, N, 3, 3]/ [BS, 3, 3]
        Returns
            
                            
        """
        img, pose_query = input_
        if pose_query.shape[-1] != self.rot_query_dim:
            pose_query = torch.flatten(pose_query, start_dim=-2, end_dim=-1)
        feat_vec = self.feature_extractor(img)
        vis_emb = self.FC_vis_emb(feat_vec)
        pose_emb = self.FC_pose_query_emb(pose_query)        
        if len(vis_emb.shape) != len(pose_emb.shape):
            vis_emb = torch.unsqueeze(vis_emb, dim=1)
        out = F.relu(torch.add(vis_emb, pose_emb))
        for layer in self.mlp_layers:
            out = layer(out)
        return out

    def _get_closest_avail_grid(self, number_queries=None):
        """
        HEALPix-SO(3) is defined only on 72 * 8^N points.
        We find the closest valid grid size (in log space) to the requested size.
        The largest grid size we consider has 19M points.
        """
        if not number_queries:
            number_queries = self.num_eval_queries
        grid_sizes = 72 * 8**np.arange(7)
        size = grid_sizes[np.argmin(np.abs(np.log(number_queries) - np.log(grid_sizes)))]
        if self.grids.get(size) is not None:
            return self.grids[size]
        else:
            print(f'Using grid of size {size} instead of requested size {number_queries}.')
            self.grids[size] = generate_healpix_grid(size=size)
            return self.grids[size]
    
    def predict_probability(self, images, rotation_mat, training=False, query_rotations=None):
        """
        Predict the probabilities of rotations, given corresponding images.
        Args:
            images: A batch of images.
            rotation_mat: The rotation matrices of interest.
            training: True or False; determines how many queries to use for
                normalization and whether to use an equivolumetric grid.
        """
        #ipdb.set_trace()
        device = images.device
        mode = self.so3_sampling_mode if training else 'grid'
        if query_rotations == None:
            query_rotations = self._generate_queries(self.num_train_queries if training else self.num_eval_queries, mode).to(device)
            rotation_mat = torch.reshape(rotation_mat, (-1, 3, 3))
            # rotate the grids of query_rotations s.t. the last matrix of
            # each batch is the corresponding rotation of interest
            delta_rot = torch.t(query_rotations[-1]) @ rotation_mat
            query_rotations = torch.einsum('aij,bjk->baik', [query_rotations, delta_rot])
        if len(query_rotations.shape)==3:
            query_rotations = torch.repeat_interleave(query_rotations.unsqueeze(dim=0), images.shape[0], dim=0)
        query_rotations = torch.reshape(query_rotations, (query_rotations.shape[0], query_rotations.shape[1], self.rotation_dim))
        query_rotations = self._positional_encoding(query_rotations)

        
        logits = torch.squeeze(self.forward((images, query_rotations)), dim=-1)
        probabilities = torch.nn.Softmax(dim=-1)(logits)
        # scale by the volume per grid point.
        probabilities = probabilities * (query_rotations.shape[1] / np.pi**2)
        # probabilities of the rotations of interest
        return probabilities[:, -1]


    def predict_rotation(self, images, gradient_ascent=False, pc_can=None, pc_obs=None):
        """
        Returns predicted rotations for a batch of images.
        The mode of the distribution is approximated, found either as the argmax
        over a set of samples, or by running gradient ascent on the probability with
        the sample argmax as the starting point.
        Args:
            images: A batch of images on which to estimate the pose.
            gradient_ascent: True or False; whether to perform gradient ascent after
                finding the argmax over the sample rotations, to squeeze out a little
                more performance.
        Returns:
            A tensor of rotations of shape BSx3x3.
        """
        device = images.device
        query_rotations = self._generate_queries(2**15, mode='grid')
        query_rotations = query_rotations.to(device)
        batch_size = images.shape[0]
        query_rot_inp = self._positional_encoding(query_rotations)
        query_rot_inp = torch.repeat_interleave(query_rot_inp.unsqueeze(dim=0), batch_size, dim=0)

        logits = torch.squeeze(self.forward((images, query_rot_inp)), dim=-1)
        max_ind = torch.max(logits, 1)
        max_ind = torch.reshape(max_ind[1], [batch_size])

        max_rotations = torch.index_select(query_rotations, 0, max_ind)

        if not gradient_ascent:
            max_rotations = torch.reshape(max_rotations, [-1, 3, 3])
            return max_rotations
        
        initial_step_size = step_size = 1e-3
        #step_size_list = [1e-2, 1e-3, 1e-4]
        #step_size = step_size_list[0]
        decay_param = 1
        #step_size_scheduler = lambda x: initial_step_size * 1/(1+decay_param*x)
        num_iteration = 70
        query_rot_quat= tt.matrix_to_quaternion(max_rotations)
        query_rot_quat.requires_grad = True
        #query_rot_quat = torch.autograd.Variable(query_rot_quat, requires_grad=True )
        
        def _grad_asc_step(query_rot_quat):
            
            inp = tt.quaternion_to_matrix(query_rot_quat)
            #dist = SMDist(pc_can, pc_obs, inp)
            #print("Distances: ", dist)
            #inp = torch.repeat_interleave(inp.unsqueeze(dim=0), batch_size, dim=0)
            inp = self._positional_encoding(inp)
            logits = torch.squeeze(self.forward((images, inp)), dim=-1)

            loss = -torch.mean(logits)
            grads = torch.autograd.grad(loss, query_rot_quat)[0]
            with torch.no_grad():
                query_rot_quat += -grads*step_size
                query_rot_quat = F.normalize(query_rot_quat)
            return
            
        for _ in range(num_iteration):
            #step_size = step_size_scheduler(_)
            #print(f"Step Size: {str(step_size)}")
            """if num_iteration==10:
                step_size = step_size_list[1]
            if num_iteration==50:
                step_size = step_size_list[2]"""
            _grad_asc_step(query_rot_quat)
        
        max_rotations = tt.quaternion_to_matrix(query_rot_quat)

        return max_rotations


    def _generate_queries(self, number_queries, mode='random'):
        """Generate query rotations from SO(3) using an equivolumetric grid or 
        an uniform distribution over SO(3).

        Arguments:
            number_queries: The number of queries.
            mode: 'random' or 'grid'; determines whether an uniform distribution over SO(3)
                or an equivolumetric grid is used
        Returns:
            A tensor of roation metrices of shape [number_queries, 3, 3].
        """
        if mode == 'random':
            return self._generate_queries_random(number_queries)
        else:
            assert mode == 'grid'
            return self._get_closest_avail_grid(number_queries)

    def _generate_queries_random(self, number_queries):
        """Generate rotation matrices uniformly distributed over SO(3) at random.

        Arguments:
            number_queries: The number of queries

        Returns:
            A tensor of roation metrices of shape [number_queries, 3, 3].
        """
        random_rotations = tt.random_rotations(number_queries, dtype=torch.float32)
        return random_rotations

    def _positional_encoding(self, query_rotations):
        """This handles the positional encoding.
            Arguments:
                query_rotations: tensor of shape [N, len_rotation] / [bs, N, len_rotation]
                    or [N, 3, 3] / [bs, N, 3, 3]
            Returns:
                Tensor of shape [N, rot_query_dim] or [bs, N, rot_query_dim].
        """
        #ipdb.set_trace()
        if query_rotations.shape[-1] != self.rotation_dim:
            query_rotations = torch.flatten(query_rotations, -2, -1)

        if self.num_fourier_comp == 0:
            return query_rotations

        #ipdb.set_trace()
        
        def func(freq):
            return torch.cat([torch.sin(query_rotations* freq), torch.cos(query_rotations*freq)], -1)

        pos_enc = []
        for freq in self.frequencies:
            pos_enc.append(func(freq))

        query_rotations = torch.cat(pos_enc, dim=-1)
        
        
        """
        qshape = query_rotations.shape
        if len(qshape) == 4:
            query_rotations = query_rotations.permute((1,2,0,3))
            query_rotations = torch.reshape(query_rotations, [qshape[1],qshape[2], self.rot_query_dim])
        else:
            query_rotations = query_rotations.permute((1,0,2))
            query_rotations = torch.reshape(query_rotations, [-1, self.rot_query_dim])
        """
        return query_rotations


    def compute_loss(self, images, rot_mat_gt):
        """Return the negative log likelihood of the ground truth rotation matrix.
            Argumentss:
                images: A batch of images.
                rot_mat_gt: The ground truth rotation matrices associated with the
                    batch of images.
            Returns:
                A scalar, the loss of the batch.
        """

        probabilities = self.predict_probability(images, rot_mat_gt, training=True)
        
        loss = -torch.mean(torch.log(probabilities))
        return loss


    def output_pdf(self, images, num_queries=None, query_rotations=None):
        """
        Returns a normalized distribution over pose, given a batch of images.
        Arguments:
            images: A batch of images on which to estimate the pose.
            num_queries: The number of queries to evaluate the probability for.
            query_rotations: If supplied, these rotations will be used to evaluate the
                distribution and normalize it, instead using the kwarg num_queries.
        Returns:
            Both the rotations and their corresponding probabilities.
        """ 
        with torch.no_grad():
            device = images.device
            if num_queries is None:
                num_queries = self.num_eval_queries
            if query_rotations is None:
                query_rotations = self._get_closest_avail_grid(num_queries)
                query_rotations = query_rotations.to(device).float()
            batch_size = images.shape[0]
            rot_inp = self._positional_encoding(query_rotations)
            rot_inp = torch.repeat_interleave(rot_inp.unsqueeze(dim=0), batch_size, dim=0)
            logits = torch.squeeze(self.forward((images, rot_inp)), dim=-1)
            probabilities = torch.nn.Softmax(dim=-1)(logits)
       
        return query_rotations, probabilities
