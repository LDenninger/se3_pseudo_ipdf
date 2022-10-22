
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import generate_cartesian_grid
from .backbones import ResNet



class ImplicitTranslation(nn.Module):
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
    def __init__(self, resnet_depth, feat_dim, num_fourier_comp=0,
                resnet_layer=0,
                mlp_layer_sizes=[256]*2,
                num_train_queries=2**12, num_eval_queries=80000,
                translation_range=torch.tensor([[-0.1,0.1],[-0.1,0.1],[2.4,2.6]])):
        super(ImplicitTranslation, self).__init__()
        self.feat_dim = feat_dim
        self.translation_dim = 3
        self.query_dim = self.translation_dim #The queries consist of a flattened 3x3 orthonormal matrices and 1x3 translation vector
        self.trans_query_dim = self.translation_dim if (num_fourier_comp == 0) else (self.translation_dim * num_fourier_comp * 2)
        self.num_fourier_comp = num_fourier_comp
        self.num_train_queries = num_train_queries
        self.num_eval_queries = num_eval_queries
        self.translation_range = translation_range

        frequencies = torch.arange(self.num_fourier_comp, dtype=torch.float32)
        self.frequencies = torch.pow(2., frequencies)   

        self.grids = {} # populated on-demand.
        self._generate_queries(self.num_train_queries)
        self._generate_queries(self.num_eval_queries)

        # ResNet + MLP
        self.feature_extractor = ResNet(depth=resnet_depth, layer=resnet_layer, pretrained=True) 
        self.FC_vis_emb = nn.Linear(self.feat_dim, mlp_layer_sizes[0])
        self.FC_translation_query_emb = nn.Linear(self.trans_query_dim, mlp_layer_sizes[0])

        self.mlp_layers_translation= nn.ModuleList()
        
        for i in range(1, len(mlp_layer_sizes)):
            self.mlp_layers_translation.append(nn.Linear(mlp_layer_sizes[i-1], mlp_layer_sizes[i]))
            self.mlp_layers_translation.append(nn.ReLU())
        self.mlp_layers_translation.append(nn.Linear(mlp_layer_sizes[-1], 1))
        
    def forward(self, input_):
        """
        Arguments:
            input_: Tuple of    Image of shape [BS, Ch, H, W]
                                query_matrices of shape [BS, N, len_query]/ [BS, len_query]
                                    or [BS, N, 4, 3]/ [BS, 4, 3]
        Returns
            
                            
        """

        img, pose_query = input_

        feat_vec = self.feature_extractor(img)
        vis_emb = self.FC_vis_emb(feat_vec)
        trans_emb = self.FC_translation_query_emb(pose_query)        
        if len(vis_emb.shape) != len(trans_emb.shape):
            vis_emb = torch.unsqueeze(vis_emb, dim=1)

        out = F.relu(torch.add(vis_emb, trans_emb))
        for layer in self.mlp_layers_translation:
            out = layer(out)
        return out

    
    
    def predict_probability(self, images, translation, training=False):
        """
        Predict the probabilities of rotations, given corresponding images.
        Args:
            images: A batch of images.
            input_matrix: The matrices of interest consisting of a 3x3 rotation matrix and 1x3 transaltion vector.
            training: True or False; determines how many queries to use for
                normalization and whether to use an equivolumetric grid.
            query_rotations. Pre-defined rotations that should be used for normalization
        """

        device = images.device
        query_translation = self._generate_queries(self.num_train_queries if training else self.num_eval_queries).to(device)

        #Replace last translation with the ground truth translation
        query_translation = torch.repeat_interleave(query_translation.unsqueeze(0), translation.shape[0], dim=0)
        query_translation = torch.cat([query_translation, translation.unsqueeze(1).to(device)], dim=1)
        #query_matrices[:,-1,-1] = transformation_matrix[:,-1]

    
        query_translation = self._positional_encoding(query_translation)

        
        logits = self.forward((images, query_translation))
        logits = logits.squeeze(-1)

        probabilities = torch.nn.Softmax(dim=-1)(logits)

        # scale by the volume per grid point.
        probabilities = probabilities * (query_translation.shape[1] /(1**3))

        
        # probabilities of the rotations of interest
        return probabilities[:, -1]


    def predict_translation(self, images, gradient_ascent=False):
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
        query_translation = self._generate_queries(2**15)
        query_translation = query_translation.to(device)
        batch_size = images.shape[0]
        query_rot_inp = self._positional_encoding(query_translation)
        query_rot_inp = torch.repeat_interleave(query_rot_inp.unsqueeze(dim=0), batch_size, dim=0)

        logits = torch.squeeze(self.forward((images, query_rot_inp)), dim=-1)
        max_ind = torch.max(logits, 1)
        max_ind = torch.reshape(max_ind[1], [batch_size])

        max_rotations = torch.index_select(query_translation, 0, max_ind)

        if not gradient_ascent:
            return max_rotations
        step_size = 1e-3
        num_iteration = 30
        query_translation = max_rotations
        query_translation.requires_grad = True
        #query_rot_quat = torch.autograd.Variable(query_rot_quat, requires_grad=True )

        def _grad_asc_step(query_translation):
            inp = self._positional_encoding(query_translation)
            logits = torch.squeeze(self.forward((images, inp)), dim=-1)

            loss = -torch.mean(logits)
            grads = torch.autograd.grad(loss, query_translation)[0]
            
            with torch.no_grad():
                query_translation += -grads*step_size
            
            return
        
        for _ in range(num_iteration):
            _grad_asc_step(query_translation)

        return query_translation.detach()
    def _generate_queries(self, number_queries=None):
        """Generate query matrices from SE(3) using an equivolumetric grid or 
        an uniform distribution over SO(3) and a cartesian grid for the translation component.
        Arguments:
            number_queries: The number of queries.
            mode: 'random' or 'grid'; determines whether an uniform distribution over SO(3)
                or an equivolumetric grid is used
            rotation_only: Determines if the returned queries are elements from SO(3) or SE(3)
            training: Determines the number of translation queries generated by the cartesian grid
        Returns:
            A tensor of roation metrices of shape [number_queries, 4, 3] or [number_queries, 3, 3].
        """
        if number_queries is None:
            number_queries = self.num_eval_queries
        if self.grids.get(number_queries) is not None:
            return self.grids[number_queries]

        translation_query = generate_cartesian_grid(self.translation_range, size=number_queries)
        #translation_query = self.scale_translation(translation_query)
        #trans_test = self.scale_to_original(translation_query)

        #Append the points of the cartesian grid randomly to the rotation queries
        self.grids[number_queries] = translation_query
        return self.grids[number_queries]

    def scale_translation(self, grid):
        """Scale the translation vectors to the range of [-1,1]
        """

        new_max = 1
        new_min = -1

        old_max = self.translation_range[:,1]
        old_min = self.translation_range[:,0]

        grid[:,0] = (new_max-new_min)/(old_max[0]-old_min[0])* (grid[:,0]-old_min[0]) + new_min 
        grid[:,1] = (new_max-new_min)/(old_max[1]-old_min[1])* (grid[:,1]-old_min[1]) + new_min 
        grid[:,2] = (new_max-new_min)/(old_max[2]-old_min[2])* (grid[:,2]-old_min[2]) + new_min 

        return grid

    def scale_to_original(self, grid):
    
        old_max = 1
        old_min = -1

        new_max = self.translation_range[:,1]
        new_min = self.translation_range[:,0]

        grid[:,0] = (new_max[0]-new_min[0])/(old_max-old_min)* (grid[:,0]-old_min) + new_min[0]
        grid[:,1] = (new_max[1]-new_min[1])/(old_max-old_min)* (grid[:,1]-old_min) + new_min[1]
        grid[:,2] = (new_max[2]-new_min[2])/(old_max-old_min)* (grid[:,2]-old_min) + new_min[2] 

        return grid

    def _positional_encoding(self, query_translation):
        """This handles the positional encoding.
            Arguments:c
                query_rotations: tensor of shape [N, len_query] / [bs, N, len_query]
                    or [N, 4, 3] / [bs, N, 4, 3]
            Returns:
                Tensor of shape [N, rot_query_dim] or [bs, N, rot_query_dim].
        """
        if query_translation.shape[-1] != self.query_dim:
            query_translation = torch.flatten(query_translation, -2, -1)

        if self.num_fourier_comp == 0:
            return query_translation

        #ipdb.set_trace()
        
        def func(freq, matrix):
            return torch.cat([torch.sin(matrix* freq), torch.cos(matrix*freq)], -1)

        trans_enc = []
        for freq in self.frequencies:
            trans_enc.append(func(freq, query_translation))


        query_translation = torch.cat(trans_enc, dim=-1)
      
        
        """
        qshape = query_rotations.shape
        if len(qshape) == 4:
            query_rotations = query_rotations.permute((1,2,0,3))
            query_rotations = torch.reshape(query_rotations, [qshape[1],qshape[2], self.rot_query_dim])
        else:
            query_rotations = query_rotations.permute((1,0,2))
            query_rotations = torch.reshape(query_rotations, [-1, self.rot_query_dim])
        """
        return query_translation


    def compute_loss(self, images, translation_gt=None):
        """Return the negative log likelihood of the ground truth rotation matrix.
            Argumentss:
                images: A batch of images.
                rot_mat_gt: The ground truth rotation matrices associated with the
                    batch of images.
            Returns:
                A scalar, the loss of the batch.
        """
        probabilities = self.predict_probability(images, translation_gt, training=True)
        
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
                query_translation = self._generate_queries(num_queries).to(device)
                #query_matrices[:,-1,:] = torch.tensor([0,0,2.5])
            images = images.to(device)
            batch_size = images.shape[0]
            input = self._positional_encoding(query_translation)
            input = torch.repeat_interleave(input.unsqueeze(dim=0), batch_size, dim=0)
            logits = torch.squeeze(self.forward((images, input)), dim=-1)
            probabilities = torch.nn.Softmax(dim=-1)(logits)
       
        return query_translation, probabilities