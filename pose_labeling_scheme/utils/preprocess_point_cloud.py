import open3d as o3d


def preprocess_point_cloud(point_cloud, size, transformation=None, verbose=False):
        """Preprocess the point cloud saved as a tensor and construct a Open3d point cloud structure for
        the open3d pipeline. The point cloud is downsampled, normals are approximated and the FPFH features are computed.
        
        """
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud.cpu().numpy())

        pcd_down = pcd.voxel_down_sample(size)

        if transformation is not None:
                pcd_down.transform(transformation.cpu().numpy())

        # Define normals
        radius_normal = size * 2

        pcd_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        # Compute features in 33-dimensional FPFH-feature space
        radius_feature = size * 5
        if verbose:
                print(":: Estimate normal with search radius %.3f." % radius_normal)
                print(":: Compute FPFH feature with search radius %.3f." % radius_feature)

        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                pcd_down,
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh
