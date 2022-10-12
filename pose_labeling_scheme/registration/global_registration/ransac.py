import open3d as o3d

def ransac_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    threshold = 1.5 * voxel_size

    # Run the RANSAC-algorithm
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source=source_down, target=target_down, source_feature=source_fpfh, target_feature=target_fpfh, mutual_filter=True,
        max_correspondence_distance=threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        #estimation_method=o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
        #estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        ransac_n=5,
        checkers= [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                threshold)],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.9))
    return result