import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    # tranformation process 
    # convert point to homo
    # implementation can be improved!!
    ans = []# normal np append will overwrite each time
    points = np.array(points)

    if points.ndim==1:
        # append dont update in place!!!
        points = np.append(points,1)
        T = np.array(T)
        newpoint = T@points
        return newpoint[:3] # since applied to 3 dimensional vector 
    else:
        for point in points:
        # append dont update in place!!!
            point = np.append(point,1)
            T = np.array(T)
            newpoint = T@point
            ans.append(newpoint[:3])
        return np.array(ans) # since applied to 3 dimensional vector  