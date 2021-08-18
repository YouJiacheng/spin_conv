import torch
import torch.nn.functional as F

# stolen from PyTorch3D, but for right multiplication
def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    r11 = 1 - two_s * (j * j + k * k)
    r12 = two_s * (i * j - k * r)
    r13 = two_s * (i * k + j * r)
    r21 = two_s * (i * j + k * r)
    r22 = 1 - two_s * (i * i + k * k)
    r23 = two_s * (j * k - i * r)
    r31 = two_s * (i * k - j * r)
    r32 = two_s * (j * k + i * r)
    r33 = 1 - two_s * (i * i + j * j)

    "for column vector, i.e. matrix @ vector"
    # o = torch.stack(
    #     (
    #         r11, r12, r13,
    #         r21, r22, r23,
    #         r31, r32, r33,
    #     ),
    #     -1,
    # )

    "for row vector, i.e. vector @ matrix"
    o = torch.stack(
        (
            r11, r21, r31,
            r12, r22, r32,
            r13, r23, r33,
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

# stolen from PyTorch3D
def axis_angle_to_quaternion(axis_angle):
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions

# stolen from PyTorch3D
def axis_angle_to_matrix(axis_angle):
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def rotation_to_z(target: torch.Tensor):
    """
    Compute right multiplication rotation matrices which rotate target vectors to z-axis
    i.e. target @ ret_mat along z axis
    Args:
        target: tensor of shape (..., 3)
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    target_unit = F.normalize(target, p=2, dim=-1)  # (..., 3)
    z_unit = torch.tensor([0, 0, 1], dtype=target_unit.dtype).expand(target_unit.shape)  # (..., 3)

    axis = F.normalize(torch.cross(target_unit, z_unit, dim=-1), p=2, dim=-1)  # (..., 3)
    angle = target_unit[..., -1:].acos()  # (..., 1)
    return axis_angle_to_matrix(axis * angle)  # (..., 3, 3)


def repeat_tuple(value, n: int):
    return (value, ) * n


if __name__ == '__main__':
    pass
    x = torch.tensor([[[2, 3, 3]] * 3] * 2, dtype=torch.float32)
    # x = torch.tensor([2, 3, 3], dtype=torch.float32)
    m = rotation_to_z(x)
    print(x.unsqueeze(-2)@m)
