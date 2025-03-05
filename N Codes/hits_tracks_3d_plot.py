import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd 
    

# Function to rotate data so the z-axis becomes horizontal (aligned with x-axis)
def rotate_axes(x, y, z):
    return z, y, -x  # Swap and invert to rotate the system


def plot_cylinder(ax, radius, length, num_points, color='black', alpha=0.1):
    """Plots a hollow cylinder (without caps) centered at the origin along the Z-axis."""
    theta = np.linspace(0, 2 * np.pi, num_points)
    z = np.linspace(-length / 2, length / 2, num_points)
    theta, z = np.meshgrid(theta, z)

    # Cylinder coordinates
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    # Rotate to align along X-axis (instead of default Z-axis)
    x_rot, y_rot, z_rot = rotate_axes(x, y, z)

    # Plot the surface
    ax.plot_surface(x_rot, y_rot, z_rot, color=color, alpha=alpha, edgecolor='none')
    #ax.plot_surface(x, y, z, color=color, alpha=alpha, edgecolor='none')

def plot_cones(ax, base_radius, height, num_points=50, color="gray", alpha=1):
    """
    Plots two cones symmetrically along the x-axis.

    Parameters:
        ax (matplotlib.axes._subplots.Axes3DSubplot): The 3D plot axis.
        base_radius (float): Radius of the circular base.
        height (float): Height of each cone.
        num_points (int, optional): Number of points for smoothness. Default is 50.
        color (str, optional): Color of the cones. Default is "gray".
        alpha (float, optional): Transparency of the cones. Default is 0.5.
    """

    # Generate a circular base
    theta = np.linspace(0, 2 * np.pi, num_points)
    x_circle = base_radius * np.cos(theta)
    y_circle = base_radius * np.sin(theta)
    z_circle = np.zeros_like(theta)  # Base at z = 0

    # Apex of the cones (one at +height, one at -height along the x-axis)
    cone1_apex = [0,0 ,height]   # Forward direction
    cone2_apex = [0,0, -height]  # Backward direction

    # Rotate points to align with the x-axis
    x_circle_rot, y_circle_rot, z_circle_rot = rotate_axes(x_circle, y_circle, z_circle)
    cone1_apex_rot = rotate_axes(*cone1_apex)
    cone2_apex_rot = rotate_axes(*cone2_apex)
    circle_points_rot = np.column_stack((x_circle_rot, y_circle_rot, z_circle_rot))
    circle_points = np.column_stack ((x_circle,y_circle,z_circle))

    # Plot the circular base
    ax.plot(x_circle_rot, y_circle_rot, z_circle_rot, color=color)

    #Plot the faces of the first cone
    for i in range(len(circle_points_rot) - 1):
        face = [cone1_apex_rot, circle_points_rot[i], circle_points_rot[i + 1]]
        ax.add_collection3d(Poly3DCollection([face], alpha=alpha, color=color))
    
    #Close the first cone
    ax.add_collection3d(Poly3DCollection([[cone1_apex_rot, circle_points_rot[-1], circle_points_rot[0]]], alpha=alpha, color=color))

    #Plot the faces of the second cone
    for i in range(len(circle_points_rot) - 1):
        face = [cone2_apex_rot, circle_points_rot[i], circle_points_rot[i + 1]]
        ax.add_collection3d(Poly3DCollection([face], alpha=alpha, color=color))

    #Close the second cone
    ax.add_collection3d(Poly3DCollection([[cone2_apex_rot, circle_points_rot[-1], circle_points_rot[0]]], alpha=alpha, color=color))


    # # UN-ROTATED

    # ax.plot(x_circle, y_circle, z_circle, color=color)
    # for i in range(len(circle_points_rot) - 1):
    #     face = [cone1_apex, circle_points[i], circle_points[i + 1]]
    #     ax.add_collection3d(Poly3DCollection([face], alpha=alpha, color=color))


    # ax.add_collection3d(Poly3DCollection([[cone1_apex, circle_points[-1], circle_points[0]]], alpha=alpha, color=color))

    # for i in range(len(circle_points) - 1):
    #     face = [cone2_apex, circle_points[i], circle_points[i + 1]]
    #     ax.add_collection3d(Poly3DCollection([face], alpha=alpha, color=color))

    # ax.add_collection3d(Poly3DCollection([[cone2_apex, circle_points[-1], circle_points[0]]], alpha=alpha, color=color))

def plot_helix(ax,track_params, track_number, colour, B= 1, num_points=5000, length_factor=5):
    """
    Plots a helical track for a charged particle in a magnetic field.

    Parameters:
    - ax: The 3D axis to plot the helix.
    - mc_p: Total momentum (MeV).
    - mc_pt: Transverse momentum (MeV).
    - mc_phi: Azimuthal angle (radians).
    - mc_lam: Lambda angle (radians).
    - mc_theta: Theta angle (radians).
    - mc_vx, mc_vy, mc_vz: Initial position (mm).
    - q: Charge of the particle (default is elementary charge in Coulombs).
    - B: Magnetic field strength (Tesla), default 1T for simplicity.
    - num_points: Number of points to plot the helix.
    - length_factor: Factor controlling the helix length.

    Returns:
    - Plots the helical track in the given 3D axis.
    """

    mc_type, mc_p, mc_pt, mc_phi, mc_lam, mc_theta, mc_vx, mc_vy, mc_vz, mc_vr = track_params

    if mc_type == 92:   #92 = electron in signal decay
        q_e = -1.6e-19
    elif mc_type == 91:  #91 = positron in signal decay
        q_e = 1.6e-19

    # Convert momentum from MeV/c to kg·m/s
    MeV_to_kgms = 5.344286e-22
    p_SI = mc_p * MeV_to_kgms
    pt_SI = mc_pt * MeV_to_kgms

    # Compute helix radius in meters
    r_helix_mm = pt_SI / (q_e * B) * 1e3


    # Convert radius to mm for consistency with detector dimensions
    #r_helix_mm = r_helix_m * 1e3
    #r_helix_mm = mc_vr
    pitch = r_helix_mm * np.tan(mc_lam)
   # Pitch (determines how much it moves in z per turn)
    # Generate helical trajectory

# Generate helix
# Generate helix parameter t
# Generate helix parameter t
    t = np.linspace(0, length_factor * np.pi, num_points)  # Parameter along the helix
    if mc_type == 92:
        t = np.linspace(0, -length_factor * np.pi, num_points)  # Parameter along the helix

# Ensure the helix starts at (mc_vx, mc_vy)
    x_helix = mc_vx + r_helix_mm * (np.cos(t) - 1) * np.cos(mc_phi + np.pi + np.pi/2) - r_helix_mm * np.sin(t) * np.sin(mc_phi + np.pi+np.pi/2)
    y_helix = mc_vy + r_helix_mm * (np.cos(t) - 1) * np.sin(mc_phi + np.pi+ np.pi/2) + r_helix_mm * np.sin(t) * np.cos(mc_phi + np.pi+ np.pi/2)
    z_helix = mc_vz + pitch * t


    # Apply rotation
    x_rot, y_rot, z_rot = rotate_axes(x_helix, y_helix, z_helix)
    mc_vx_rot, mc_vy_rot, mc_vz_rot = rotate_axes(mc_vx, mc_vy, mc_vz)


    print(f"Track {track_number}: mc_vx={mc_vx}, mc_vy={mc_vy}, mc_vz={mc_vz}")
    # print(f"Track {track_number}: mc_vx_rot={mc_vx_rot}, mc_vy_rot={mc_vy_rot}, mc_vz_rot={mc_vz_rot}")
    print(f"Track {track_number}: Expected Start ({mc_vx}, {mc_vy}, {mc_vz})")
    print(f"Track {track_number}: Actual Start ({x_helix[0]}, {y_helix[0]}, {z_helix[0]})")


    # Plot the helix
    #ax.scatter(mc_vx, mc_vy, mc_vz, color = "black", s=10, label=f"mc_vertex track{track_number}")
    ax.scatter(mc_vx_rot, mc_vy_rot, mc_vz_rot, color = "black", s = 10) #label = f"ROTATED vertex track {track_number}"
    ax.plot(x_rot, y_rot, z_rot, color=colour, linewidth=1, alpha = 0.5) #, label= f"Helix Track {track_number}"
    #ax.plot(x_helix, y_helix, z_helix, color="black", linewidth=2, label= f"Helix Track {track_number}")



# Trirec data for helix plotting
file_name = "/root/Mu3eProject/RawData/TrirecFiles/signal1_98_32652/trirec_data_signal1_98_32652_frames.csv"
trirec_track_data = pd.read_csv(file_name)

#Hits 3d data for hit point plotting
hits_file_name = "/root/Mu3eProject/RawData/HitData/hits_data_signal1_98_32652_with_mcinfo_global_testing_2.csv"
hits_data_global = pd.read_csv(hits_file_name)




######  Helix parameters (sort later) ################################

track_ids = [77700, 77701, 77702]  # List of track numbers
track_params = {}  # Dictionary to store track data

for track_id in track_ids:
    track = trirec_track_data[trirec_track_data["mc_tid"] == track_id]
    track_params[track_id] = track.iloc[0][["mc_type", "mc_p", "mc_pt", "mc_phi", 
                                                "mc_lam", "mc_theta", "mc_vx", "mc_vy", 
                                                "mc_vz", "mc_vr"]]



# # Helix Parameters
# mc_p = 34.51379776     # Total momentum (MeV)
# mc_pt = 25.93590736    # Transverse momentum (MeV)
# mc_phi = 1.017338276   # Azimuthal angle (radians)
# mc_lam = 0.720516682   # Lambda angle (radians)
# mc_theta = 0.850279689 # Theta angle (radians)
# mc_vx = -8.559747696   # Initial x-position (mm)
# mc_vy = 0.864106059    # Initial y-position (mm)
# mc_vz = 27.43914986    # Initial z-position (mm)



#########################################################################

######################### Detector setup, all coordinates in mm ##################################
# Central target
cone_z_length = 50  # Height of each cone
base_radius = 19  # Radius of the circular base
base_width = 38  # Diameter of the base

# Central barrel layers
# Layer 1
l1_z_length = 124.7
l1_radius = 23.3
# Layer 2
l2_z_length = 124.7
l2_radius = 29.8
# Layer 3
l3_z_length = 351.9
l3_radius = 73.9
#Layer 4
l4_z_length = 372.6
l4_radius = 86.3
####################################################################################################
##############  Actual hits points #############################
hits_data_track_1 = hits_data_global[hits_data_global["tid"] == track_ids[0]]          # Use if want to plot a single track
hits_data_track_coords_1 = hits_data_track_1[["gx", "gy", "gz"]]
hits_data_track_rot_1 = np.array([rotate_axes(gx, gy, gz) for gx, gy, gz in hits_data_track_coords_1.to_numpy()])

hits_data_track_2 = hits_data_global[hits_data_global["tid"] == track_ids[1]]          # Use if want to plot a single track
hits_data_track_coords_2 = hits_data_track_2[["gx", "gy", "gz"]]
hits_data_track_rot_2 = np.array([rotate_axes(gx, gy, gz) for gx, gy, gz in hits_data_track_coords_2.to_numpy()])

hits_data_track_3 = hits_data_global[hits_data_global["tid"] == track_ids[2]]          # Use if want to plot a single track
hits_data_track_coords_3 = hits_data_track_3[["gx", "gy", "gz"]]
hits_data_track_rot_3 = np.array([rotate_axes(gx, gy, gz) for gx, gy, gz in hits_data_track_coords_3.to_numpy()])

hits_data_size_cut = hits_data_global[hits_data_global["frameNumber"]<9000]
hits_data_global_coordinates = hits_data_size_cut[["gx", "gy", "gz"]]
hit_points_rot = np.array([rotate_axes(gx, gy, gz) for gx, gy, gz in hits_data_global_coordinates.to_numpy()])

##### For plotting mc points for verifying on target
# p_filter = 105
# mc_points_total = trirec_segs_data[["mc_vx","mc_vy","mc_vz","mc_p"]] 
# mc_points_filtered = mc_points_total[mc_points_total['mc_p']<=p_filter]
# mc_points_plotting = mc_points_filtered[["mc_vx", "mc_vy", "mc_vz"]].to_numpy()
#mc_points_rot = np.array([rotate_axes(x, y, z) for x, y, z in mc_points_plotting])

# x_helix_rot, y_helix_rot, z_helix_rot = rotate_axes(x_helix, y_helix, z_helix)



# Initialize the 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
#remove the axes
ax.set_axis_off()



# Plot the track points
#ax.scatter(mc_points_rot[:, 0], mc_points_rot[:, 1], mc_points_rot[:, 2], color="green", s=1e-5, label="Mc")
#ax.scatter(hit_points_rot[:, 0], hit_points_rot[:, 1], hit_points_rot[:, 2], color="red", s=0.01, label="All hits")


# #3 sets of track hits
# ax.scatter(hits_data_track_rot_1[:, 0], hits_data_track_rot_1[:, 1], hits_data_track_rot_1[:, 2],  color="green", s=10, label = f"hits track {track_ids[0]}")
# ax.scatter(hits_data_track_rot_2[:, 0], hits_data_track_rot_2[:, 1], hits_data_track_rot_2[:, 2], color="red", s=10,label = f"hits track {track_ids[1]}")
# ax.scatter(hits_data_track_rot_3[:, 0], hits_data_track_rot_3[:, 1], hits_data_track_rot_3[:, 2], color="blue", s=10,label = f"hits track {track_ids[2]}")
# # # #no labels
ax.scatter(hits_data_track_rot_1[:, 0], hits_data_track_rot_1[:, 1], hits_data_track_rot_1[:, 2],  color="green", s=5)
ax.scatter(hits_data_track_rot_2[:, 0], hits_data_track_rot_2[:, 1], hits_data_track_rot_2[:, 2], color="red", s=5)
ax.scatter(hits_data_track_rot_3[:, 0], hits_data_track_rot_3[:, 1], hits_data_track_rot_3[:, 2], color="blue", s=5)


#ax.scatter(hits_data_track_coords_69264["gx"], hits_data_track_coords_69264["gy"], hits_data_track_coords_69264["gz"], color="green", s=10, label="hits track 69264")

# Plot the 3 helices
plot_helix(ax, track_params[track_ids[0]],track_number = track_ids[0], colour = "green")
plot_helix(ax, track_params[track_ids[1]],track_number = track_ids[1], colour = "red")
plot_helix(ax, track_params[track_ids[2]],track_number = track_ids[2], colour = "blue")

# Plot detector layers and target
plot_cylinder(ax, l1_radius, l1_z_length, num_points = 9, color='blue', alpha=0.05)
plot_cylinder(ax, l2_radius, l2_z_length,num_points = 11, color='green', alpha=0.05)
plot_cylinder(ax, l3_radius, l3_z_length,num_points = 25, color='red', alpha=0.05)
plot_cylinder(ax, l4_radius, l4_z_length,num_points = 29, color='purple', alpha=0.05)
plot_cones(ax, base_radius= base_radius, height=cone_z_length, color="gray", alpha=0.5)

# Ensure labels don't repeat in the legend
handles, labels = ax.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
ax.legend(unique_labels.values(), unique_labels.keys())
# Set axes labels and limits
ax.set_xlabel("x")
ax.set_ylabel("Y")
ax.set_zlabel("z")
n=175
ax.set_xlim([-n, n])
ax.set_ylim([-n, n])
ax.set_zlim([-n, n])
#ax.set_box_aspect([2, 2, 1])  # Adjust aspect ratio for better proportions

location = "/root/Mu3eProject/3_helices_plot_view5.svg"
ax.view_init(elev=17, azim=125)  # Rotate by 180 degrees in azimuth

plt.savefig(location, transparent = True, bbox_inches='tight')


plt.show()




