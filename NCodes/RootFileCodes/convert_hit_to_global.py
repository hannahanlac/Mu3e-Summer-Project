import uproot
import pandas as pd
import numpy as np


class Hit(object):
    """Decodes a 32 bit hit ID (as found in the 'hit_pixelid' branch) into its constituent parts"""
    def __init__(self, hitIndex):
        self.hitIndex = hitIndex
    def __str__(self):
        return "%d (station %d, layer %d, ladder %d, chip %d, pixel [%d, %d])" % (self.hitIndex, self.station(), self.layer(), self.phi(), self.z(), self.x(), self.y())
    def chipid(self): return self.hitIndex >> 16
    def pixelid(self): return self.hitIndex
    def column(self):
        """The raw pixel column relative to the left side of the chip. Different layers orient chips differently, so "left" is not consistent."""
        return (self.hitIndex >> 8) & 0xFF
    def row(self):
        """The raw pixel row relative to the bottom of the chip. All layers orient rows in the opposite direction to phi."""
        return self.hitIndex & 0xFF
    def x(self):
        """The pixel position along the beam line, taking account of chip orientation. See section 1.1.2 of the specbook."""
        if self.layer() < 3: return 256 - self.column()
        else: return self.column()
    def y(self):
        """The pixel position around phi, taking account of chip orientation so that increasing y moves with increasing phi."""
        return 250 - self.row()
    def station(self):
        """The station the chip is in. 0 = central; 1 = upstream; 2 = downstream."""
        return int(self.chipid() / (0x1 << 12))
    def layer(self): return int((self.chipid() / (0x1 << 10)) % 4 + 1)
    def phi(self):
        """How far around in phi the chip is. Essentially which ladder the chip is on."""
        return int((self.chipid() / (0x1 << 5)) % (1 << 5) + 1)
    def z(self):
        """How far along the ladder the chip is. Higher number is further downstream."""
        zt = self.chipid() % (1<<5);
        if self.layer() == 3:
            return zt - 7
        elif self.layer() == 4:
            return zt - 6;
        else: return zt;

    def sensor_data(self, sensor_tree, entry_index):
        """Extract the 3D sensor data for a given entry index."""
        branches = ["vx", "vy", "vz", "rowx", "rowy", "rowz", "colx", "coly", "colz"]
        arrays = sensor_tree.arrays(branches)

        v = np.array([arrays["vx"][entry_index], arrays["vy"][entry_index], arrays["vz"][entry_index]])
        drow = np.array([arrays["rowx"][entry_index], arrays["rowy"][entry_index], arrays["rowz"][entry_index]])
        dcol = np.array([arrays["colx"][entry_index], arrays["coly"][entry_index], arrays["colz"][entry_index]])
        
        return v, drow, dcol    

def GlobalCalculator(v,drow,dcol, row_number, column_number):
    """Function that makes the final 3d coordinates of a hit from the """

    hit_global_coordinates = v + drow * (0.5 + row_number) + dcol * (0.5 + column_number)
    return hit_global_coordinates

def preprocess_chip_id_mapping(sensor_tree):
    """Creates a dictionary mapping the unique sensor id to the entry index in the sensors tree, for fast lookups."""
    id_branch = sensor_tree["sensor"].array()
    return {chip_id: idx for idx, chip_id in enumerate(id_branch)}


file = uproot.open("/root/Mu3eProject/RawData/v5.3/signal1_99_32652_execution_1_run_num_561343_sort.root")
mu3eTree = file['mu3e'].arrays(['hit_pixelid', 'hit_timestamp', 'hit_mc_i', 'hit_mc_n']) # Opens just the Mu3eTree branches we need
sensor_tree = file["alignment/sensors;1"]  # Replace with your actual sensor tree name

#Load id branch
id_branch = sensor_tree["sensor"].array()  # This gives you a numpy array of IDs in the 'id' branch

sensor_id_to_index = preprocess_chip_id_mapping(sensor_tree)


# hit_id = 697562353 # 32-bit hitID.
# hit = Hit(hit_id)
hit = Hit(393472) 
sensor_id = hit.chipid()
print(f"This hit has chip ID: {sensor_id}")

chip_id_index = sensor_id_to_index.get(sensor_id, None)

# Extract 3D sensor data
v, drow, dcol = hit.sensor_data(sensor_tree, chip_id_index)

# Compute 3D coordinates:
hit_global_coords = GlobalCalculator(v,drow,dcol, hit.row(), hit.column())
print(f"v:{v}")
print(f"drow{drow}")
print(f"dcol{dcol}")

print(f"Final 3D coordinates: {hit_global_coords}")


