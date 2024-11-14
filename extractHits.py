import uproot
import awkward as ak

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

def printHitsInFrame(filename, frame_number):
    inputFile = uproot.open(filename)
    mu3eTree = inputFile['mu3e'].arrays()
    mu3eFrame = ak.Array([mu3eTree[frame_number]])
    #mu3eFrameArray = ak.Array([mu3eFrame])

    #print(mu3eTree)
    # print(type(mu3eTree))
    # print()
    # print(type(mu3eFrame))
    print(mu3eFrame)
    # print()
    # print(mu3eFrameArray)
    # print(type(mu3eFrameArray))

    for hitsInFrame in mu3eFrame['hit_pixelid']:
        for hitIndex in hitsInFrame:
            hit = Hit(hitIndex)
            print(hit)
        break

# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) == 1:
#         print("You need to specify an input file(s)")
#         sys.exit(-1)

    # for argument in sys.argv[1:]:
    #     print("File:", argument)
    #     printHitsInFirstFrame(argument)


# Inputting a file and testing the output
file_path = "/root/Mu3eProject/RawData/HitData/signal1_1_1944629_execution_1_run_num_836827_sort.root"
frame_number = 1 # Choose which frame you want the hit info for

print()
print("Test with the file:", file_path) 
print("Frame number:", frame_number)
print()
printHitsInFrame(file_path, frame_number)
