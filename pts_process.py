# %%
import struct
import numpy as np
from tqdm import tqdm

# %%
finput = 'points.bin'

vertices = []

with open(finput, 'rb') as file:
    # Read the number of vertices
    num_vertices = struct.unpack('i', file.read(4))[0]

    # Read the vertices' coordinates
    for _ in range(num_vertices):
        coords = struct.unpack('fff', file.read(12))
        vertices.append(coords)
print(f'{num_vertices=}')
vertices[:10]

# %%
finput1 = 'tris.bin'
tris = []

with open(finput1, 'rb') as file:
    # Read the number of tris
    num_tris = struct.unpack('i', file.read(4))[0]

    # Read the triangle indices
    for _ in range(num_tris):
        tri = struct.unpack('iii', file.read(12))
        tris.append(tri)
print(f'{num_tris=}')
tris[:100]

# %%
# pre process data
pt_tris = {}
for i in range(len(vertices)):
    pt_tris[i] = []

for t in range(len(tris)):
    tri = tris[t]
    pt_tris[tri[0]].append(t)
    pt_tris[tri[1]].append(t)
    pt_tris[tri[2]].append(t)

pt_tris, len(pt_tris)

# %%
sorted_pts = []
sorted_dict = {}

eps = 0.05

# %%
def get_neibors(pt_index):
    nbs = []
    tri_ids = pt_tris[pt_index]
    def is_nb(idx):
        return idx != pt_index and idx not in nbs
    for t in tri_ids:
        tri = tris[t]
        if is_nb(tri[0]):
            nbs.append(tri[0])
        if is_nb(tri[1]):
            nbs.append(tri[1])
        if is_nb(tri[2]):
            nbs.append(tri[2])
    return nbs

def dis(pos1, pos2):
    v = []
    v.append(pos1[0] - pos2[0])
    v.append(pos1[1] - pos2[1])
    v.append(pos1[2] - pos2[2])
    return (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) ** 0.5

def sort_pts(start_index):

    count = 0
    selected = start_index
    while selected >= 0:        
        nbs = get_neibors(selected)

        pt_pos = vertices[selected]
        nbs_live = []
        nbs_y = []
        for nb in nbs:
            if nb in sorted_dict:
                continue

            nb_pos = vertices[nb]
            # d = dis(pt_pos, nb_pos)
            nbs_y.append(nb_pos[1])
            nbs_live.append(nb)
            
        if len(nbs_live) == 0:
            break

        nbs_live = np.array(nbs_live)
        nbs_y = np.array(nbs_y)

        sort_idx = np.argsort(nbs_y)
        nbs_live_sorted = np.take(nbs_live, sort_idx)
        # nbs_y_sorted = np.take(nbs_y, sort_idx)

        selected = nbs_live_sorted[0]
        # if len(nbs_dis_sorted) > 1:
        #     if abs(nbs_dis_sorted[1] - nbs_dis_sorted[0]) < eps:
        #         nb_pos0 = vertices[nbs_live_sorted[0]]
        #         nb_pos1 = vertices[nbs_live_sorted[1]]
        #         selected = nbs_live_sorted[0] if nb_pos0[1] <= nb_pos1[1] else nbs_live_sorted[1]
        # mark self is selected
        sorted_pts.append(selected)
        sorted_dict[selected] = 1
        count += 1
        # print(f'{selected}', end=' ', flush=True)
        # if count % 64 == 0:
        #     print('\n', flush=True)
                
    return count
            

# %%

sorted_pts.append(0)
sorted_dict[0] = 1

num_sorted = 0
start_from = 0
offset = -1
interval = 0
v_limit = 970000    # num_vertices, early quit to debug
while num_sorted < v_limit:
    just_sorted = sort_pts(start_from)
    if just_sorted == 0:
        start_from = sorted_pts[offset]
        offset -= 1
    else:
        offset = -1
    num_sorted += just_sorted
    interval += 1
    if interval % 1000 == 0:
        print(f'{num_sorted}', end=' ', flush=True)
    if interval % 64000 == 0:
        print('\n')


# %%
with open('outputs.bin', 'wb') as f:
    f.write(struct.pack('i', len(sorted_pts))) 
    for pt in sorted_pts:
        f.write(struct.pack('i', pt))


