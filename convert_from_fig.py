from scipy.io import loadmat
import numpy as np
from rve import RVE
import torch
from scipy.interpolate import interp1d

file_path = "data/Mock-FibTraIncCol_mock.fig"
# Load figure data (with helpful struct unpacking)
data = loadmat(file_path, squeeze_me=True, struct_as_record=False)
fig_struct = data['hgS_070000']

def extract_lines(hg_obj):
    """Recursively extract all line (X,Y,Z) data from MATLAB HG structures."""
    results = []
    if hasattr(hg_obj, 'children'):
        for child in np.atleast_1d(hg_obj.children):
            results.extend(extract_lines(child))
    if hasattr(hg_obj, 'properties'):
        props = hg_obj.properties
        if hasattr(props, 'XData') and hasattr(props, 'YData'):
            x = np.array(props.XData).flatten()
            y = np.array(props.YData).flatten()
            z = np.array(props.ZData).flatten() if hasattr(props, 'ZData') else None
            results.append((x, y, z))
    return results

def resample_line(x, y, z, resolution):
    t = np.linspace(0, 1, len(x))
    t_new = np.linspace(0, 1, resolution)
    fx, fy, fz = interp1d(t, x), interp1d(t, y), interp1d(t, z)
    return np.column_stack([fx(t_new), fy(t_new), fz(t_new)])

# Extract all lines
lines = extract_lines(fig_struct)
lengths = [len(x) for x, _, _ in lines]
resolution = max(lengths)

# Stack each line into (resolution, 3)
line_arrays = [np.column_stack([x, y, z]) if len(x) == resolution else resample_line(x, y, z, resolution) for x, y, z in lines]

# Convert to a single tensor
fibre_coords = torch.tensor(np.stack(line_arrays))  # shape: (N, resolution, 3)
print(fibre_coords.shape)

# Create dummy RVE
rve = RVE.dummy(fibre_coords)
rve.save("airtex", 0, 0)