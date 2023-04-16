import numpy as np


def create_tiling(feat_range, bins, offset, i):
    w = (feat_range[1] - feat_range[0]) / bins
    return np.linspace(feat_range[0], feat_range[1], bins + 1)[1:-1] + (i+1) * (w / number_tilings) * offset


def create_tilings(feature_ranges, number_tilings, bins, displacement):
    tilings = []
    for tile in range(number_tilings):
        tiling = []
        for i, feature in enumerate(feature_ranges):
            tiles = create_tiling(feature, bins[i], displacement[i], tile)
            tiling.append(tiles)
        tilings.append(tiling)
    return np.asarray(tilings)


def tile_coding(feature, tilings):
    codings = []
    for tiling in tilings:
        coding = []
        for i, feat in enumerate(feature):
            code = np.digitize(feat, tiling[i])
            coding.append(code)
        codings.append(coding)
    return np.asarray(codings)


feature_ranges = [[-1.2, 0.6], [-0.07, 0.07]]#, [-1, 0, 1]]
number_tilings = 4
bins = [20, 20]#, 20]
displacement = [1, 3]#, 1]

tilings = create_tilings(feature_ranges, number_tilings, bins, displacement)

feature = [0.2, 0.01]#, 0]
print(tile_coding(feature, tilings))
